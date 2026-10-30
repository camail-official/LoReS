"""Tune and independently time FP32/INT8 recurrent decode kernels."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch
import triton

from lores.backends.vllm.serving import source_identity, vllm_source
from lores.evaluation.output_store import create_run, sha256, write_json_once
from lores.kernels.state_int.packing.packed_state_kernel_naive import PackedState, pack_state, unpack_state

GRID3 = list(itertools.product([8, 16, 32, 64, 128], [1, 2, 4, 8], [1, 2, 3]))
GRID_M2 = list(itertools.product([2, 4, 8, 16, 32, 64], [1, 2, 4, 8]))


def cycle(items):
    iterator = itertools.cycle(items)
    return lambda: next(iterator)


def make_timer(n_buffers):
    def timed(fn):
        for _ in range(6):
            fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(n_buffers):
                fn()
        for _ in range(2):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repeats = max(1, 512 // n_buffers)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / (repeats * n_buffers)

    return timed


def shape_for(profile, arch):
    if arch == "mamba2":
        return dict(
            heads=128 if profile == "nemo" else 64,
            dim=64,
            state=128,
            groups=8 if profile == "nemo" else 1,
            conv_bytes=61440 if profile == "nemo" else 0,
        )
    return dict(
        heads=16 if profile == "qwen" else 32, value_heads=32, key=128, value=128, conv_bytes=49152 if profile == "qwen" else 0
    )


class Pools:
    """Cycle independent states with the model's padded cache-page strides."""

    def __init__(self, shape, layers, asym, conv_bytes, axis="value", group=None):
        from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

        self.seed = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1
        self.axis, self.group = axis, group
        if axis == "value":
            self.packed_seed = pack_state(self.seed, asym=asym)
            self.unpack = unpack_state
        else:  # KDA key axis: one scale per key channel per group of value rows
            codes, scales = quantize_key_axis(self.seed, asym, None, group)
            self.packed_seed = PackedState(codes, scales, shape, asym)
            self.unpack = lambda p: dequantize_key_axis(p.codes, p.scales, p.asym)
        self.states, self.packed = [], []
        self.page_bytes = {}
        for arm in ("fp32", "int8"):
            examples = [self.seed] if arm == "fp32" else [self.packed_seed.codes, self.packed_seed.scales]
            sizes = [x[0].numel() * x.element_size() for x in examples]
            page_bytes = conv_bytes + sum(sizes)
            if conv_bytes:
                page_bytes = triton.cdiv(page_bytes, 65536) * 65536
            self.page_bytes[arm] = page_bytes
            for _ in range(layers):
                storage = torch.empty((shape[0], page_bytes), device="cuda", dtype=torch.uint8)
                offset, views = conv_bytes, []
                for example, size in zip(examples, sizes):
                    views.append(storage[:, offset : offset + size].view(example.dtype).view(example.shape))
                    offset += size
                if arm == "fp32":
                    self.states.append(views[0])
                else:
                    self.packed.append(PackedState(*views, shape, asym))
            self.reset(arm)

    def scale_map(self, p):
        """Per-element scale of ``p`` (broadcast to the state shape), for the one-step error bound."""
        sc = p.scales[..., 0] if p.asym else p.scales
        if self.axis == "value":
            return sc.unsqueeze(-1).expand(p.shape)
        if self.group:
            return sc.repeat_interleave(self.group, dim=2)
        return sc.unsqueeze(2).expand(p.shape)

    def reset(self, arm):
        if arm == "fp32":
            for state in self.states:
                state.copy_(self.seed)
        else:
            for packed in self.packed:
                packed.codes.copy_(self.packed_seed.codes)
                packed.scales.copy_(self.packed_seed.scales)


def delta_case(arch, batch, shape, pools, args):
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd_kernel,
        fused_recurrent_gated_delta_rule_packed_decode_kernel,
    )

    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import (
        fused_recurrent_gated_delta_rule_packed_decode_int,
    )
    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int

    h, hv, k, v = (shape[name] for name in ("heads", "value_heads", "key", "value"))

    def rand(*dims):
        return torch.randn(dims, device="cuda", dtype=torch.bfloat16)

    idx = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    seeds = torch.arange(batch, device="cuda", dtype=torch.int32) + 1234 if args.state_sr else None
    out = torch.empty((batch, 1, hv, v), device="cuda", dtype=torch.bfloat16)
    if arch == "gdn":
        mixed = rand(batch, 2 * h * k + hv * v)
        ga, gb = rand(batch, hv), rand(batch, hv)
        a_log, dt_bias = rand(hv).float(), rand(hv).float()
    else:
        q, key, value = rand(batch, 1, h, k), rand(batch, 1, h, k), rand(batch, 1, hv, v)
        gate = torch.nn.functional.logsigmoid(torch.randn(batch, 1, hv, k, device="cuda"))
        beta = torch.rand(batch, 1, hv, device="cuda", dtype=torch.bfloat16)

    def fp_factory(cfg):
        bv, warps, stages = cfg
        nxt = cycle(pools.states)

        def fn():
            state = nxt()
            common = dict(
                o=out,
                h0=state,
                ht=state,
                ssm_state_indices=idx,
                scale=k**-0.5,
                stride_init_state_token=state.stride(0),
                stride_final_state_token=state.stride(0),
                stride_indices_seq=idx.stride(0),
                H=h,
                HV=hv,
                K=k,
                V=v,
                BK=triton.next_power_of_2(k),
                BV=bv,
                USE_QK_L2NORM_IN_KERNEL=True,
                num_warps=warps,
                num_stages=stages,
            )
            if arch == "gdn":
                fused_recurrent_gated_delta_rule_packed_decode_kernel[(triton.cdiv(v, bv), batch * hv)](
                    mixed_qkv=mixed,
                    a=ga,
                    b=gb,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    stride_mixed_qkv_tok=mixed.stride(0),
                    stride_a_tok=ga.stride(0),
                    stride_b_tok=gb.stride(0),
                    SOFTPLUS_THRESHOLD=20.0,
                    **common,
                )
            else:
                fused_recurrent_gated_delta_rule_fwd_kernel[(1, triton.cdiv(v, bv), batch * hv)](
                    q=q,
                    k=key,
                    v=value,
                    g=gate,
                    beta=beta,
                    cu_seqlens=None,
                    num_accepted_tokens=None,
                    N=batch,
                    T=1,
                    B=batch,
                    stride_indices_tok=1,
                    IS_BETA_HEADWISE=False,
                    INPLACE_FINAL_STATE=True,
                    IS_KDA=True,
                    **common,
                )
            return out

        return fn

    def int_factory(cfg):
        nxt = cycle(pools.packed)

        def fn():
            packed = nxt()
            common = dict(
                use_qk_l2norm_in_kernel=True, sr_seed=seeds, sr_salt=1, asym=packed.asym, fast=args.state_fast, launch=cfg
            )
            if arch == "gdn":
                fused_recurrent_gated_delta_rule_packed_decode_int(
                    mixed, ga, gb, a_log, dt_bias, k**-0.5, packed.codes, packed.scales, out, idx, **common
                )
                return out
            return fused_recurrent_kda_int(
                q, key, value, gate, beta, None, packed.codes, packed.scales, idx, axis=pools.axis, **common
            )

        return fn

    from lores.backends.vllm.decode_tuning import gdn_decode_config
    from lores.backends.vllm.state import _gdn_launch

    current_fp = gdn_decode_config(batch, (32, 1, 3)) if arch == "gdn" else (32, 1, 3)
    current_int = _gdn_launch(batch) if arch == "gdn" else ((8, 1, 3) if pools.axis == "value" else (pools.group or v, 4, 2))
    return {"fp32": fp_factory, "int8": int_factory}, {"fp32": current_fp, "int8": current_int}


def mamba_case(batch, shape, pools, args):
    from vllm.model_executor.layers.mamba.ops import mamba_ssm

    from lores.backends.vllm.state import _ssu_launch
    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    h, dim, n, groups = (shape[name] for name in ("heads", "dim", "state", "groups"))

    def rand(*dims):
        return torch.randn(dims, device="cuda", dtype=torch.bfloat16)

    x, dt = rand(batch, h, dim), rand(batch, h)[:, :, None].expand(-1, -1, dim)
    a = -torch.exp(torch.rand(h, device="cuda"))[:, None, None].expand(-1, dim, n)
    dt_bias = rand(h)[:, None].expand(-1, dim)
    d = rand(h)[:, None].expand(-1, dim)
    b, c = rand(batch, groups, n), rand(batch, groups, n)
    out = torch.empty_like(x)
    idx = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    seeds = torch.arange(batch, device="cuda", dtype=torch.int32) + 1234 if args.state_sr else None
    common = dict(D=d, dt_bias=dt_bias, dt_softplus=True, state_batch_indices=idx, out=out, null_block_id=0)

    def fp_factory(cfg):
        nxt = cycle(pools.states)

        def fn():
            with mamba_ssm.override_ssm_config(cfg):
                mamba_ssm.selective_state_update(nxt(), x, dt, a, b, c, **common)
            return out

        return fn

    def int_factory(cfg):
        nxt = cycle(pools.packed)

        def fn():
            packed = nxt()
            selective_state_update_int(
                packed.codes,
                packed.scales,
                x,
                dt,
                a,
                b,
                c,
                sr_seed=seeds,
                sr_salt=1,
                asym=packed.asym,
                fast=args.state_fast,
                block_m=cfg[0],
                num_warps=cfg[1],
                **common,
            )
            return out

        return fn

    default = mamba_ssm.try_get_optimal_ssm_config(dim, n, batch, h, "float32", False)
    return {"fp32": fp_factory, "int8": int_factory}, {"fp32": default, "int8": _ssu_launch(batch)}


def check_step(pools, factories, defaults):
    pools.reset("int8")
    pools.states[0].copy_(pools.unpack(pools.packed[0]))
    expected = factories["fp32"](defaults["fp32"])().clone()
    actual = factories["int8"](defaults["int8"])()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=torch.finfo(actual.dtype).eps)
    restored = pools.unpack(pools.packed[0])
    scale = pools.scale_map(pools.packed[0])
    error = (restored - pools.states[0]).abs()
    bound = 1.05 * scale + 2e-6 * pools.states[0].abs() + 1e-7
    if not torch.isfinite(restored).all() or not (error <= bound).all():
        raise RuntimeError(f"State differs by more than one SR quantization step: {error.max().item()}")
    return {
        "passed": True,
        "output_max_abs_error": (actual - expected).abs().max().item(),
        "state_max_abs_error": error.max().item(),
    }


@torch.no_grad()
def benchmark(args, arch, batch):
    torch.manual_seed(args.seed)
    shape = shape_for(args.profile, arch)
    layers = args.layers or (1 if args.check_only else max(4, 512 // batch))
    tail = (
        (shape["heads"], shape["dim"], shape["state"])
        if arch == "mamba2"
        else (shape["value_heads"], shape["value"], shape["key"])
    )
    asym = (arch != "gdn") if args.state_asym is None else args.state_asym
    axis = "value" if arch != "kda" or args.state_axis == "value" else "key"
    group = 32 if args.state_axis == "key32" else None
    pools = Pools((batch + 1, *tail), layers, asym, shape["conv_bytes"], axis, group)
    factories, defaults = (
        mamba_case(batch, shape, pools, args) if arch == "mamba2" else delta_case(arch, batch, shape, pools, args)
    )
    row = dict(
        arch=arch,
        batch=batch,
        shape=shape,
        layers=layers,
        page_bytes=pools.page_bytes,
        state_sr=args.state_sr,
        state_asym=asym,
        state_fast=args.state_fast,
        state_axis=args.state_axis,
        check=check_step(pools, factories, defaults),
    )
    if args.check_only:
        return row
    timed = make_timer(layers)
    candidates, selected = {}, {}
    grid = GRID_M2 if arch == "mamba2" else GRID3
    for arm in ("fp32", "int8"):
        candidates[arm] = []
        arm_grid = grid
        if arm == "int8" and axis == "key":  # the key axis fixes BV: the whole channel (V) or the 32-value group
            arm_grid = [cfg for cfg in grid if cfg[0] == (group or shape["value"])]
        for cfg in dict.fromkeys([defaults[arm], *arm_grid]):
            try:
                samples = []
                for _ in range(args.search_reps):
                    pools.reset(arm)
                    samples.append(timed(factories[arm](cfg)))
                result = dict(launch=cfg, us=samples, median_us=statistics.median(samples))
            except (triton.OutOfResources, triton.CompilationError) as ex:
                result = dict(launch=cfg, error=str(ex))
            candidates[arm].append(result)
        eligible = [x for x in candidates[arm] if "median_us" in x]
        if not eligible:
            raise RuntimeError(f"No valid {arch} {arm} launch at batch {batch}")
        selected[arm] = min(eligible, key=lambda x: x["median_us"])["launch"]
    row["selected_check"] = check_step(pools, factories, selected)
    samples = {"fp32": [], "int8": []}
    for rep in range(args.reps):
        for arm in ("fp32", "int8") if rep % 2 == 0 else ("int8", "fp32"):
            pools.reset(arm)
            samples[arm].append(timed(factories[arm](selected[arm])))
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    row.update(
        candidates=candidates,
        selected=selected,
        measured_us=samples,
        median_us=medians,
        speedup=medians["fp32"] / medians["int8"],
    )
    print(
        f"{arch} B={batch}: fp32={medians['fp32']:.3f} us, int8={medians['int8']:.3f} us, speedup={row['speedup']:.3f}x",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("equistate", "nemo", "qwen"), default="equistate")
    parser.add_argument("--arch", choices=("gdn", "kda", "mamba2"), action="append")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--batches", type=int, nargs="+", help="explicit sweep batch sizes")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--search-reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-sr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-asym", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--state-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--state-axis", choices=("value", "key", "key32"), default="value", help="KDA only: key = one scale per key channel, key32 = per 32-value group")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gpu-tag", default="h100")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(x is not None and x < 1 for x in (args.batch, args.layers, args.reps, args.search_reps)):
        parser.error("batch, layers and repetitions must be positive")
    arches = args.arch or {"equistate": ["gdn", "kda", "mamba2"], "nemo": ["mamba2"], "qwen": ["gdn"]}[args.profile]
    if args.profile != "equistate" and arches != (["mamba2"] if args.profile == "nemo" else ["gdn"]):
        parser.error("model profiles must use their model's architecture")
    if args.state_axis != "value" and arches != ["kda"]:
        parser.error("--state-axis key/key32 is KDA-only")
    batches = (
        [args.batch]
        if args.batch
        else args.batches
        if args.batches
        else ([1, 2, 4, 8, 16, 32, 64] + ([128] if args.profile == "equistate" else []) if args.sweep else [1])
    )
    config = dict(
        vars(args),
        architectures=arches,
        batches=batches,
        grid_delta=GRID3,
        grid_mamba=GRID_M2,
        sr_seed_policy="fixed per-slot seeds during CUDA-graph replays; layer salt 1",
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    source = vllm_source()
    gpu = torch.cuda.get_device_name()
    target = {"h100": "NVIDIA H100 80GB HBM3", "rtx": "Quadro RTX 6000"}
    if not args.check_only and gpu != target.get(args.gpu_tag):
        raise RuntimeError(f"Wrong timing GPU: {gpu}, requested {args.gpu_tag}")
    config.update(source_identity(), gpu=gpu, vllm_source=source, producer_sha256=sha256(__file__))
    run = create_run("benchmarks", "shared", f"kernel-{args.profile}", config, device=args.gpu_tag, root=args.output_root)
    print(f"OUTPUT {run}", flush=True)
    rows, tuning = [], {"metadata": config}
    for arch, batch in itertools.product(arches, batches):
        row = benchmark(args, arch, batch)
        rows.append(row)
        write_json_once(run / f"{arch}_b{batch}.json", row)
        if not args.check_only:
            for arm, launch in row["selected"].items():
                kind = ("mamba" if arch == "mamba2" else arch) + ("_fp" if arm == "fp32" else "_int8")
                key = batch * row["shape"]["heads"] if kind == "mamba_fp" else batch
                tuning.setdefault(kind, {})[str(key)] = (
                    dict(BLOCK_SIZE_M=launch[0], num_warps=launch[1]) if kind == "mamba_fp" else launch
                )
        torch.cuda.empty_cache()
    write_json_once(run / "results.json", rows)
    write_json_once(run / "tuning.json", tuning)
    write_json_once(run / "complete.json", {"cells": len(rows), "tuning_sha256": sha256(run / "tuning.json")})
    print(f"COMPLETE {run}", flush=True)


if __name__ == "__main__":
    main()
