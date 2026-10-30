"""Adapt vLLM's accounted code/scale caches to LoReS's INT8 state kernels.

Mamba-2, GDN and KDA keep uint8 codes and FP32 scales in the allocator-owned cache.
Prefill uses per-mixer FP32 scratch, packs at handoff, and unpacks for continuation;
decode updates the integer pools in place.
Value-axis packing runs the fused kernel, key-axis packing the torch packer; both serve all three mixers.
Enable through ``additional_config["linq"]["state_bits"] = 8``.
"""

import torch

from lores.backends.vllm import decode_tuning
from lores.backends.vllm.config import current as linq_config
from lores.backends.vllm.config import set_current
from lores.backends.vllm.rotations import layer_index

_PACK_SEED = __import__("itertools").count(1)  # prefill handoff runs eagerly: a host counter is graph-safe here


def linq_set_config(vllm_config):
    """Pin the serving config; the INT8 state needs plain slots (no prefix caching, no speculative decoding)."""
    cfg = set_current(vllm_config)
    if cfg.state_bits:
        assert vllm_config.cache_config.mamba_cache_mode == "none", "LINQ int state: mamba_cache_mode must be 'none'"
        assert vllm_config.speculative_config is None, "LINQ int state: spec decode unsupported"
    return cfg


def linq_bits() -> int:
    """0 = disabled, else 8 (validated by LinqConfig)."""
    return linq_config().state_bits


def linq_cache_dtypes(dtypes):
    """Per-slot cache dtypes: (conv, codes uint8, scales fp32) under the INT8 state, else the fp ``dtypes``."""
    return (dtypes[0], torch.uint8, torch.float32) if linq_bits() else tuple(dtypes)


def linq_cache_shapes(conv_shape, heads, value_dim, key_dim):
    """Per-slot shapes under the INT8 state: conv, codes [heads, V, K], scales [heads, V | K(, 2)]; None when off."""
    if not linq_bits():
        return None
    cfg = linq_config()
    rows = key_dim if cfg.state_axis == "key" else value_dim
    return conv_shape, (heads, value_dim, key_dim), ((heads, rows, 2) if cfg.state_asym else (heads, rows))


def _layer_salt(mixer) -> int:
    """Layer index + 1 from the mixer's prefix: decorrelates the SR noise across layers."""
    return layer_index(getattr(mixer, "prefix", "") or "") + 1


def _pools(mixer):
    """Codes/scales pools from the layer's accounted cache state (see get_state_shape).

    Returns None during vLLM's memory-profiling phase, when the layer still holds the
    dummy placeholder cache — callers must no-op then. Side allocations are wrong here:
    vLLM sizes the slot pool to fill the GPU, so anything unaccounted either overflows
    (index-OOB) or OOMs.
    """
    kv = mixer.kv_cache
    if len(kv) < 3 or kv[1].numel() == 0:
        return None
    return kv[1], kv[2]  # conv, codes, scales -- the fp state is prefill scratch


# Nemotron launch configurations, cold-L2 sweep on H100.
_TUNED_VLLM = {1: (8, 2), 2: (16, 4), 4: (8, 1), 8: (8, 1), 16: (32, 2), 32: (32, 2), 64: (32, 2), 128: (32, 2)}


def _ssu_launch(batch):
    """(block_m, num_warps) at the batch floor of the Mamba-2 INT8 table; (None, None) = the kernel's own table."""
    return decode_tuning.floor_entry(decode_tuning.override_table("mamba_int8") or _TUNED_VLLM, batch) or (None, None)


_SCRATCH: dict = {}


def linq_scratch(mixer, n, device):
    """Prefill-only fp32 state buffer ``[n, *codes shape]``, grown on demand -- never part of the page.

    One buffer per state shape, shared by all layers: each layer consumes it before the next layer runs
    (same stream), so per-layer copies would only multiply memory vLLM does not account for.
    """
    tail = tuple(mixer.get_state_shape()[1])
    key = (tail, str(device))
    buf = _SCRATCH.get(key)
    if buf is None or buf.shape[0] < n:
        _SCRATCH[key] = buf = None  # drop the old buffer before allocating the larger one
        buf = _SCRATCH[key] = torch.empty((n, *tail), dtype=torch.float32, device=device)
    return buf[:n]


@torch.no_grad()
def linq_pack_slots(mixer, state_indices, states):
    """Pack ``states`` (fp [n, H, D, N]; GDN/KDA [n, HV, V, K]) into the slots ``state_indices`` (prefill handoff)."""
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots

    pools = _pools(mixer)
    if pools is None:  # profiling-phase dummy cache
        return
    codes, scales = pools
    cfg = linq_config()
    if cfg.state_axis == "key":  # one scale per key channel over all value rows (torch path, once per prompt)
        from lores.kernels.state_int.packing.pack_state_key_axis_naive import pack_key_axis_to_slots

        pack_key_axis_to_slots(
            states, codes, scales, state_indices, asym=cfg.state_asym, sr_seed=next(_PACK_SEED) if cfg.state_sr else None
        )
        return
    pack_state_to_slots(
        states.contiguous().to(torch.float32),
        codes,
        scales,
        state_indices,
        asym=cfg.state_asym,
        sr_seed=next(_PACK_SEED) if cfg.state_sr else None,
        fast=cfg.state_fast,
    )


@torch.no_grad()
def linq_decode(mixer, x, dt, A, B, C, D, dt_bias, state_indices, out, seq_lens=None):
    """Int-state decode step; mirrors the ``selective_state_update`` call it replaces."""
    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    if state_indices.dim() == 2:  # block-table form [batch, num_blocks]
        assert state_indices.shape[1] == 1, f"LINQ int state: expected one state block per seq, got {tuple(state_indices.shape)}"
        state_indices = state_indices.squeeze(1)
    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    key_axis = linq_config().state_axis == "key"
    block_m, num_warps = (None, None) if key_axis else _ssu_launch(x.shape[0])  # key axis: one program per head
    selective_state_update_int(
        codes,
        scales,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
        out=out,
        null_block_id=0,  # vLLM v1 pads decode batches with the reserved null block 0
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced: a per-call view costs ~4 us of host time per layer
        asym=linq_config().state_asym,
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        block_m=block_m,
        num_warps=num_warps,
        fast=linq_config().state_fast,
        axis=linq_config().state_axis,
    )


@torch.no_grad()
def linq_unpack_slots(mixer, state_indices, out):
    """Dequantize slots into `out` (chunked-prefill continuation reads the int pool)."""
    from lores.kernels.state_int.packing.pack_state_kernel import unpack_state_from_slots

    codes, scales = _pools(mixer)
    if linq_config().state_axis == "key":
        from lores.kernels.state_int.packing.pack_state_key_axis_naive import unpack_key_axis_from_slots

        return unpack_key_axis_from_slots(out, codes, scales, state_indices, asym=linq_config().state_asym)
    return unpack_state_from_slots(out, codes, scales, state_indices, asym=linq_config().state_asym)


# Gated DeltaNet (qwen3_next / qwen3_5): int state in vLLM's own value-major [slots, HV, V, K] layout.


# (BV, warps, stages) per batch for the copy of vLLM's packed decode kernel, cold-L2 sweep on H100 SXM.
_TUNED_GDN = {
    1: (128, 2, 2),
    2: (8, 1, 3),
    4: (16, 1, 1),
    8: (16, 1, 3),
    16: (16, 1, 2),
    32: (16, 1, 1),
    64: (16, 1, 2),
    128: (16, 1, 1),
}


def _gdn_launch(batch):
    """(BV, warps, stages) at the batch floor of the GDN INT8 table, or None (the kernel copy's stock launch)."""
    return decode_tuning.floor_entry(decode_tuning.override_table("gdn_int8") or _TUNED_GDN, batch)


@torch.no_grad()
def linq_gdn_decode_vllm(mixer, q, k, v, a, b, A_log, dt_bias, scale, cu_seqlens, state_indices, seq_lens=None):
    """Int-state GDN decode with the copy of vLLM's own kernel (fused_sigmoid_gating_int).

    Same arguments as vLLM's ``fused_sigmoid_gating_delta_rule_update`` call at the decode
    sites (q/k/v are the rearranged ``[1, T, H, K]`` tensors); returns ``o`` shaped like it.
    """
    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_sigmoid_gating_delta_rule_update_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    return fused_sigmoid_gating_delta_rule_update_int(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        codes,
        scales,
        state_indices,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced persistent buffer, see linq_decode
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym,
        fast=linq_config().state_fast,
        axis=linq_config().state_axis,
    )


def linq_gdn_decode_packed_vllm(mixer, mixed_qkv, a, b, A_log, dt_bias, scale, state_indices, out, seq_lens=None):
    """Int-state GDN decode-only step with the copy of vLLM's packed kernel (reads [q|k|v] directly)."""
    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_recurrent_gated_delta_rule_packed_decode_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    return fused_recurrent_gated_delta_rule_packed_decode_int(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        codes,
        scales,
        out,
        state_indices,
        use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None,
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym,
        fast=linq_config().state_fast,
        launch=None if linq_config().state_axis == "key" else _gdn_launch(mixed_qkv.shape[0]),
        axis=linq_config().state_axis,
    )


# Kimi-Linear (KDA): int8 pool in vLLM's own [slots, H, V, K] layout, decode on the copy of vLLM's fused_recurrent_kda.
def linq_kda_decode_vllm(mixer, q, k, v, g, beta, cu_seqlens, state_indices, seq_lens=None):
    """Int-state KDA decode step; same arguments as vLLM's ``fused_recurrent_kda`` call site."""
    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    return fused_recurrent_kda_int(
        q,
        k,
        v,
        g,
        beta,
        None,
        codes,
        scales,
        state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced persistent buffer, see linq_decode
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym,
        fast=linq_config().state_fast,
        axis=linq_config().state_axis,
    )
