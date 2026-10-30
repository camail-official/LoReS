"""W/A model preparation: rotations and quantization."""

from __future__ import annotations

import logging
import types

import torch
import transformers

from lores.config import PTQConfig

log = logging.getLogger("lores")


def _spinquant_args(cfg: PTQConfig) -> types.SimpleNamespace:
    """Adapt PTQ settings to the inherited calibration routines."""
    return types.SimpleNamespace(
        w_bits=cfg.quant.w_bits,
        w_groupsize=cfg.quant.w_groupsize,
        w_clip=cfg.quant.w_clip,
        percdamp=0.01,
    )


def install_online_hadamards(model, targets) -> int:
    """Attach FP32 online Hadamards to wrappers around the selected linear modules."""
    from lores.utils import hadamard_utils
    from lores.wa_ptq import quantizers

    targets = {id(module) for module in targets}
    count = 0
    for wrapper in quantizers.find_qlayers(model, layers=[quantizers.ActQuantWrapper]).values():
        if id(wrapper.module) in targets:
            had_K, K = hadamard_utils.get_hadK(wrapper.module.in_features)
            wrapper.online_full_had = wrapper.fp32_had = True
            wrapper.had_K, wrapper.K = had_K, K
            count += 1
    return count


def _prepare_hybrid(cfg: PTQConfig, model, spec):
    """Prepare opt-in hybrid-model rotations and activation quantization.

    R1 includes norm fusion; R4 pairs with R1 unless overridden; R_o is independent.
    Fold inverse rotations before wrapping activations, then attach matching online
    rotations by module identity. Weights come from llm-compressor exports.
    """
    from lores.wa_ptq import quantizers

    if cfg.quant.w_bits < 16:
        raise ValueError(f"{spec.label}: hybrid weights are quantized by llm-compressor exports (scripts.export), not here")

    rot = cfg.rotation
    # Resolve unset (None) flags to an explicit off.
    rot.residual = bool(rot.residual)
    rot.o = bool(rot.o)
    if rot.residual:
        from lores.rotations.matrices import get_orthogonal_matrix

        torch.manual_seed(rot.seed)
        R1 = get_orthogonal_matrix(spec.hidden, device=spec.embed.weight.device).to(torch.float32)
        spec.fuse_norms(model)
        spec.rotate_residual(model, R1, fold_r1=True)
        log.info(
            "%s residual R1 (%d×%d) folded into backbone (embeddings/lm_head/all residual-facing projections); norms fused",
            spec.label,
            spec.hidden,
            spec.hidden,
        )

    # R4: the online down-proj-input Hadamard, paired with R1 (SpinQuant pairs them). Inverse folded into the weight
    # here; matching forward Hadamard installed on each down-proj ActQuantWrapper (by identity) after add_actquant.
    r4 = rot.r4 if rot.r4 is not None else bool(rot.residual)  # bundled with R1 unless --rotate_r4/--no-rotate_r4 is set
    r4_sign = r4 and rot.r4_signs
    r4_dps: list = []  # the raw down-proj Linears (matched by identity after add_actquant)
    r4_signs_map: dict = {}  # inner down-proj Linear -> its sign D (consumed by the wrapper install below)
    if r4:
        from lores.rotations.folding import fold_hadamard_input
        from lores.rotations.matrices import r4_signs

        for si, parent, attr in spec.down_projs():
            dp = getattr(parent, attr)
            if r4_sign:
                D = r4_signs(dp.in_features, si, dtype=dp.weight.dtype)
                dp.weight.data = (dp.weight.data.double() * D.double().to(dp.weight.device)).to(dp.weight.dtype)
                r4_signs_map[dp] = D.to(dp.weight.device)  # keyed by the Linear; ActQuantWrapper.module is this same dp
            fold_hadamard_input(dp)
            r4_dps.append(dp)
        log.info(
            "%s R4%s: Hadamard folded into %d down-projections",
            spec.label,
            " (randomized-sign H·D)" if r4_sign else "",
            len(r4_dps),
        )

    # R_o: Hadamard on each recurrent mixer's o-proj input (after the gated norm). Only the recurrent layers.
    r_o = bool(rot.o)
    r_o_ops: list = []  # the raw o-proj Linears (matched by identity after add_actquant)
    if r_o:
        from lores.rotations.folding import fold_hadamard_input

        for op in spec.o_projs():
            fold_hadamard_input(op)
            r_o_ops.append(op)
        log.info("%s R_o: inverse Hadamard folded into %d o-proj weights", spec.label, len(r_o_ops))

    quantizers.add_actquant(spec.actquant_root)

    if r_o:  # online Hadamard on each o-proj ActQuantWrapper input, matched by identity (per-linear get_hadK)
        n_on = install_online_hadamards(model, r_o_ops)
        log.info("%s R_o: online Hadamard on %d o-proj ActQuantWrapper inputs", spec.label, n_on)
    if r4:  # online Hadamard on each down-proj ActQuantWrapper input, matched by identity (per-linear get_hadK)
        n_on = install_online_hadamards(model, r4_dps)
        if r4_sign:  # wrap each down-proj ActQuantWrapper so D is applied to the activation before its online Hadamard
            from lores.rotations.wrappers import SignedInputWrapper

            for _si, parent, attr in spec.down_projs():
                dp = getattr(parent, attr)
                if isinstance(dp, quantizers.ActQuantWrapper) and dp.module in r4_signs_map:
                    setattr(parent, attr, SignedInputWrapper(dp, r4_signs_map[dp.module]))
        log.info(
            "%s R4%s: online Hadamard on %d down-proj ActQuantWrapper inputs",
            spec.label,
            " + per-down-proj sign D (SignedInputWrapper)" if r4_sign else "",
            n_on,
        )

    if cfg.quant.a_bits < 16:
        n = 0
        for name, ql in quantizers.find_qlayers(model, layers=[quantizers.ActQuantWrapper]).items():
            if "lm_head" in name:
                continue
            ql.quantizer.configure(bits=cfg.quant.a_bits, groupsize=cfg.quant.a_groupsize, sym=cfg.quant.a_sym)
            n += 1
        log.info(
            "%s linear-input act quant: a_bits=%d on %d ActQuantWrappers (groupsize=%d)",
            spec.label,
            cfg.quant.a_bits,
            n,
            cfg.quant.a_groupsize,
        )
    return model


def prepare_model(cfg: PTQConfig, model, *, calib_loader=None):
    """In-place model preparation; architecture adapters define the supported targets."""
    import importlib

    from lores.models.registry import detect_arch

    transformers.set_seed(cfg.seed)
    model.eval()
    arch = detect_arch(model)
    adapter = importlib.import_module(f"lores.models.{arch}.ptq")
    if arch == "mixer_zoo":
        return adapter.prepare_mixer_zoo(cfg, model, calib_loader=calib_loader)
    return _prepare_hybrid(cfg, model, adapter.build_ptq_spec(model))
