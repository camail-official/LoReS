"""MixerZoo W/A preparation and GPTQ replay for its uniform block tree."""

from __future__ import annotations

import logging

import torch

from lores.config import PTQConfig
from lores.wa_ptq.gptq import _gptq_impl
from lores.wa_ptq.prepare import _spinquant_args, install_online_hadamards

log = logging.getLogger("lores")


def _mixers(model):
    from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers

    return list(iter_mixer_zoo_layers(model))


def _mlp_intermediate_size(model) -> int:
    """Read the realized MLP width; config.intermediate_size may be None."""
    return model.model.layers[0].mlp.gate_proj.weight.shape[0]


def _down_proj_groupsize(model, a_groupsize: int) -> int:
    """Use the requested group width when divisible; otherwise preserve group count."""
    if a_groupsize <= 0:
        return -1
    inter = _mlp_intermediate_size(model)
    if inter % a_groupsize == 0:
        return a_groupsize
    hidden = model.config.hidden_size
    assert hidden % a_groupsize == 0, f"a_groupsize {a_groupsize} doesn't divide hidden_size {hidden}"
    group_num = hidden // a_groupsize
    assert inter % group_num == 0, (
        f"a_groupsize {a_groupsize} gives {group_num} groups, which doesn't divide intermediate_size {inter}"
    )
    return inter // group_num


def _add_actquant_and_setup_online_had(model, *, r4: bool, o: bool) -> None:
    """Wrap linears and attach folded rotations' matching online Hadamards."""
    from lores.wa_ptq import quantizers

    targets = []
    if r4:
        targets.extend(block.mlp.down_proj for block in model.model.layers)
    if o:
        targets.extend(mix.o_proj for mix in _mixers(model))
    quantizers.add_actquant(model)
    install_online_hadamards(model, targets)


def _configure_linear_input_quant(cfg: PTQConfig, model) -> None:
    """Configure activation quantization after weight calibration."""
    if cfg.quant.a_bits >= 16:
        return
    from lores.wa_ptq import quantizers

    vhead_proj, vhead_dim = "o_proj", _mixers(model)[0].head_v_dim
    down_gs = _down_proj_groupsize(model, cfg.quant.a_groupsize)
    n = 0
    for name, ql in quantizers.find_qlayers(model, layers=[quantizers.ActQuantWrapper]).items():
        if "lm_head" in name:
            continue
        groupsize = cfg.quant.a_groupsize
        if vhead_proj in name:
            groupsize = vhead_dim
        if "down_proj" in name:
            groupsize = down_gs
        ql.quantizer.configure(bits=cfg.quant.a_bits, groupsize=groupsize, sym=cfg.quant.a_sym)
        n += 1
    log.info(
        "linear-input activation quant: a_bits=%d sym=%s on %d ActQuantWrappers (groupsize=%d; %s→%dg; down_proj→%dg)",
        cfg.quant.a_bits,
        cfg.quant.a_sym,
        n,
        cfg.quant.a_groupsize,
        vhead_proj,
        vhead_dim,
        down_gs,
    )


def _unfuse_swiglu_if_needed(cfg: PTQConfig, model, *, r4: bool) -> None:
    """Disable fused SwiGLU, which bypasses down_proj's quantization/rotation wrapper."""
    if not (cfg.quant.a_bits < 16 or r4):
        return
    n = 0
    for block in model.model.layers:
        mlp = getattr(block, "mlp", None)
        if mlp is not None and getattr(mlp, "fuse_swiglu", False):
            mlp.fuse_swiglu = False
            n += 1
    if n:
        log.info("un-fused SwiGLU in %d MLP block(s) so down_proj's ActQuantWrapper (input quant / R4 Hadamard) runs", n)


def _resolve_rotation_defaults(rot) -> None:
    """Fill unset rotation defaults, preserving explicit settings."""
    for field, default in (("residual", True), ("o", True)):
        if getattr(rot, field) is None:
            setattr(rot, field, default)


def prepare_mixer_zoo(cfg: PTQConfig, model, *, calib_loader=None):
    """Prepare MixerZoo rotations and W/A quantization.

    Fold rotations before wrapping; calibrate weights before activation quantization.
    """
    rot = cfg.rotation
    _resolve_rotation_defaults(rot)

    # Unset R4 follows residual rotation.
    r4 = rot.r4 if rot.r4 is not None else bool(rot.residual)
    r_o = rot.o

    if rot.residual:
        from lores.models.mixer_zoo.residual import fuse_norms, rotate_residual
        from lores.rotations.matrices import get_orthogonal_matrix

        fuse_norms(model)
        dev = model.model.embeddings.weight.device

        torch.manual_seed(rot.seed)
        R1 = get_orthogonal_matrix(model.config.hidden_size, device=dev)
        rotate_residual(model, R1.to(torch.float32), rotate_r4=r4, rotate_o=r_o)
        log.info(
            "residual rotation R1 (%d×%d) folded into embeddings/lm_head/all projections; norms fused; R4=%s R_o=%s",
            model.config.hidden_size,
            model.config.hidden_size,
            r4,
            r_o,
        )
    elif r4 or r_o:
        from lores.models.mixer_zoo.residual import rotate_residual

        rotate_residual(model, None, rotate_r4=r4, rotate_o=r_o, fold_r1=False)
        log.info("R4=%s / R_o=%s: inverse Hadamard(s) folded into down_proj / o_proj input dim (no R1)", r4, r_o)

    _add_actquant_and_setup_online_had(model, r4=r4, o=r_o)

    _unfuse_swiglu_if_needed(cfg, model, r4=r4)

    if cfg.quant.w_bits < 16:
        if calib_loader is None:
            raise ValueError("GPTQ weight quant needs calibration data — pass calib_loader (run_ptq builds it)")
        gptq_mixer_zoo_fwrd(model, calib_loader, cfg.device, _spinquant_args(cfg))
        model.to(cfg.device)
        log.info(
            "GPTQ (%d calib samples) weight quant: w_bits=%d groupsize=%d clip=%s (lm_head unquantized)",
            cfg.quant.calib_nsamples,
            cfg.quant.w_bits,
            cfg.quant.w_groupsize,
            cfg.quant.w_clip,
        )

    _configure_linear_input_quant(cfg, model)

    return model


MIXER_ZOO_GPTQ_SEQUENTIAL = [
    [
        "attn.q_proj.module",
        "attn.k_proj.module",
        "attn.v_proj.module",
        "attn.g_proj.module",
        "attn.b_proj.module",
        "attn.gk_proj.module",
        "attn.gk_proj.0.module",
    ],
    ["attn.gk_proj.1.module"],
    ["attn.o_proj.module"],
    ["mlp.gate_proj.module", "mlp.up_proj.module"],
    ["mlp.down_proj.module"],
]


@torch.no_grad()
def gptq_mixer_zoo_fwrd(model, dataloader, dev, args):
    """Quantize all MixerZoo block linears, including decay and beta projections."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    seqlen = getattr(model, "seqlen", None) or dataloader[0][0].shape[1]
    dtype = next(iter(model.parameters())).dtype

    # Fused SwiGLU bypasses down_proj hooks, leaving its calibration Hessian empty.
    refused_mlps = []
    for block in layers:
        mlp = getattr(block, "mlp", None)
        if mlp is not None and getattr(mlp, "fuse_swiglu", False):
            mlp.fuse_swiglu = False
            refused_mlps.append(mlp)
    try:
        _gptq_impl(model, dataloader, dev, args, layers, seqlen, dtype, MIXER_ZOO_GPTQ_SEQUENTIAL, "MixerZoo")
    finally:
        for mlp in refused_mlps:
            mlp.fuse_swiglu = True
        model.config.use_cache = use_cache
