"""Install recurrent-state precision, quantizers, and rotations on HF models."""

import re

ROTATIONS = {
    "nemotronh": ("none", "rk", "rv"),
    "qwen3_5": ("none", "rk", "rv"),
    "kimi": ("none", "rv"),
}


def state_quant_mode(mode, arch, axis):
    """Translate a public format/rounding mode and semantic axis to HF storage grouping."""
    if arch not in ROTATIONS or axis not in ("value", "key"):
        raise ValueError(f"Unsupported state grouping: {arch=}, {axis=}")
    if mode == "off" or re.fullmatch(r"int[2-8]_tensor", mode):
        return mode
    match = re.fullmatch(r"(int[2-8]|fpE\d+M\d+)(_asym)?(_sr)?", mode)
    if not match or (match[1].startswith("fp") and match[2]):
        raise ValueError(f"Invalid mode {mode!r}; use e.g. int8_asym_sr with --state-axis value|key, not block/block_dk")
    code, asym, sr = match.groups(default="")
    # Nemotron stores [V,K]; Qwen and Kimi store [K,V].
    gran = "block" if (arch == "nemotronh") == (axis == "value") else "block_dk"
    return f"{code}_{gran}{asym}{sr}"


def pin_fp32_state(model, arch, *, strict=False):
    """Pin recurrent stores to FP32; strict mode rejects already-rounded caches."""
    if arch == "nemotronh":
        from lores.backends.hf.nemotronh.integration import install_nemotronh_fp32_state_cache as install
    elif arch == "qwen3_5":
        from lores.backends.hf.qwen3_5.integration import install_qwen3_5_fp32_state_cache as install
    else:
        from lores.backends.hf.kimi_linear.integration import install_kimi_fp32_state_cache as install
    count = install(model, strict=strict)
    if strict and not count:
        raise ValueError(f"No {arch} recurrent layers were pinned to FP32")
    return count


def state_installer(model, arch, *, block_size=128):
    """Return the mode setter for the architecture's recurrent mixers."""
    if arch == "nemotronh":
        from lores.backends.hf.nemotronh.integration import install_nemotronh_state_quant as install
    elif arch == "qwen3_5":
        from lores.backends.hf.qwen3_5.integration import install_qwen3_5_state_quant as install
    else:
        from lores.backends.hf.kimi_linear.integration import install_kimi_state_quant as install
    return lambda mode: install(model, mode=mode, block_size=block_size)


def install_rotations(model, arch, rotation):
    """Install carried rotations; Nemotron kernels must already be loaded."""
    if rotation not in ROTATIONS[arch]:
        raise ValueError(f"Unsupported {arch} rotation: {rotation}")
    if rotation == "none":
        return
    if arch == "nemotronh":
        from lores.backends.hf.nemotronh.rotations import install_nemotronh_rotations

        install_nemotronh_rotations(model, rk=rotation == "rk", rv=rotation == "rv", seed=0)
    elif arch == "qwen3_5":
        from lores.backends.hf.qwen3_5.rotations import install_qwen3_5_rk, install_qwen3_5_rv

        {"rk": install_qwen3_5_rk, "rv": install_qwen3_5_rv}[rotation](model, seed=0)
    else:
        from lores.backends.hf.kimi_linear.rotations import install_kimi_rv

        install_kimi_rv(model, seed=0)


def state_quant_mixers(model):
    """Mixers carrying a live decode-time state-quant hook (module order = layer order)."""
    return [m for m in model.modules() if getattr(m, "_sq_handle", None) is not None]
