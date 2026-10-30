# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# From SpinQuant (https://github.com/facebookresearch/SpinQuant): eval_utils/gptq_utils.py.

"""Install exported W/A fake-quant weights; reject incompatible saved weights."""

from __future__ import annotations

from pathlib import Path

import torch

from lores.wa_ptq.recipes import BEST_STACK, W_GROUPSIZE, make_wa_config

LLMC_SCHEME = {4: "W4A16", 8: "W8A16"}


def _accept_stamp(st, *, arch: str, w_bits: int, rot_flags: list) -> bool:
    """Whether a saved llm-compressor artifact matches this request."""
    return (
        isinstance(st, dict)
        and st.get("quantizer") == "llmcompressor"
        and st.get("scheme") == LLMC_SCHEME.get(w_bits)
        and st.get("groupsize") == W_GROUPSIZE
        and st.get("folded") == rot_flags
        and st.get("arch") == arch
    )


def _weights_only(sd: dict) -> dict:
    """Exclude runtime activation-quantizer buffers so weights work with any a_bits."""
    return {k: v for k, v in sd.items() if "quantizer." not in k}


def install_wa(model, w_bits: int, a_bits: int, device: str = "cuda", w_file: str | None = None, arch: str = "qwen3_5") -> None:
    """Install exported backbone weights, then configure activation quantization."""
    from lores.wa_ptq.prepare import prepare_model

    rot_flags = BEST_STACK[arch]
    saved = None
    if w_bits < 16:
        if not (w_file and Path(w_file).exists()):
            raise FileNotFoundError(f"w_bits={w_bits} needs an exported weight file (scripts.export.wa_fake_quant): {w_file!r}")
        obj = torch.load(w_file, map_location="cpu")
        found = obj.get("stamp") if isinstance(obj, dict) else None
        if not _accept_stamp(found, arch=arch, w_bits=w_bits, rot_flags=rot_flags):
            raise RuntimeError(
                f"{w_file} does not match this request: stamp {found!r}, want an llmcompressor export of "
                f"{LLMC_SCHEME.get(w_bits)} at groupsize {W_GROUPSIZE} with folded={rot_flags} for arch {arch!r}."
            )
        saved = obj["state_dict"]
        print(f"[wa_quant] {w_file}: accepted llmcompressor artifact", flush=True)
    prepare_model(make_wa_config(arch, a_bits=a_bits, device=device), model)
    if saved is not None:
        missing, unexpected = model.load_state_dict(_weights_only(saved), strict=False)
        assert not unexpected and all("quantizer." in k for k in missing), (missing[:5], unexpected[:5])
        print(f"[wa_quant] loaded GPTQ w{w_bits} state from {w_file}", flush=True)
