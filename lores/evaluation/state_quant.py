"""Mask-based state-quantization setup shared by evaluation runners."""

import json
import re

import torch

from lores.backends.hf.state import state_quant_mixers


def headbits_gran(mode):
    """Preserve the base mode's grouping and rounding suffix."""
    m = re.fullmatch(r"int\d+_(block(?:_dk)?(?:_sr|_asym|_asym_sr)?)", mode)
    assert m, f"--headbits needs an int<n>_block[_dk][_sr|_asym|_asym_sr] mode (HeadBitsQuant reuses its gran), got {mode!r}"
    return m.group(1)


def bind_mask_quantizers(model, args, mode):
    """Bind per-layer head-bit allocations, with an axis mask when given."""
    if mode == "off" or not (args.headbits or args.mixhead):
        return
    from lores.evaluation.output_store import source_path
    from lores.fake_quant.state import AxisBitsQuant, HeadBitsQuant

    if not args.headbits:
        raise ValueError("--mixhead requires --headbits")
    mixers = state_quant_mixers(model)
    assert mixers, "--mixhead/--headbits: no state-quant mixers installed"
    layout = "vk" if args.model == "nemotronh" else "kv"

    def load(path):
        return json.loads(source_path(path).read_text())

    hb = load(args.headbits)
    if args.mixhead:
        masks = load(args.mixhead)
        assert len(mixers) == len(hb) == len(masks), f"{len(mixers)} mixers vs {len(hb)} bits / {len(masks)} mask layers"
        for i, m in enumerate(mixers):
            m._state_quant = AxisBitsQuant(torch.tensor(masks[str(i)]), torch.tensor(hb[str(i)]), layout=layout, mode=mode)
    else:
        gran = headbits_gran(mode)
        assert len(mixers) == len(hb), f"{len(mixers)} mixers vs {len(hb)} bits layers"
        for i, m in enumerate(mixers):
            m._state_quant = HeadBitsQuant(gran, torch.tensor(hb[str(i)]))
