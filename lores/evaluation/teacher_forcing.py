"""Evaluation sequences and teacher-forced token losses."""

from __future__ import annotations

import torch


def to_rows(enc, n_seqs, seqlen):
    """Slice a token encoding into contiguous, non-overlapping evaluation rows."""
    ids = enc.input_ids
    n = min(n_seqs, ids.shape[1] // seqlen)
    if n == 0:
        raise RuntimeError(f"corpus too short: {ids.shape[1]} < {seqlen}")
    return ids[:, : n * seqlen].reshape(n, seqlen)


def make_sequences(tok, seqlen, n_seqs):
    """Build contiguous evaluation rows from WikiText-2's test split."""
    from lores.evaluation.data import get_wikitext2

    return to_rows(get_wikitext2(tokenizer=tok, seqlen=seqlen, eval_mode=True), n_seqs, seqlen)


@torch.no_grad()
def decode_per_position(model, seqs, prefill, seed, *, device=None, allow_state_clipping=False):
    """Evaluate all sequences from scratch, returning complete FP32 losses and argmax IDs."""
    from lores.fake_quant.quantizers import saturation_events

    if seqs.ndim != 2 or not len(seqs) or not 1 <= prefill < seqs.shape[1]:
        raise ValueError("Expected nonempty token sequences and 1 <= prefill < sequence length")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = seqs.device
    torch.manual_seed(seed)
    n_seqs, seqlen = seqs.shape
    n_dec = seqlen - prefill
    ce = torch.full((n_seqs, n_dec), float("nan"), dtype=torch.float32)
    am = torch.zeros(n_seqs, n_dec, dtype=torch.long)
    for s in range(n_seqs):
        saturation_start = saturation_events()
        torch.manual_seed(seed + s)  # reproducible, independent per-sequence random streams
        ids = seqs[s : s + 1].to(device)
        out = model(input_ids=ids[:, :prefill], use_cache=True, return_dict=True)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        for p in range(n_dec):
            t = prefill + p
            ce[s, p] = -torch.log_softmax(logits.float(), -1)[0, ids[0, t]].item()
            am[s, p] = int(logits.argmax(-1).item())
            out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True, return_dict=True)
            cache, logits = out.past_key_values, out.logits[:, -1, :]
        if not torch.isfinite(ce[s]).all():
            raise ValueError(f"Non-finite token loss in sequence {s}")
        clipped = saturation_events() - saturation_start
        if clipped:
            if not allow_state_clipping:
                raise ValueError(f"State saturation in sequence {s}; refusing to return clipped losses")
            print(f"  seq {s + 1}: {clipped} state elements handled by the existing clipping guard", flush=True)
        print(f"  seq {s + 1}/{n_seqs} done", flush=True)
    if not torch.isfinite(ce).all():
        raise ValueError("Incomplete teacher-forced loss coverage")
    return ce, am
