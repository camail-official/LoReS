# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# From SpinQuant (https://github.com/facebookresearch/SpinQuant): utils/data_utils.py.

"""WikiText-2 tokenization and calibration sampling."""

import random

import datasets


def wikitext_tokens(tokenizer, split):
    """Tokenize one complete WikiText-2 split."""
    data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt")


def get_wikitext2(tokenizer, nsamples=128, seed=0, seqlen=2048, eval_mode=False):
    """Return test encodings or seeded training windows for calibration."""
    enc = wikitext_tokens(tokenizer, "test" if eval_mode else "train")
    if eval_mode:
        return enc
    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, start : start + seqlen]
        target = inp.clone()
        target[:, :-1] = -100
        samples.append((inp, target))
    return samples


def train_tokens(tok, n_tokens):
    """Take contiguous training tokens without random sampling."""
    enc = wikitext_tokens(tok, "train").input_ids
    assert enc.shape[1] >= n_tokens, "train corpus too short"
    return enc[:, :n_tokens]
