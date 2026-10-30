"""GPQA preprocessing for the NeMo-Skills-prompt task.

Copied from lm-eval's gpqa/cot_zeroshot/utils.py: normalization and choice shuffling
retain the stock task's answer mapping.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datasets


def preprocess(text):
    """lm-eval's choice-text normalisation (verbatim)."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")

    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Shuffle the four choices and record the answer letter (verbatim from lm-eval)."""

    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
