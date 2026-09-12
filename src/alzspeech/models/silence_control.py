"""Silence-only control classifier — the mandatory Clever Hans sanity check.

Liu, Feng, Yuan & Ling (Interspeech 2024, arXiv:2406.07410) found that silence
segments alone reach near-100% accuracy on the Pitt corpus, implicating
recording/processing artifacts rather than genuine disease signal (see
docs/research/sota_papers.md, pitfall #3). Any "real" model's reported
accuracy in this project must be compared against this control's accuracy;
`alzspeech.training.train` always trains and reports both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from alzspeech.features.text_features import Word

SILENCE_FEATURE_NAMES = (
    "silence_ratio",
    "num_pauses",
    "mean_pause_duration",
    "speech_duration",
)


def compute_silence_features(words: Sequence[Word], total_duration: float, pause_threshold_s: float = 0.5) -> np.ndarray:
    """Derive silence/pause summary stats from ASR word timestamps only —
    deliberately no lexical content, so this control cannot pick up any
    linguistic signal.
    """
    if not words or total_duration <= 0:
        return np.zeros(len(SILENCE_FEATURE_NAMES), dtype=np.float32)

    speech_duration = sum(w.end - w.start for w in words)
    gaps = [b.start - a.end for a, b in zip(words, words[1:]) if b.start > a.end]
    pauses = [g for g in gaps if g >= pause_threshold_s]

    silence_ratio = max(0.0, 1.0 - speech_duration / total_duration)
    num_pauses = float(len(pauses))
    mean_pause_duration = float(np.mean(pauses)) if pauses else 0.0

    return np.array([silence_ratio, num_pauses, mean_pause_duration, speech_duration], dtype=np.float32)


class SilenceControlClassifier(nn.Module):
    """A small MLP over silence-only features — no access to lexical content
    or acoustic embeddings, by design.
    """

    def __init__(self, n_features: int = len(SILENCE_FEATURE_NAMES), hidden: int = 8, num_labels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
