"""Transcript-level linguistic features: pause-token insertion and disfluency stats.

Implements the "insert pause markers into the transcript before fine-tuning a
transformer" template from Yuan et al. 2021 (arXiv:2106.08689), which reported
89.6% accuracy on ADReSS — see docs/research/sota_papers.md section 3.1.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

PAUSE_TOKEN = "[PAUSE]"
FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "hmm", "ah"}


@dataclass
class Word:
    """A single ASR-aligned word with timing, in seconds."""

    text: str
    start: float
    end: float


def insert_pause_tokens(words: Sequence[Word], pause_threshold_s: float = 0.5) -> str:
    """Reconstruct a transcript from aligned words, inserting PAUSE_TOKEN
    wherever the gap to the previous word exceeds ``pause_threshold_s``.
    """
    if not words:
        return ""
    pieces = [words[0].text]
    for prev, cur in zip(words, words[1:]):
        gap = cur.start - prev.end
        if gap >= pause_threshold_s:
            pieces.append(PAUSE_TOKEN)
        pieces.append(cur.text)
    return " ".join(pieces)


_WORD_RE = re.compile(r"[a-zA-Z']+")


def disfluency_features(transcript: str) -> dict:
    """Cheap, model-free lexical/disfluency summary stats for a transcript.

    Used both as auxiliary classifier features (concatenated before the final
    layer, per Yuan et al.) and as a sanity-check that synthetic/real
    transcripts carry the expected label-correlated direction.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(transcript)]
    n_words = len(tokens)
    if n_words == 0:
        return {
            "n_words": 0,
            "n_pause_tokens": transcript.count(PAUSE_TOKEN),
            "filler_count": 0,
            "filler_rate": 0.0,
            "type_token_ratio": 0.0,
            "avg_word_len": 0.0,
            "repeated_word_rate": 0.0,
        }

    filler_count = sum(1 for t in tokens if t in FILLERS)
    n_unique = len(set(tokens))
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)

    return {
        "n_words": n_words,
        "n_pause_tokens": transcript.count(PAUSE_TOKEN),
        "filler_count": filler_count,
        "filler_rate": filler_count / n_words,
        "type_token_ratio": n_unique / n_words,
        "avg_word_len": sum(len(t) for t in tokens) / n_words,
        "repeated_word_rate": repeats / max(n_words - 1, 1),
    }
