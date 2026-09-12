"""PMI co-occurrence graph construction.

Implements the single highest-ablation-value component of the Gated
Multi-Graph Fusion architecture (Xiao et al. 2026, arXiv:2606.31186; see
docs/research/sota_papers.md section 3.4): a PMI-weighted word co-occurrence
graph computed against a healthy-speaker reference corpus, intended to encode
narrative-logic deviation rather than plain lexical content.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np

from alzspeech.features.text_features import _WORD_RE


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


class PMILookup:
    """Word-pair PMI scores estimated from a reference corpus (e.g. cn-only
    training transcripts), using a sliding co-occurrence window.
    """

    def __init__(self, window: int = 3):
        self.window = window
        self.pair_counts: Counter = Counter()
        self.word_counts: Counter = Counter()
        self.n_windows = 0

    def fit(self, reference_transcripts: Iterable[str]) -> PMILookup:
        for text in reference_transcripts:
            tokens = _tokenize(text)
            for i, w in enumerate(tokens):
                self.word_counts[w] += 1
                for j in range(i + 1, min(i + 1 + self.window, len(tokens))):
                    pair = tuple(sorted((w, tokens[j])))
                    self.pair_counts[pair] += 1
                    self.n_windows += 1
        return self

    def pmi(self, w1: str, w2: str) -> float:
        if self.n_windows == 0:
            return 0.0
        pair = tuple(sorted((w1, w2)))
        p_pair = self.pair_counts.get(pair, 0) / self.n_windows
        if p_pair == 0:
            return 0.0
        total_words = sum(self.word_counts.values()) or 1
        p_w1 = self.word_counts.get(w1, 0) / total_words
        p_w2 = self.word_counts.get(w2, 0) / total_words
        if p_w1 == 0 or p_w2 == 0:
            return 0.0
        return math.log(p_pair / (p_w1 * p_w2) + 1e-12)


def build_graph(
    transcript: str, pmi_lookup: PMILookup, threshold: float = 0.3, window: int = 3
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a single-document PMI co-occurrence graph.

    Returns (node_words, edge_index [2, E], edge_weight [E]). Nodes are the
    unique words in the transcript (in first-occurrence order); an edge is
    kept only if the reference-corpus PMI of the pair exceeds ``threshold``.
    """
    tokens = _tokenize(transcript)
    node_words: list[str] = []
    word_to_idx: dict[str, int] = {}
    for w in tokens:
        if w not in word_to_idx:
            word_to_idx[w] = len(node_words)
            node_words.append(w)

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i, w in enumerate(tokens):
        for j in range(i + 1, min(i + 1 + window, len(tokens))):
            w2 = tokens[j]
            if w == w2:
                continue
            a, b = word_to_idx[w], word_to_idx[w2]
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            score = pmi_lookup.pmi(w, w2)
            if score > threshold:
                seen_pairs.add(key)
                edges.append((a, b))
                edges.append((b, a))
                weights.append(score)
                weights.append(score)

    if not edges:
        # Guarantee at least self-loops so downstream GAT layers have valid
        # (possibly degenerate) graphs to operate on for sparse transcripts.
        edges = [(i, i) for i in range(len(node_words))]
        weights = [1.0 for _ in node_words]

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    return node_words, edge_index, edge_weight
