"""Speaker-level cross-validation.

The literature review (docs/research/sota_papers.md, pitfall #2) documents that
segment- or utterance-level splits leak speaker identity between train and
validation sets and can inflate reported accuracy by up to ~28 percentage
points on this exact dataset. Every split produced by this module is grouped
by speaker: a given speaker's rows always land entirely in one fold. There is
deliberately no segment-level split path anywhere in this codebase.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold


def speaker_kfold(
    speaker_ids: Sequence,
    labels: Sequence | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) index arrays, grouped by speaker.

    If ``labels`` is given, uses StratifiedGroupKFold so class balance is
    preserved across folds as well as speaker grouping; otherwise falls back
    to a plain GroupKFold.
    """
    speaker_ids = np.asarray(speaker_ids)
    n = len(speaker_ids)
    dummy_X = np.zeros((n, 1))

    if labels is not None:
        labels = np.asarray(labels)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(dummy_X, labels, groups=speaker_ids)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(dummy_X, groups=speaker_ids)

    for train_idx, val_idx in splits:
        _assert_no_leakage(speaker_ids, train_idx, val_idx)
        yield train_idx, val_idx


def leave_one_speaker_out(speaker_ids: Sequence) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) index arrays for leave-one-speaker-out CV."""
    speaker_ids = np.asarray(speaker_ids)
    dummy_X = np.zeros((len(speaker_ids), 1))
    splitter = LeaveOneGroupOut()
    for train_idx, val_idx in splitter.split(dummy_X, groups=speaker_ids):
        _assert_no_leakage(speaker_ids, train_idx, val_idx)
        yield train_idx, val_idx


def _assert_no_leakage(speaker_ids: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    train_speakers = set(speaker_ids[train_idx].tolist())
    val_speakers = set(speaker_ids[val_idx].tolist())
    overlap = train_speakers & val_speakers
    if overlap:
        raise AssertionError(
            f"Speaker leakage detected between train and validation folds: {sorted(overlap)!r}"
        )
