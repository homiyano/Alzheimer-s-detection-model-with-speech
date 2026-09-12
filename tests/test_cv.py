import numpy as np

from alzspeech.cv import leave_one_speaker_out, speaker_kfold


def _make_grouped_data(n_speakers=20, rows_per_speaker=3, seed=0):
    rng = np.random.default_rng(seed)
    speaker_ids = np.repeat(np.arange(n_speakers), rows_per_speaker)
    labels = np.repeat(rng.integers(0, 2, size=n_speakers), rows_per_speaker)
    return speaker_ids, labels


def test_speaker_kfold_no_leakage():
    speaker_ids, labels = _make_grouped_data()
    for train_idx, val_idx in speaker_kfold(speaker_ids, labels, n_splits=4):
        train_speakers = set(speaker_ids[train_idx])
        val_speakers = set(speaker_ids[val_idx])
        assert train_speakers.isdisjoint(val_speakers)
        assert len(val_idx) > 0
        assert len(train_idx) > 0


def test_speaker_kfold_covers_all_rows_across_folds():
    speaker_ids, labels = _make_grouped_data()
    all_val_idx = set()
    for _, val_idx in speaker_kfold(speaker_ids, labels, n_splits=4):
        all_val_idx.update(val_idx.tolist())
    assert all_val_idx == set(range(len(speaker_ids)))


def test_leave_one_speaker_out_no_leakage():
    speaker_ids, _ = _make_grouped_data(n_speakers=6, rows_per_speaker=2)
    n_folds = 0
    for train_idx, val_idx in leave_one_speaker_out(speaker_ids):
        val_speakers = set(speaker_ids[val_idx])
        assert len(val_speakers) == 1
        train_speakers = set(speaker_ids[train_idx])
        assert val_speakers.isdisjoint(train_speakers)
        n_folds += 1
    assert n_folds == 6


def test_speaker_kfold_without_labels_uses_group_kfold():
    speaker_ids, _ = _make_grouped_data()
    folds = list(speaker_kfold(speaker_ids, labels=None, n_splits=4))
    assert len(folds) == 4
    for train_idx, val_idx in folds:
        assert set(speaker_ids[train_idx]).isdisjoint(set(speaker_ids[val_idx]))
