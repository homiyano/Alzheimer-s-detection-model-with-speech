"""Manifest schema, loading, and validation.

A manifest is a CSV describing where audio lives and its labels, without
containing any protected data itself — see docs/DATA_ACCESS.md for how to
build one from a TalkBank-provided ADReSSo/Pitt Corpus download.

Required columns:
    speaker_id   str   stable per-participant identifier (never split across CV folds)
    audio_path   str   path to a mono wav file, absolute or relative to the manifest file
    label        str   "ad" (Alzheimer's/dementia) or "cn" (cognitively normal control)

Optional columns:
    split        str   "train" / "test", if the source dataset defines an official split
    mmse         float MMSE score (0-30), for the regression task
    dataset      str   source corpus name, e.g. "adresso", "pitt"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("speaker_id", "audio_path", "label")
OPTIONAL_COLUMNS = ("split", "mmse", "dataset")
VALID_LABELS = {"ad", "cn"}


class ManifestError(ValueError):
    """Raised when a manifest fails schema or consistency validation."""


def load_manifest(path: str | Path, check_files_exist: bool = True) -> pd.DataFrame:
    """Load and validate a manifest CSV, returning a normalized DataFrame."""
    path = Path(path)
    df = pd.read_csv(path)
    validate_manifest(df, base_dir=path.parent, check_files_exist=check_files_exist)
    df = df.copy()
    df["label"] = df["label"].str.lower()
    return df


def validate_manifest(
    df: pd.DataFrame, base_dir: str | Path | None = None, check_files_exist: bool = True
) -> None:
    """Validate a manifest DataFrame in place; raises ManifestError on failure."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest is missing required columns: {missing}")

    if df["speaker_id"].isna().any():
        raise ManifestError("Manifest has rows with a missing speaker_id")

    if df["audio_path"].isna().any():
        raise ManifestError("Manifest has rows with a missing audio_path")

    labels = df["label"].str.lower()
    bad_labels = set(labels.unique()) - VALID_LABELS
    if bad_labels:
        raise ManifestError(f"Manifest has invalid label values {bad_labels}; expected one of {VALID_LABELS}")

    # A given speaker must not have conflicting labels across rows.
    label_counts = df.assign(label=labels).groupby("speaker_id")["label"].nunique()
    inconsistent = label_counts[label_counts > 1]
    if not inconsistent.empty:
        raise ManifestError(
            f"Speakers with inconsistent labels across rows: {inconsistent.index.tolist()}"
        )

    duplicates = df.duplicated(subset=["speaker_id", "audio_path"])
    if duplicates.any():
        raise ManifestError(
            f"Manifest has duplicate (speaker_id, audio_path) rows at index {df.index[duplicates].tolist()}"
        )

    if "split" in df.columns:
        bad_splits = set(df["split"].dropna().unique()) - {"train", "test", "val"}
        if bad_splits:
            raise ManifestError(f"Manifest has invalid split values {bad_splits}")

    if check_files_exist:
        base_dir = Path(base_dir) if base_dir is not None else Path(".")
        missing_files = []
        for p in df["audio_path"]:
            fp = Path(p)
            if not fp.is_absolute():
                fp = base_dir / fp
            if not fp.exists():
                missing_files.append(str(p))
        if missing_files:
            raise ManifestError(f"Manifest references {len(missing_files)} missing audio file(s), e.g. {missing_files[:5]}")


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
