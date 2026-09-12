#!/usr/bin/env python
"""Build a manifest.csv from a TalkBank-provided ADReSSo-style directory tree.

Expects (adjust --ad-subdir/--cn-subdir if your download differs; the exact
layout has varied across ADReSSo challenge years):

    <root>/train/audio/ad/*.wav
    <root>/train/audio/cn/*.wav
    <root>/test/audio/ad/*.wav   (only if you have test labels)
    <root>/test/audio/cn/*.wav

Speaker id is derived from each wav file's stem. Does not read or copy any
audio content — see docs/DATA_ACCESS.md for how to obtain <root> itself via
a TalkBank DUA; this script only never touches or embeds the protected data
in a way that would violate the DUA (it references paths, not content).

Usage:
    python scripts/prepare_manifest.py --root /path/to/ADReSSo21 --out data/raw/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alzspeech.data.manifest import save_manifest, validate_manifest


def scan_split(root: Path, split: str, ad_subdir: str, cn_subdir: str) -> list[dict]:
    rows = []
    for label, subdir in (("ad", ad_subdir), ("cn", cn_subdir)):
        label_dir = root / split / "audio" / subdir
        if not label_dir.is_dir():
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            rows.append(
                {
                    "speaker_id": wav_path.stem,
                    "audio_path": str(wav_path.resolve()),
                    "label": label,
                    "split": split if split != "train" else "train",
                    "dataset": "adresso",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Root of the extracted ADReSSo download")
    parser.add_argument("--out", required=True, type=Path, help="Output manifest CSV path")
    parser.add_argument("--ad-subdir", default="ad", help="Subdirectory name for AD-labeled audio (default: ad)")
    parser.add_argument("--cn-subdir", default="cn", help="Subdirectory name for control-labeled audio (default: cn)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], help="Which split subdirectories to scan (default: train test)"
    )
    args = parser.parse_args()

    rows = []
    for split in args.splits:
        rows.extend(scan_split(args.root, split, args.ad_subdir, args.cn_subdir))

    if not rows:
        raise SystemExit(
            f"No audio files found under {args.root} for splits {args.splits}. "
            "Check --root/--ad-subdir/--cn-subdir match your actual download layout."
        )

    df = pd.DataFrame(rows)
    validate_manifest(df, check_files_exist=True)
    save_manifest(df, args.out)
    print(f"Wrote manifest with {len(df)} rows ({df['speaker_id'].nunique()} speakers) to {args.out}")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
