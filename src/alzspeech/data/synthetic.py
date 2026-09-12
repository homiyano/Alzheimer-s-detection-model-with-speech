"""Synthetic data generator for DUA-free testing of the full pipeline.

Real ADReSSo/Pitt Corpus audio cannot be redistributed (see docs/DATA_ACCESS.md),
so unit tests and CI cannot ship real recordings. This module fabricates
structurally-similar (but content-free) data: short sine/noise wav files plus
templated transcripts, with a label-correlated signal (transcript length,
disfluency rate, pause count) loosely mirroring the direction of real AD vs.
control differences reported in the literature. It exists purely to exercise
the pipeline's plumbing (shapes, CV splitting, training loop) end-to-end — it
carries no diagnostic signal and must never be used to claim model performance.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_CN_WORDS = ["the", "boy", "is", "reaching", "for", "a", "cookie", "jar", "while", "his", "sister", "watches", "the", "mother", "is", "drying", "dishes", "and", "water", "is", "overflowing", "from", "the", "sink", "outside", "the", "window", "two", "children", "can", "be", "seen", "near", "a", "curb"]

_AD_WORDS = ["the", "boy", "um", "is", "um", "reaching", "for", "uh", "the", "thing", "the", "jar", "and", "uh", "the", "woman", "is", "uh", "washing", "and", "the", "the", "water", "is", "uh", "you", "know", "spilling", "and", "um", "there's", "a", "a", "thing", "outside", "I", "don't", "know"]

_FILLERS = {"um", "uh", "you know", "I don't know"}


def _synthesize_wav(path: Path, duration_s: float, sample_rate: int = 16000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    freq = rng.uniform(120, 260)
    tone = 0.1 * np.sin(2 * np.pi * freq * t)
    noise = 0.01 * rng.standard_normal(t.shape)
    audio = (tone + noise).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def _synthesize_transcript(label: str, rng: random.Random) -> str:
    base = list(_AD_WORDS if label == "ad" else _CN_WORDS)
    rng.shuffle(base)
    n_words = rng.randint(15, 30) if label == "ad" else rng.randint(25, 45)
    return " ".join(base[i % len(base)] for i in range(n_words))


def generate_synthetic_dataset(
    out_dir: str | Path,
    n_speakers_per_class: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic manifest + audio files under ``out_dir``.

    Returns the manifest DataFrame (also written to ``out_dir/manifest.csv``).
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    rng_py = random.Random(seed)
    rows = []

    speaker_idx = 0
    for label in ("ad", "cn"):
        for _ in range(n_speakers_per_class):
            speaker_id = f"S{speaker_idx:03d}_{label}"
            speaker_idx += 1
            duration = rng_py.uniform(8.0, 25.0)
            wav_path = audio_dir / f"{speaker_id}.wav"
            _synthesize_wav(wav_path, duration, seed=seed + speaker_idx)
            transcript = _synthesize_transcript(label, rng_py)
            mmse = rng_py.uniform(5, 22) if label == "ad" else rng_py.uniform(24, 30)
            split = "train" if rng_py.random() < 0.8 else "test"
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "audio_path": str(wav_path.relative_to(out_dir)),
                    "label": label,
                    "split": split,
                    "mmse": round(mmse, 1),
                    "dataset": "synthetic",
                    "transcript": transcript,
                }
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False)
    return df
