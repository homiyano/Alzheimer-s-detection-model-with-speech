#!/usr/bin/env python
"""Run ASR + feature extraction over a manifest, caching results to disk.

For every row: transcribe with Whisper (word-level timestamps), build a
pause-token-augmented transcript, compute cheap disfluency stats, compute
silence-only control features, and extract frozen Whisper-encoder frame
embeddings. Everything is cached under <out-dir>/ so training scripts never
need to re-run ASR or the (comparatively expensive) encoder forward pass.

Usage:
    python scripts/extract_features.py --manifest data/raw/manifest.csv --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from alzspeech.data.manifest import load_manifest, save_manifest
from alzspeech.features.asr import WhisperTranscriber
from alzspeech.features.audio_features import FrozenWhisperEncoder
from alzspeech.features.text_features import disfluency_features, insert_pause_tokens
from alzspeech.models.silence_control import compute_silence_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--asr-model", default="openai/whisper-base")
    parser.add_argument("--audio-encoder-model", default="openai/whisper-base")
    parser.add_argument("--pause-threshold-s", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    df = load_manifest(args.manifest, check_files_exist=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_feat_dir = args.out_dir / "audio_features"
    audio_feat_dir.mkdir(exist_ok=True)

    transcriber = WhisperTranscriber(model_name=args.asr_model, device=args.device)
    encoder = FrozenWhisperEncoder(model_name=args.audio_encoder_model, device=args.device)

    transcripts, transcripts_paused, disfluencies, silence_feats = [], [], [], {}

    manifest_dir = args.manifest.parent

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = manifest_dir / audio_path
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate

        text, words = transcriber.transcribe(audio_path)
        text_paused = insert_pause_tokens(words, pause_threshold_s=args.pause_threshold_s)
        disf = disfluency_features(text_paused)
        silence = compute_silence_features(words, duration, pause_threshold_s=args.pause_threshold_s)

        transcripts.append(text)
        transcripts_paused.append(text_paused)
        disfluencies.append(disf)
        silence_feats[row["speaker_id"]] = silence

        frames = encoder.encode_file(audio_path)
        np.save(audio_feat_dir / f"{row['speaker_id']}.npy", frames)

    df = df.copy()
    df["transcript"] = transcripts
    df["transcript_paused"] = transcripts_paused
    for key in disfluencies[0].keys() if disfluencies else []:
        df[f"disfluency_{key}"] = [d[key] for d in disfluencies]

    save_manifest(df, args.out_dir / "manifest.csv")
    np.savez(args.out_dir / "silence_features.npz", **{k: v for k, v in silence_feats.items()})

    with open(args.out_dir / "extraction_config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    print(f"Wrote features for {len(df)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
