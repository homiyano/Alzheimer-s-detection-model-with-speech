#!/usr/bin/env python
"""Speaker-level k-fold CV training for any of the three models, plus the
mandatory silence-only control run on identical folds for comparison.

Usage:
    python scripts/train.py --config configs/text_baseline.yaml \
        --features-dir data/processed --output-dir outputs/text_baseline

Always trains and reports the silence-only control classifier (see
alzspeech.models.silence_control) on the same speaker folds as the real
model, per the Clever Hans guardrail in docs/ARCHITECTURE.md. If the real
model's accuracy does not clearly exceed the control's, treat the result
with suspicion rather than as a genuine result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from alzspeech.cv import speaker_kfold
from alzspeech.data.manifest import load_manifest
from alzspeech.features.pmi_graph import PMILookup, build_graph
from alzspeech.models.gated_fusion import GatedFusionClassifier
from alzspeech.models.pmi_graph import PMIGraphClassifier
from alzspeech.models.silence_control import SilenceControlClassifier
from alzspeech.models.text_baseline import prepare_tokenizer_and_model
from alzspeech.training.datasets import (
    FusionDataset,
    GraphDataset,
    SilenceDataset,
    TextDataset,
    collate_fusion,
    collate_graph,
    collate_silence,
    collate_text,
)
from alzspeech.training.forward_fns import fusion_forward, graph_forward, silence_forward, text_forward
from alzspeech.training.loop import train_and_evaluate


def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_silence_features(features_dir: Path) -> dict[str, np.ndarray]:
    npz = np.load(features_dir / "silence_features.npz")
    return {k: npz[k] for k in npz.files}


def run_silence_control(df: pd.DataFrame, features_dir: Path, cfg: dict, device: str) -> list[dict]:
    silence_features = load_silence_features(features_dir)
    speaker_ids = df["speaker_id"].to_numpy()
    labels = (df["label"] == "ad").astype(int).to_numpy()

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(speaker_kfold(speaker_ids, labels, n_splits=cfg["n_splits"], seed=cfg["seed"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        model = SilenceControlClassifier()
        train_loader = DataLoader(SilenceDataset(train_df, silence_features), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_silence)
        val_loader = DataLoader(SilenceDataset(val_df, silence_features), batch_size=cfg["batch_size"], collate_fn=collate_silence)
        result = train_and_evaluate(model, train_loader, val_loader, silence_forward, epochs=cfg["epochs"], lr=cfg["lr"], device=device)
        fold_results.append({"fold": fold, **result["val_metrics"]})
    return fold_results


def run_text(df: pd.DataFrame, cfg: dict, device: str) -> tuple[list[dict], torch.nn.Module, dict]:
    speaker_ids = df["speaker_id"].to_numpy()
    labels = (df["label"] == "ad").astype(int).to_numpy()

    fold_results = []
    model = None
    for fold, (train_idx, val_idx) in enumerate(speaker_kfold(speaker_ids, labels, n_splits=cfg["n_splits"], seed=cfg["seed"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        tokenizer, hf_model = prepare_tokenizer_and_model(cfg["model_name"])
        from alzspeech.models.text_baseline import TextDisfluencyClassifier

        model = TextDisfluencyClassifier(model_obj=hf_model)
        train_loader = DataLoader(
            TextDataset(train_df, tokenizer, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_text,
        )
        val_loader = DataLoader(
            TextDataset(val_df, tokenizer, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_text,
        )
        result = train_and_evaluate(model, train_loader, val_loader, text_forward, epochs=cfg["epochs"], lr=cfg["lr"], device=device)
        fold_results.append({"fold": fold, **result["val_metrics"]})
    return fold_results, model, {"vocab_size": len(tokenizer)}


def run_fusion(df: pd.DataFrame, features_dir: Path, cfg: dict, device: str) -> tuple[list[dict], torch.nn.Module, dict]:
    audio_dir = features_dir / cfg["audio_features_dir"]
    audio_features = {sid: np.load(audio_dir / f"{sid}.npy") for sid in df["speaker_id"]}
    audio_hidden = next(iter(audio_features.values())).shape[1]

    speaker_ids = df["speaker_id"].to_numpy()
    labels = (df["label"] == "ad").astype(int).to_numpy()

    fold_results = []
    model = None
    for fold, (train_idx, val_idx) in enumerate(speaker_kfold(speaker_ids, labels, n_splits=cfg["n_splits"], seed=cfg["seed"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        tokenizer, text_encoder = prepare_tokenizer_and_model(cfg["text_model_name"])
        text_encoder = text_encoder.base_model  # bare encoder (no classification head) with resized embeddings
        model = GatedFusionClassifier(audio_hidden_size=audio_hidden, fusion_dim=cfg["fusion_dim"], text_encoder_obj=text_encoder)
        train_loader = DataLoader(
            FusionDataset(train_df, tokenizer, audio_features, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fusion,
        )
        val_loader = DataLoader(
            FusionDataset(val_df, tokenizer, audio_features, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_fusion,
        )
        result = train_and_evaluate(model, train_loader, val_loader, fusion_forward, epochs=cfg["epochs"], lr=cfg["lr"], device=device)
        fold_results.append({"fold": fold, **result["val_metrics"]})
    return fold_results, model, {"audio_hidden_size": audio_hidden, "vocab_size": len(tokenizer)}


def run_graph(df: pd.DataFrame, cfg: dict, device: str) -> tuple[list[dict], torch.nn.Module, dict]:
    speaker_ids = df["speaker_id"].to_numpy()
    labels = (df["label"] == "ad").astype(int).to_numpy()

    fold_results = []
    model = None
    vocab: dict[str, int] = {}
    for fold, (train_idx, val_idx) in enumerate(speaker_kfold(speaker_ids, labels, n_splits=cfg["n_splits"], seed=cfg["seed"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        # Reference corpus for PMI is the CN (healthy) subset of this fold's
        # training data only, never validation data.
        reference_corpus = train_df.loc[train_df["label"] == "cn", cfg["text_column"]].tolist()
        lookup = PMILookup(window=cfg["pmi_window"]).fit(reference_corpus)

        vocab = {"<pad>": 0}
        graphs = {}
        for _, row in df.iterrows():
            node_words, edge_index, _ = build_graph(row[cfg["text_column"]], lookup, threshold=cfg["pmi_threshold"], window=cfg["pmi_window"])
            node_ids = []
            for w in node_words:
                if w not in vocab:
                    vocab[w] = len(vocab)
                node_ids.append(vocab[w])
            graphs[row["speaker_id"]] = (np.array(node_ids, dtype=np.int64), edge_index)

        tokenizer, text_encoder = prepare_tokenizer_and_model(cfg["text_model_name"])
        text_encoder = text_encoder.base_model
        model = PMIGraphClassifier(
            vocab_size=len(vocab), text_encoder_obj=text_encoder,
            node_embed_dim=cfg["node_embed_dim"], graph_hidden_dim=cfg["graph_hidden_dim"],
        )
        train_loader = DataLoader(
            GraphDataset(train_df, tokenizer, graphs, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_graph,
        )
        val_loader = DataLoader(
            GraphDataset(val_df, tokenizer, graphs, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_graph,
        )
        result = train_and_evaluate(model, train_loader, val_loader, graph_forward, epochs=cfg["epochs"], lr=cfg["lr"], device=device)
        fold_results.append({"fold": fold, **result["val_metrics"]})
    return fold_results, model, {"vocab": vocab}


def summarize(fold_results: list[dict], label: str) -> dict:
    metrics = [k for k in fold_results[0].keys() if k not in ("fold", "accuracy_ci")]
    summary = {"label": label}
    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path, help="Output dir from scripts/extract_features.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-silence-control", action="store_true", help="Not recommended — see module docstring")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.features_dir / "manifest.csv", check_files_exist=False)
    train_df = df[df.get("split", "train") == "train"].reset_index(drop=True)
    print(f"Loaded {len(train_df)} training rows ({train_df['speaker_id'].nunique()} speakers) for CV.")

    dispatch = {"text": run_text, "fusion": lambda d, c, dev: run_fusion(d, args.features_dir, c, dev), "graph": run_graph}
    if cfg["model_type"] not in dispatch:
        raise SystemExit(f"Unknown model_type {cfg['model_type']!r}; expected one of {list(dispatch)}")

    fold_results, last_model, extra = dispatch[cfg["model_type"]](train_df, cfg, device)
    summary = summarize(fold_results, cfg["model_type"])
    print(json.dumps(summary, indent=2))

    # NOTE: this saves the model from the *last* CV fold, trained on only
    # (n_splits-1)/n_splits of the training data — CV is for validating the
    # approach, not producing a deployable checkpoint. Retrain on the full
    # training split (see docs/ARCHITECTURE.md) before shipping a checkpoint.
    checkpoint = {"model_type": cfg["model_type"], "config": cfg, "state_dict": last_model.state_dict(), **extra}
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    if not args.skip_silence_control:
        silence_cfg = {"n_splits": cfg["n_splits"], "seed": cfg["seed"], "batch_size": cfg.get("batch_size", 8), "epochs": 30, "lr": 1e-2}
        silence_folds = run_silence_control(train_df, args.features_dir, silence_cfg, device)
        silence_summary = summarize(silence_folds, "silence_control")
        print(json.dumps(silence_summary, indent=2))

        real_acc = summary["accuracy"]["mean"]
        control_acc = silence_summary["accuracy"]["mean"]
        if real_acc <= control_acc + 0.02:
            print(
                f"WARNING: {cfg['model_type']} model accuracy ({real_acc:.3f}) is not clearly above the "
                f"silence-only control ({control_acc:.3f}). See docs/research/sota_papers.md pitfall #3 "
                "(Clever Hans silence artifact) before trusting this result."
            )

    (args.output_dir / "cv_results.json").write_text(
        json.dumps({"model": summary, "silence_control": silence_summary if not args.skip_silence_control else None}, indent=2)
    )
    print(f"Wrote CV results to {args.output_dir / 'cv_results.json'}")


if __name__ == "__main__":
    main()
