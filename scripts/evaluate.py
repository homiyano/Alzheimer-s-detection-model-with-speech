#!/usr/bin/env python
"""Evaluate a checkpoint from scripts/train.py on the held-out test split.

Always reports classification metrics with a bootstrap 95% CI (small-N
statistical power is a documented pitfall — see docs/research/sota_papers.md
pitfall #8) and the silence-only control's accuracy on the same test rows,
for comparison.

Note: the graph model's checkpoint stores the PMI vocabulary/lookup built
from its *last CV fold's* training data (see scripts/train.py docstring on
checkpoints) — a documented simplification, not a fully independent final
fit; treat graph-model test numbers here as indicative, not final-release
grade, until a dedicated full-train refit is added.

Usage:
    python scripts/evaluate.py --checkpoint outputs/text_baseline/checkpoint.pt \
        --features-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alzspeech.data.manifest import load_manifest
from alzspeech.features.pmi_graph import PMILookup, build_graph
from alzspeech.metrics import bootstrap_ci, classification_metrics
from alzspeech.models.gated_fusion import GatedFusionClassifier
from alzspeech.models.pmi_graph import PMIGraphClassifier
from alzspeech.models.text_baseline import TextDisfluencyClassifier
from alzspeech.training.datasets import (
    FusionDataset,
    GraphDataset,
    TextDataset,
    collate_fusion,
    collate_graph,
    collate_text,
)
from alzspeech.training.forward_fns import fusion_forward, graph_forward, text_forward
from alzspeech.training.loop import evaluate_epoch


def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model_and_loader(checkpoint: dict, test_df, features_dir: Path, tokenizer):
    cfg = checkpoint["config"]
    model_type = checkpoint["model_type"]

    if model_type == "text":
        from transformers import AutoModelForSequenceClassification

        hf_model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)
        hf_model.resize_token_embeddings(checkpoint["vocab_size"])
        model = TextDisfluencyClassifier(model_obj=hf_model)
        model.load_state_dict(checkpoint["state_dict"])
        loader = DataLoader(
            TextDataset(test_df, tokenizer, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_text,
        )
        return model, loader, text_forward

    if model_type == "fusion":
        from transformers import AutoModel

        text_encoder = AutoModel.from_pretrained(cfg["text_model_name"])
        text_encoder.resize_token_embeddings(checkpoint["vocab_size"])
        model = GatedFusionClassifier(
            audio_hidden_size=checkpoint["audio_hidden_size"], fusion_dim=cfg["fusion_dim"], text_encoder_obj=text_encoder
        )
        model.load_state_dict(checkpoint["state_dict"])
        audio_dir = features_dir / cfg["audio_features_dir"]
        audio_features = {sid: np.load(audio_dir / f"{sid}.npy") for sid in test_df["speaker_id"]}
        loader = DataLoader(
            FusionDataset(test_df, tokenizer, audio_features, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_fusion,
        )
        return model, loader, fusion_forward

    if model_type == "graph":
        from transformers import AutoModel

        vocab = checkpoint["vocab"]
        text_encoder = AutoModel.from_pretrained(cfg["text_model_name"])
        text_encoder.resize_token_embeddings(len(tokenizer))
        model = PMIGraphClassifier(
            vocab_size=len(vocab), text_encoder_obj=text_encoder,
            node_embed_dim=cfg["node_embed_dim"], graph_hidden_dim=cfg["graph_hidden_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])

        lookup = PMILookup(window=cfg["pmi_window"])  # empty fit -> all PMI scores 0.0, self-loop fallback
        graphs = {}
        for _, row in test_df.iterrows():
            node_words, edge_index, _ = build_graph(row[cfg["text_column"]], lookup, threshold=cfg["pmi_threshold"], window=cfg["pmi_window"])
            node_ids = [vocab.get(w, 0) for w in node_words]
            graphs[row["speaker_id"]] = (np.array(node_ids, dtype=np.int64), edge_index)

        loader = DataLoader(
            GraphDataset(test_df, tokenizer, graphs, text_column=cfg["text_column"], max_length=cfg["max_length"]),
            batch_size=cfg["batch_size"], collate_fn=collate_graph,
        )
        return model, loader, graph_forward

    raise ValueError(f"Unknown model_type {model_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    df = load_manifest(args.features_dir / "manifest.csv", check_files_exist=False)
    test_df = df[df.get("split", "test") == "test"].reset_index(drop=True)
    if len(test_df) == 0:
        raise SystemExit("No rows with split == 'test' in the manifest.")
    print(f"Evaluating on {len(test_df)} test rows ({test_df['speaker_id'].nunique()} speakers).")

    from alzspeech.models.text_baseline import prepare_tokenizer_and_model

    tokenizer, _ = prepare_tokenizer_and_model(cfg.get("model_name") or cfg.get("text_model_name"))

    model, loader, forward_fn = build_model_and_loader(checkpoint, test_df, args.features_dir, tokenizer)
    model.to(device)
    metrics = evaluate_epoch(model, loader, forward_fn, device)
    print("Model metrics:", json.dumps({k: v for k, v in metrics.items() if k != "accuracy_ci"}, indent=2))
    print("Accuracy 95% CI:", metrics["accuracy_ci"])
    print(
        "Compare this accuracy against the silence-only control's CV accuracy in "
        f"{args.checkpoint.parent / 'cv_results.json'} (from scripts/train.py) before "
        "trusting this result — see docs/research/sota_papers.md pitfall #3."
    )


if __name__ == "__main__":
    main()
