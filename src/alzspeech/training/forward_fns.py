"""Per-model `forward_fn(model, batch, device) -> (logits, labels)` closures
consumed by `alzspeech.training.loop`.
"""

from __future__ import annotations

import torch
from torch import nn


def text_forward(model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    out = model(input_ids, attention_mask)
    return out.logits, labels


def fusion_forward(model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audio_frames = batch["audio_frames"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    labels = batch["label"].to(device)
    logits = model(input_ids, attention_mask, audio_frames, audio_mask)
    return logits, labels


def graph_forward(model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    node_ids_list = [n.to(device) for n in batch["node_ids_list"]]
    edge_index_list = [e.to(device) for e in batch["edge_index_list"]]
    labels = batch["label"].to(device)
    logits = model(input_ids, attention_mask, node_ids_list, edge_index_list)
    return logits, labels


def silence_forward(model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    features = batch["features"].to(device)
    labels = batch["label"].to(device)
    logits = model(features)
    return logits, labels
