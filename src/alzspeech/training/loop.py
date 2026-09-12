"""Generic train/eval loop shared by all three models.

Each model type differs in how a batch maps to logits, so callers supply a
``forward_fn(model, batch, device) -> (logits, labels)`` closure; the loop
itself (optimizer step, metric computation, bootstrap CI, history tracking)
is identical across models. This keeps `scripts/train.py` dispatch-only and
avoids duplicating the training loop three times.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from alzspeech.metrics import bootstrap_ci, classification_metrics

ForwardFn = Callable[[nn.Module, dict, str], tuple[torch.Tensor, torch.Tensor]]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    forward_fn: ForwardFn,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        optimizer.zero_grad()
        logits, labels = forward_fn(model, batch, device)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, forward_fn: ForwardFn, device: str) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy() if all_preds else torch.empty(0).numpy()
    y_true = torch.cat(all_labels).numpy() if all_labels else torch.empty(0).numpy()

    metrics = classification_metrics(y_true, y_pred) if len(y_true) else {}
    _point, lower, upper = bootstrap_ci(y_true, y_pred)
    metrics["accuracy_ci"] = (lower, upper)
    return metrics


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    forward_fn: ForwardFn,
    epochs: int = 10,
    lr: float = 2e-5,
    device: str = "cpu",
) -> dict:
    """Run full training, returning {"history": [...], "val_metrics": {...}}."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, forward_fn, device)
        val_metrics = evaluate_epoch(model, val_loader, forward_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

    final_val_metrics = evaluate_epoch(model, val_loader, forward_fn, device)
    return {"history": history, "val_metrics": final_val_metrics}
