"""Attention pooling over variable-length frame sequences.

Naive mean-pooling over frozen encoder frames discards temporal salience;
"Listening Between the Lines" (arXiv:2606.30675) explicitly credits attention
pooling over Whisper frames as part of its acoustic-branch design.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) frame features. mask: (B, T) bool, True = valid frame."""
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (weights * x).sum(dim=1)  # (B, D)
        return pooled
