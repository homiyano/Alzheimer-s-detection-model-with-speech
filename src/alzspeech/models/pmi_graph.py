"""Pick 3: PMI co-occurrence graph augmentation of the text baseline.

Template: Gated Multi-Graph Fusion via GAT (Xiao et al. 2026, arXiv:2606.31186),
scoped to just its single highest-ablation-value component — the PMI
co-occurrence graph — per docs/ARCHITECTURE.md ("Why not X"). Implemented as a
small pure-PyTorch single-head graph attention layer (no torch_geometric
dependency) since per-sample graphs here have a handful of nodes.

Graphs are batched by looping over the batch dimension in Python: dataset
scale here (~166 training speakers) makes this a reasonable simplicity/speed
tradeoff over padding-and-masking a proper sparse batch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """Minimal single-head GAT layer operating on one graph at a time.

    node_features: (N, D_in). edge_index: (2, E) source/target node indices.
    Returns updated node features (N, D_out).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = node_features.size(0)
        h = self.W(node_features)  # (N, D_out)

        src, dst = edge_index[0], edge_index[1]
        e = self.leaky_relu(self.attn(torch.cat([h[src], h[dst]], dim=-1))).squeeze(-1)  # (E,)

        # Softmax normalize attention scores per destination node.
        e_max = e.max().detach() if e.numel() > 0 else torch.tensor(0.0)
        e_exp = torch.exp(e - e_max)
        denom = torch.zeros(n, device=h.device, dtype=h.dtype).scatter_add_(0, dst, e_exp) + 1e-12
        alpha = e_exp / denom[dst]

        out = torch.zeros_like(h).index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return torch.relu(out + h)  # residual


class PMIGraphClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        text_model_name: str = "bert-base-uncased",
        node_embed_dim: int = 64,
        graph_hidden_dim: int = 64,
        num_labels: int = 2,
        text_encoder_obj: Any = None,
    ):
        super().__init__()
        if text_encoder_obj is not None:
            self.text_encoder = text_encoder_obj
        else:
            from transformers import AutoModel

            self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        self.node_embedding = nn.Embedding(vocab_size, node_embed_dim, padding_idx=0)
        self.gat = GraphAttentionLayer(node_embed_dim, graph_hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(text_hidden_size + graph_hidden_dim, num_labels),
        )

    def _pool_graph(self, node_ids: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_features = self.node_embedding(node_ids)  # (N, D)
        updated = self.gat(node_features, edge_index)  # (N, D)
        return updated.mean(dim=0)  # (D,) global mean pool

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_ids_list: list[torch.Tensor],
        edge_index_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        input_ids/attention_mask: (B, L) tokenized transcript.
        node_ids_list: length-B list of (N_i,) LongTensors, per-sample graph node vocab ids.
        edge_index_list: length-B list of (2, E_i) LongTensors, per-sample graph edges.
        """
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0]  # (B, D_text)

        graph_reprs = [
            self._pool_graph(node_ids, edge_index)
            for node_ids, edge_index in zip(node_ids_list, edge_index_list)
        ]
        graph_repr = torch.stack(graph_reprs, dim=0)  # (B, D_graph)

        fused = torch.cat([text_cls, graph_repr], dim=-1)
        return self.classifier(fused)
