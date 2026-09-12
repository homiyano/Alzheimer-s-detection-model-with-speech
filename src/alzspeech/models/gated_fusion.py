"""Pick 1: gated multimodal fusion of frozen acoustic embeddings + fine-tuned text.

Template: "Listening Between the Lines" (arXiv:2606.30675), F1 90.14% on
ADReSSo — see docs/research/sota_papers.md section 3.6 and
docs/ARCHITECTURE.md. The text branch is a fine-tunable BERT/RoBERTa encoder
(reusing the Pick-2 architecture); the acoustic branch consumes *precomputed,
frozen* Whisper-encoder frame features (see
alzspeech.features.audio_features.FrozenWhisperEncoder) so no SSL encoder
weights are ever updated during training of this model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from alzspeech.models.pooling import AttentionPooling

DEFAULT_TEXT_MODEL = "bert-base-uncased"


class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        audio_hidden_size: int,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        fusion_dim: int = 256,
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

        self.audio_pool = AttentionPooling(audio_hidden_size)
        self.text_proj = nn.Linear(text_hidden_size, fusion_dim)
        self.audio_proj = nn.Linear(audio_hidden_size, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_frames: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids/attention_mask: (B, L) tokenized transcript.
        audio_frames: (B, T, audio_hidden_size) precomputed frozen encoder frames.
        audio_mask: (B, T) bool, True = valid frame.
        """
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0]  # (B, D_text)

        audio_pooled = self.audio_pool(audio_frames, audio_mask)  # (B, D_audio)

        text_proj = self.text_proj(text_cls)
        audio_proj = self.audio_proj(audio_pooled)

        gate = torch.sigmoid(self.gate(torch.cat([text_proj, audio_proj], dim=-1)))
        fused = gate * text_proj + (1 - gate) * audio_proj

        logits = self.classifier(fused)
        return logits
