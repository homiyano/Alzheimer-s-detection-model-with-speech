"""Pick 2: text-only fine-tuned transformer with disfluency/pause encoding.

Template: Yuan et al. 2021 (arXiv:2106.08689), 89.6% accuracy on ADReSS.
Transcript is expected to already have `[PAUSE]` tokens inserted by
`alzspeech.features.text_features.insert_pause_tokens` before tokenization.
This is the lowest-engineering-risk model in the project and is also required
as an ablation baseline for `GatedFusionClassifier`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from alzspeech.features.text_features import PAUSE_TOKEN

DEFAULT_MODEL = "bert-base-uncased"


def prepare_tokenizer_and_model(model_name: str = DEFAULT_MODEL, num_labels: int = 2):
    """Load a tokenizer/model pair with `[PAUSE]` registered as a real special
    token (not split into wordpieces) and the model's embedding table resized
    to match.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


class TextDisfluencyClassifier(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL, num_labels: int = 2, model_obj: Any = None):
        super().__init__()
        if model_obj is not None:
            self.encoder = model_obj
        else:
            from transformers import AutoModelForSequenceClassification

            self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @property
    def config(self):
        return self.encoder.config
