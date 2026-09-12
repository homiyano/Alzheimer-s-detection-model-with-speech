"""Shared fixtures: tiny randomly-initialized transformer configs so model
tests exercise real forward-pass shapes/logic without downloading any
pretrained weights (fast, network-independent, safe for CI).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def tiny_bert_model_for_classification():
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=2,
    )
    return BertForSequenceClassification(config)


@pytest.fixture
def tiny_bert_encoder():
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return BertModel(config)


@pytest.fixture
def tiny_whisper_hidden_size():
    return 8
