"""End-to-end smoke tests: synthetic data -> features -> train -> evaluate,
for all three models plus the silence control. Uses a real (small) BERT
tokenizer for realistic tokenization, but tiny randomly-initialized encoder
weights so nothing downloads a large checkpoint or trains slowly. This is
the automated substitute for real-data validation, which requires a
TalkBank DUA the test environment does not have (docs/DATA_ACCESS.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from torch.utils.data import DataLoader

from alzspeech.cv import speaker_kfold
from alzspeech.data.synthetic import generate_synthetic_dataset
from alzspeech.features.pmi_graph import PMILookup, build_graph
from alzspeech.features.text_features import Word
from alzspeech.models.gated_fusion import GatedFusionClassifier
from alzspeech.models.pmi_graph import PMIGraphClassifier
from alzspeech.models.silence_control import (
    SilenceControlClassifier,
    compute_silence_features,
)
from alzspeech.models.text_baseline import TextDisfluencyClassifier
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
from alzspeech.training.forward_fns import (
    fusion_forward,
    graph_forward,
    silence_forward,
    text_forward,
)
from alzspeech.training.loop import train_and_evaluate


@pytest.fixture(scope="module")
def synthetic_df(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("synthetic")
    return generate_synthetic_dataset(out_dir, n_speakers_per_class=6, seed=3), out_dir


@pytest.fixture(scope="module")
def bert_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def _tiny_bert_for_vocab(vocab_size: int):
    from transformers import BertConfig, BertForSequenceClassification, BertModel

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
    )
    return config, BertModel, BertForSequenceClassification


def _train_val_split(df):
    speaker_ids = df["speaker_id"].to_numpy()
    labels = (df["label"] == "ad").astype(int).to_numpy()
    train_idx, val_idx = next(speaker_kfold(speaker_ids, labels, n_splits=3, seed=0))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def test_text_baseline_end_to_end(synthetic_df, bert_tokenizer):
    df, _ = synthetic_df
    train_df, val_df = _train_val_split(df)

    config, _, BertForSequenceClassification = _tiny_bert_for_vocab(bert_tokenizer.vocab_size)
    model = TextDisfluencyClassifier(model_obj=BertForSequenceClassification(config))

    train_ds = TextDataset(train_df, bert_tokenizer)
    val_ds = TextDataset(val_df, bert_tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_text)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_text)

    result = train_and_evaluate(model, train_loader, val_loader, text_forward, epochs=2, lr=1e-3, device="cpu")
    assert len(result["history"]) == 2
    assert 0.0 <= result["val_metrics"]["accuracy"] <= 1.0


def test_gated_fusion_end_to_end(synthetic_df, bert_tokenizer):
    df, _ = synthetic_df
    train_df, val_df = _train_val_split(df)

    audio_hidden = 8
    rng = np.random.default_rng(0)
    audio_features = {
        sid: rng.standard_normal((rng.integers(5, 15), audio_hidden)).astype(np.float32)
        for sid in df["speaker_id"]
    }

    config, BertModel, _ = _tiny_bert_for_vocab(bert_tokenizer.vocab_size)
    model = GatedFusionClassifier(audio_hidden_size=audio_hidden, text_encoder_obj=BertModel(config), fusion_dim=12)

    train_ds = FusionDataset(train_df, bert_tokenizer, audio_features)
    val_ds = FusionDataset(val_df, bert_tokenizer, audio_features)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fusion)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fusion)

    result = train_and_evaluate(model, train_loader, val_loader, fusion_forward, epochs=2, lr=1e-3, device="cpu")
    assert len(result["history"]) == 2
    assert 0.0 <= result["val_metrics"]["accuracy"] <= 1.0


def test_pmi_graph_end_to_end(synthetic_df, bert_tokenizer):
    df, _ = synthetic_df
    train_df, val_df = _train_val_split(df)

    reference_corpus = train_df.loc[train_df["label"] == "cn", "transcript"].tolist()
    lookup = PMILookup(window=3).fit(reference_corpus)

    vocab: dict[str, int] = {"<pad>": 0}
    graphs = {}
    for _, row in df.iterrows():
        node_words, edge_index, _ = build_graph(row["transcript"], lookup, threshold=-100.0)
        node_ids = []
        for w in node_words:
            if w not in vocab:
                vocab[w] = len(vocab)
            node_ids.append(vocab[w])
        graphs[row["speaker_id"]] = (np.array(node_ids, dtype=np.int64), edge_index)

    config, BertModel, _ = _tiny_bert_for_vocab(bert_tokenizer.vocab_size)
    model = PMIGraphClassifier(
        vocab_size=len(vocab), text_encoder_obj=BertModel(config), node_embed_dim=8, graph_hidden_dim=8
    )

    train_ds = GraphDataset(train_df, bert_tokenizer, graphs)
    val_ds = GraphDataset(val_df, bert_tokenizer, graphs)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_graph)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_graph)

    result = train_and_evaluate(model, train_loader, val_loader, graph_forward, epochs=2, lr=1e-3, device="cpu")
    assert len(result["history"]) == 2
    assert 0.0 <= result["val_metrics"]["accuracy"] <= 1.0


def test_silence_control_end_to_end(synthetic_df):
    df, out_dir = synthetic_df
    train_df, val_df = _train_val_split(df)

    silence_features = {}
    for _, row in df.iterrows():
        info = sf.info(str(out_dir / row["audio_path"]))
        duration = info.frames / info.samplerate
        # Fabricate plausible word timings spanning most of the clip.
        words = [Word("w", 0.0, duration * 0.4), Word("w", duration * 0.6, duration * 0.9)]
        silence_features[row["speaker_id"]] = compute_silence_features(words, duration)

    model = SilenceControlClassifier()
    train_ds = SilenceDataset(train_df, silence_features)
    val_ds = SilenceDataset(val_df, silence_features)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_silence)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_silence)

    result = train_and_evaluate(model, train_loader, val_loader, silence_forward, epochs=2, lr=1e-2, device="cpu")
    assert len(result["history"]) == 2
    assert 0.0 <= result["val_metrics"]["accuracy"] <= 1.0
