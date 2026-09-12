import torch

from alzspeech.models.gated_fusion import GatedFusionClassifier
from alzspeech.models.pmi_graph import GraphAttentionLayer, PMIGraphClassifier
from alzspeech.models.pooling import AttentionPooling
from alzspeech.models.silence_control import SilenceControlClassifier
from alzspeech.models.text_baseline import TextDisfluencyClassifier


def test_text_disfluency_classifier_forward(tiny_bert_model_for_classification):
    model = TextDisfluencyClassifier(model_obj=tiny_bert_model_for_classification)
    input_ids = torch.randint(0, 99, (4, 10))
    attention_mask = torch.ones(4, 10, dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1])
    out = model(input_ids, attention_mask, labels=labels)
    assert out.logits.shape == (4, 2)
    assert out.loss.item() >= 0


def test_attention_pooling_respects_mask():
    pool = AttentionPooling(hidden_size=8)
    x = torch.randn(3, 5, 8)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, :2] = True  # only first two frames valid
    out = pool(x, mask)
    assert out.shape == (3, 8)
    # padded-out frames must not receive attention weight
    x_perturbed = x.clone()
    x_perturbed[:, 2:] += 1000.0
    out_perturbed = pool(x_perturbed, mask)
    assert torch.allclose(out, out_perturbed, atol=1e-4)


def test_gated_fusion_classifier_forward(tiny_bert_encoder, tiny_whisper_hidden_size):
    model = GatedFusionClassifier(
        audio_hidden_size=tiny_whisper_hidden_size,
        text_encoder_obj=tiny_bert_encoder,
        fusion_dim=12,
    )
    input_ids = torch.randint(0, 99, (4, 10))
    attention_mask = torch.ones(4, 10, dtype=torch.long)
    audio_frames = torch.randn(4, 20, tiny_whisper_hidden_size)
    audio_mask = torch.ones(4, 20, dtype=torch.bool)
    logits = model(input_ids, attention_mask, audio_frames, audio_mask)
    assert logits.shape == (4, 2)


def test_graph_attention_layer_shapes():
    layer = GraphAttentionLayer(in_dim=6, out_dim=6)
    node_features = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])  # includes a self-loop
    out = layer(node_features, edge_index)
    assert out.shape == (5, 6)


def test_graph_attention_layer_handles_empty_graph():
    layer = GraphAttentionLayer(in_dim=6, out_dim=6)
    node_features = torch.zeros(0, 6)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out = layer(node_features, edge_index)
    assert out.shape == (0, 6)


def test_pmi_graph_classifier_forward(tiny_bert_encoder):
    model = PMIGraphClassifier(vocab_size=50, text_encoder_obj=tiny_bert_encoder, node_embed_dim=6, graph_hidden_dim=6)
    input_ids = torch.randint(0, 99, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    node_ids_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    edge_index_list = [
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0], [1]]),
    ]
    logits = model(input_ids, attention_mask, node_ids_list, edge_index_list)
    assert logits.shape == (2, 2)


def test_silence_control_classifier_forward():
    model = SilenceControlClassifier()
    features = torch.randn(5, 4)
    logits = model(features)
    assert logits.shape == (5, 2)
