import numpy as np

from alzspeech.features.pmi_graph import PMILookup, build_graph


def test_pmi_lookup_higher_for_frequent_cooccurring_pairs():
    corpus = ["the boy reaches for the cookie"] * 20 + ["a random unrelated sentence here"] * 20
    lookup = PMILookup(window=3).fit(corpus)
    pmi_related = lookup.pmi("boy", "reaches")
    pmi_unrelated = lookup.pmi("boy", "random")
    assert pmi_related > pmi_unrelated


def test_build_graph_shapes():
    corpus = ["the boy reaches for the cookie jar"] * 10
    lookup = PMILookup(window=3).fit(corpus)
    node_words, edge_index, edge_weight = build_graph(
        "the boy reaches for the jar", lookup, threshold=-100.0
    )
    assert len(node_words) == len(set(node_words))
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_weight.shape[0]
    if edge_index.shape[1] > 0:
        assert edge_index.max() < len(node_words)


def test_build_graph_empty_transcript_has_no_nodes():
    lookup = PMILookup(window=3).fit(["some reference text"])
    node_words, edge_index, edge_weight = build_graph("", lookup)
    assert node_words == []
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)


def test_build_graph_falls_back_to_self_loops_when_no_edge_passes_threshold():
    lookup = PMILookup(window=3).fit(["completely different reference corpus"])
    node_words, edge_index, _edge_weight = build_graph("brand new unseen words here", lookup, threshold=100.0)
    assert len(node_words) > 0
    assert edge_index.shape[1] == len(node_words)  # one self-loop per node
    assert np.all(edge_index[0] == edge_index[1])
