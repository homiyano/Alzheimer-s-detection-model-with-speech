from alzspeech.features.text_features import (
    PAUSE_TOKEN,
    Word,
    disfluency_features,
    insert_pause_tokens,
)


def test_insert_pause_tokens_inserts_at_large_gaps():
    words = [
        Word("the", 0.0, 0.2),
        Word("boy", 0.25, 0.5),
        Word("um", 1.5, 1.7),  # 1.0s gap -> pause before this
        Word("jar", 1.75, 2.0),
    ]
    text = insert_pause_tokens(words, pause_threshold_s=0.5)
    assert text == f"the boy {PAUSE_TOKEN} um jar"


def test_insert_pause_tokens_empty():
    assert insert_pause_tokens([]) == ""


def test_disfluency_features_basic():
    feats = disfluency_features("the the boy um is uh reaching")
    assert feats["n_words"] == 7
    assert feats["filler_count"] == 2
    assert feats["filler_rate"] == 2 / 7
    assert feats["repeated_word_rate"] > 0  # "the the" repeat


def test_disfluency_features_empty_string():
    feats = disfluency_features("")
    assert feats["n_words"] == 0
    assert feats["filler_rate"] == 0.0
