from pathlib import Path

from alzspeech.data.manifest import validate_manifest
from alzspeech.data.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset(tmp_path):
    df = generate_synthetic_dataset(tmp_path, n_speakers_per_class=3, seed=1)
    assert len(df) == 6
    assert set(df["label"]) == {"ad", "cn"}
    assert df["speaker_id"].is_unique

    for rel_path in df["audio_path"]:
        assert (tmp_path / rel_path).exists()

    validate_manifest(df, base_dir=tmp_path, check_files_exist=True)
    assert (tmp_path / "manifest.csv").exists()


def test_synthetic_is_deterministic_given_seed(tmp_path):
    df1 = generate_synthetic_dataset(tmp_path / "a", n_speakers_per_class=2, seed=7)
    df2 = generate_synthetic_dataset(tmp_path / "b", n_speakers_per_class=2, seed=7)
    assert df1["transcript"].tolist() == df2["transcript"].tolist()
    assert df1["label"].tolist() == df2["label"].tolist()
