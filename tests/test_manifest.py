import pandas as pd
import pytest

from alzspeech.data.manifest import ManifestError, load_manifest, save_manifest, validate_manifest


def _valid_df(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    return pd.DataFrame(
        [
            {"speaker_id": "S1", "audio_path": "a.wav", "label": "AD", "split": "train"},
            {"speaker_id": "S2", "audio_path": "a.wav", "label": "cn", "split": "test"},
        ]
    )


def test_valid_manifest_passes(tmp_path):
    df = _valid_df(tmp_path)
    validate_manifest(df, base_dir=tmp_path, check_files_exist=True)


def test_missing_required_column_raises():
    df = pd.DataFrame([{"speaker_id": "S1", "label": "ad"}])
    with pytest.raises(ManifestError):
        validate_manifest(df, check_files_exist=False)


def test_invalid_label_raises(tmp_path):
    df = _valid_df(tmp_path)
    df.loc[0, "label"] = "maybe"
    with pytest.raises(ManifestError):
        validate_manifest(df, base_dir=tmp_path)


def test_inconsistent_speaker_label_raises(tmp_path):
    df = _valid_df(tmp_path)
    df["speaker_id"] = "S1"  # same speaker, conflicting labels (ad vs cn)
    with pytest.raises(ManifestError):
        validate_manifest(df, base_dir=tmp_path)


def test_missing_audio_file_raises(tmp_path):
    df = _valid_df(tmp_path)
    df.loc[0, "audio_path"] = "does_not_exist.wav"
    with pytest.raises(ManifestError):
        validate_manifest(df, base_dir=tmp_path, check_files_exist=True)


def test_load_and_save_roundtrip(tmp_path):
    df = _valid_df(tmp_path)
    out = tmp_path / "manifest.csv"
    save_manifest(df, out)
    loaded = load_manifest(out, check_files_exist=True)
    assert set(loaded["label"]) == {"ad", "cn"}
