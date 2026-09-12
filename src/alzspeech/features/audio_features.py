"""Frozen acoustic feature extraction via a pretrained Whisper encoder.

The research (docs/ARCHITECTURE.md "Why not X") explicitly avoids fine-tuning
a large SSL audio encoder end-to-end on ~166 training speakers. Instead we run
the encoder frozen (no gradient, eval mode) and only train a small pooling +
classification head on top of its frame-level hidden states — matching the
"Listening Between the Lines" (arXiv:2606.30675) and HAFFormer (arXiv:2405.03952)
templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "openai/whisper-base"
WHISPER_SAMPLE_RATE = 16000


class FrozenWhisperEncoder:
    """Extracts frame-level hidden states from a frozen Whisper encoder.

    Pass ``model_obj``/``feature_extractor_obj`` directly to avoid downloading
    real weights in tests (e.g. a randomly-initialized tiny WhisperModel).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        model_obj: Any = None,
        feature_extractor_obj: Any = None,
    ):
        self.model_name = model_name
        self._device_arg = device
        self._model = model_obj
        self._feature_extractor = feature_extractor_obj
        self._device = None

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        return self._model.config.d_model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            if self._device is None:
                self._device = self._device_arg or "cpu"
            return

        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        device = self._device_arg
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
        self._model = WhisperModel.from_pretrained(self.model_name).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def encode(self, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
        """Return frame-level encoder hidden states, shape (T, hidden_size)."""
        import torch

        self._ensure_loaded()
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(f"Expected {WHISPER_SAMPLE_RATE} Hz audio, got {sample_rate}")

        inputs = self._feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(self._device)
        with torch.no_grad():
            encoder_out = self._model.encoder(input_features)
        hidden = encoder_out.last_hidden_state[0]  # (T, D)
        return hidden.cpu().numpy()

    def encode_file(self, audio_path: str | Path) -> np.ndarray:
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=WHISPER_SAMPLE_RATE, mono=True)
        return self.encode(audio, sample_rate=sr)
