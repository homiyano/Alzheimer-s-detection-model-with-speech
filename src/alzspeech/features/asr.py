"""ASR wrapper: Whisper transcription with word-level timestamps.

ADReSSo provides no manual transcripts (docs/research/sota_papers.md, section
1.1), so every text-branch feature in this project starts from a
self-generated ASR transcript. Whisper is the ASR backbone most ADReSSo
practitioners actually use (section 2.3/5) and its word-level timestamps let
us reconstruct pause locations for `alzspeech.features.text_features`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alzspeech.features.text_features import Word

DEFAULT_MODEL = "openai/whisper-base"


class WhisperTranscriber:
    """Thin wrapper around a HF ASR pipeline, with lazy model loading.

    Pass ``pipeline_obj`` directly (e.g. a mock, or a pipeline built with a
    tiny randomly-initialized config) to avoid downloading real weights in
    tests.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, pipeline_obj: Any = None):
        self.model_name = model_name
        self.device = device
        self._pipeline = pipeline_obj

    def _get_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline

            device = self.device
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device,
                return_timestamps="word",
            )
        return self._pipeline

    def transcribe(self, audio_path: str | Path) -> tuple[str, list[Word]]:
        """Return (full_text, word_list) for a mono wav file."""
        result = self._get_pipeline()(str(audio_path))
        text = result.get("text", "").strip()
        words: list[Word] = []
        for chunk in result.get("chunks", []) or []:
            start, end = chunk.get("timestamp", (None, None))
            if start is None or end is None:
                continue
            words.append(Word(text=chunk["text"].strip(), start=float(start), end=float(end)))
        return text, words
