"""torch Datasets + collate functions for each model type.

Label convention throughout: "ad" -> 1 (positive class), "cn" -> 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

LABEL_TO_ID = {"cn": 0, "ad": 1}


class TextDataset(Dataset):
    """Tokenized transcript classification, for TextDisfluencyClassifier and
    the text branch of GatedFusionClassifier/PMIGraphClassifier.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, text_column: str = "transcript", max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row[self.text_column],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long),
        }


def collate_text(batch: list[dict]) -> dict:
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


class FusionDataset(Dataset):
    """Text tokens + precomputed frozen audio frame features, for
    GatedFusionClassifier.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        audio_features: dict[str, np.ndarray],
        text_column: str = "transcript",
        max_length: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.audio_features = audio_features
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row[self.text_column], truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        audio = torch.from_numpy(self.audio_features[row["speaker_id"]]).float()
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "audio_frames": audio,
            "label": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long),
        }


def collate_fusion(batch: list[dict]) -> dict:
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])

    audio_lens = [b["audio_frames"].shape[0] for b in batch]
    max_len = max(audio_lens)
    hidden = batch[0]["audio_frames"].shape[1]
    audio_frames = torch.zeros(len(batch), max_len, hidden)
    audio_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        length = b["audio_frames"].shape[0]
        audio_frames[i, :length] = b["audio_frames"]
        audio_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_frames": audio_frames,
        "audio_mask": audio_mask,
        "label": labels,
    }


class GraphDataset(Dataset):
    """Text tokens + precomputed PMI graph, for PMIGraphClassifier."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        graphs: dict[str, tuple[np.ndarray, np.ndarray]],
        text_column: str = "transcript",
        max_length: int = 128,
    ):
        """``graphs`` maps speaker_id -> (node_ids [N], edge_index [2, E])."""
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.graphs = graphs
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row[self.text_column], truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        node_ids, edge_index = self.graphs[row["speaker_id"]]
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "node_ids": torch.from_numpy(node_ids).long(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "label": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long),
        }


def collate_graph(batch: list[dict]) -> dict:
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "node_ids_list": [b["node_ids"] for b in batch],
        "edge_index_list": [b["edge_index"] for b in batch],
        "label": labels,
    }


class SilenceDataset(Dataset):
    """Silence-only summary features, for SilenceControlClassifier."""

    def __init__(self, df: pd.DataFrame, silence_features: dict[str, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.silence_features = silence_features

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        features = torch.from_numpy(self.silence_features[row["speaker_id"]]).float()
        return {"features": features, "label": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long)}


def collate_silence(batch: list[dict]) -> dict:
    features = torch.stack([b["features"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"features": features, "label": labels}
