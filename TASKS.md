# Task List

Derived from [docs/research/](docs/research/) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Phase 0 — Research (done)
- [x] Survey SOTA papers on ADReSS/ADReSSo Alzheimer's speech detection
- [x] Survey Hugging Face models/datasets and GitHub repos for existing open-source solutions
- [x] Write architecture/design-decision doc

## Phase 1 — Project foundation (done)
- [x] Repo scaffolding (.gitignore, directory layout)
- [x] `pyproject.toml` / `requirements.txt` with pinned core deps
- [x] `docs/DATA_ACCESS.md` — TalkBank/DementiaBank DUA request process, manifest format spec
- [x] `alzspeech.cv` — speaker-level k-fold / leave-one-speaker-out splitting utilities + tests
- [x] `alzspeech.data.manifest` — manifest schema + loader/validator
- [x] `alzspeech.data.synthetic` — synthetic audio/transcript generator for DUA-free testing

## Phase 2 — Feature extraction (done)
- [x] `alzspeech.features.asr` — Whisper transcription wrapper with word-level timestamps
- [x] `alzspeech.features.text_features` — pause/filler token insertion, disfluency & lexical-richness features
- [x] `alzspeech.features.audio_features` — frozen Whisper-encoder embedding extraction (attention pooling)
- [x] `scripts/prepare_manifest.py`, `scripts/extract_features.py`

## Phase 3 — Models (done)
- [x] `alzspeech.models.SilenceControlClassifier` — Clever-Hans sanity-check baseline
- [x] `alzspeech.models.TextDisfluencyClassifier` — Pick 2: BERT/RoBERTa + pause tokens
- [x] `alzspeech.models.GatedFusionClassifier` — Pick 1: frozen acoustic embeddings + text, gated fusion
- [x] `alzspeech.models.PMIGraphClassifier` — Pick 3: PMI co-occurrence graph + GAT augmentation

## Phase 4 — Training & evaluation (done)
- [x] `alzspeech.training` — generic speaker-CV training loop, config-driven
- [x] `alzspeech.metrics` — accuracy/F1/precision/recall/specificity + bootstrap CI, MMSE RMSE
- [x] `scripts/train.py`, `scripts/evaluate.py` CLIs
- [x] `configs/*.yaml` for each model

## Phase 5 — Tests & verification (done)
- [x] Unit tests for CV splitting (no speaker leakage), manifest validation, feature extraction shapes
- [x] End-to-end smoke test: synthetic data -> features -> train (multi-epoch) -> evaluate, for all 3 models + silence control
- [x] Silence-control comparison wired into training/eval scripts
- [x] All 4 CLI scripts (prepare_manifest, extract_features, train, evaluate) manually smoke-tested end-to-end against synthetic data with real (small) HuggingFace checkpoints on CPU — all passed, including the leakage/Clever-Hans guardrail firing correctly on near-chance synthetic accuracy

## Phase 6 — Documentation & release hygiene (done)
- [x] `README.md` — project overview, setup, usage, research summary, status
- [x] `docs/MODEL_CARD.md` template per Hugging Face model card conventions
- [x] `LICENSE` check (MIT already present) + explicit note on data licensing (TalkBank DUA, no redistribution) in README/DATA_ACCESS
- [x] `CONTRIBUTING.md`

## What's left — requires real data access (outside this session's reach)
- [ ] Obtain a TalkBank DUA for ADReSSo/Pitt Corpus (the user must do this personally — see `docs/DATA_ACCESS.md`)
- [ ] Run `scripts/prepare_manifest.py` → `extract_features.py` → `train.py` → `evaluate.py` on real data for all three models
- [ ] Fill in `docs/MODEL_CARD.md` with real speaker-CV + held-out test results per model, compared against the silence control
- [ ] If results hold up: publish checkpoints with a filled-in model card (checking TalkBank DUA redistribution terms first — see `docs/DATA_ACCESS.md` "Redistribution reminder")
- [ ] Optional stretch, once the above baseline is validated: full three-graph fusion (semantic + dependency + PMI), full Pitt Corpus / other DementiaBank subsets for more training data, cross-corpus generalization evaluation (per pitfall #6)
