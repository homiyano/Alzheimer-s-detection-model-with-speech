# Task List

Derived from [docs/research/](docs/research/) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Phase 0 — Research (done)
- [x] Survey SOTA papers on ADReSS/ADReSSo Alzheimer's speech detection
- [x] Survey Hugging Face models/datasets and GitHub repos for existing open-source solutions
- [x] Write architecture/design-decision doc

## Phase 1 — Project foundation
- [x] Repo scaffolding (.gitignore, directory layout)
- [ ] `pyproject.toml` / `requirements.txt` with pinned core deps
- [ ] `docs/DATA_ACCESS.md` — TalkBank/DementiaBank DUA request process, manifest format spec
- [ ] `alzspeech.cv` — speaker-level k-fold / leave-one-speaker-out splitting utilities + tests
- [ ] `alzspeech.data.manifest` — manifest schema + loader/validator
- [ ] `alzspeech.data.synthetic` — synthetic audio/transcript generator for DUA-free testing

## Phase 2 — Feature extraction
- [ ] `alzspeech.features.asr` — Whisper transcription wrapper with word-level timestamps
- [ ] `alzspeech.features.text_features` — pause/filler token insertion, disfluency & lexical-richness features
- [ ] `alzspeech.features.audio_features` — frozen Whisper-encoder embedding extraction (attention pooling)
- [ ] `scripts/prepare_manifest.py`, `scripts/extract_features.py`

## Phase 3 — Models (three staged picks from the literature review)
- [ ] `alzspeech.models.SilenceControlClassifier` — Clever-Hans sanity-check baseline
- [ ] `alzspeech.models.TextDisfluencyClassifier` — Pick 2: BERT/RoBERTa + pause tokens
- [ ] `alzspeech.models.GatedFusionClassifier` — Pick 1: frozen acoustic embeddings + text, gated fusion
- [ ] `alzspeech.models.PMIGraphClassifier` — Pick 3: PMI co-occurrence graph + GAT augmentation

## Phase 4 — Training & evaluation
- [ ] `alzspeech.training` — generic speaker-CV training loop, checkpointing, config-driven
- [ ] `alzspeech.metrics` — accuracy/F1/precision/recall/specificity + bootstrap CI, MMSE RMSE
- [ ] `scripts/train.py`, `scripts/evaluate.py` CLIs
- [ ] `configs/*.yaml` for each model

## Phase 5 — Tests & verification
- [ ] Unit tests for CV splitting (no speaker leakage), manifest validation, feature extraction shapes
- [ ] End-to-end smoke test: synthetic data -> features -> train (1 epoch) -> evaluate, for all 3 models
- [ ] Silence-control comparison wired into the smoke test

## Phase 6 — Documentation & release hygiene
- [ ] `README.md` — project overview, setup, usage, research summary, results table (to fill in once trained on real data), ethics/limitations section
- [ ] `docs/MODEL_CARD.md` template per Hugging Face model card conventions
- [ ] `LICENSE` check (MIT already present) + explicit note on data licensing (TalkBank DUA, no redistribution)
- [ ] `CONTRIBUTING.md`

## Explicitly out of scope for this session
- Actually training on real ADReSSo data — requires a TalkBank DUA the user must obtain personally (documented in `docs/DATA_ACCESS.md`); the codebase is built and verified end-to-end against synthetic data so it is ready to run the moment real data is available.
- Full three-graph fusion (semantic + dependency + PMI) — only the highest-ablation-value PMI graph is implemented; documented as a future extension.
