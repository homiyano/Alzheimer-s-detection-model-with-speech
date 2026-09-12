# Architecture & Design Decisions

Based on [docs/research/sota_papers.md](research/sota_papers.md) and [docs/research/huggingface_github_survey.md](research/huggingface_github_survey.md).

## Summary of research findings

- **No usable pretrained open-source Alzheimer's-speech model exists** on Hugging Face or GitHub — every AD-specific checkpoint found is undocumented, near-chance, or abandoned. This project trains from scratch on ADReSSo, using strong generic pretrained backbones (Whisper, BERT/RoBERTa) as feature extractors.
- Current published SOTA on ADReSSo classification is ~87-90% accuracy/F1 (Gated Multi-Graph Fusion via GAT, "Listening Between the Lines", CogniAlign), reached via **frozen or lightly-tuned SSL features + fusion of acoustic and linguistic modalities**, not by fine-tuning large encoders end-to-end on the ~166-speaker training set.
- The dataset is tiny (166 train speakers) and the field has **well-documented pitfalls**: speaker/segment leakage inflating accuracy by up to 28pp, a "Clever Hans" silence-only artifact reaching near-100% accuracy, and weak cross-corpus generalization. Any credible open-source release must actively guard against these, not just report a headline number.

## Chosen approach: staged build, three models sharing one pipeline

We implement the three "top picks" from the literature review as increasingly sophisticated models sharing the same data/feature pipeline, so each stage is independently useful and testable:

1. **Text baseline** (`alzspeech.models.TextDisfluencyClassifier`) — BERT/RoBERTa fine-tuned on ASR transcripts with pause/filler tokens inserted at detected silence positions (Yuan et al. 2021 template). Cheapest, easiest to debug, ~88-90% accuracy ceiling in the literature. This is also required as an ablation baseline for the fusion model.
2. **Gated multimodal fusion** (`alzspeech.models.GatedFusionClassifier`) — frozen Whisper-encoder acoustic embeddings (attention-pooled) fused with the text branch via a learned gate ("Listening Between the Lines" template). No SSL encoder fine-tuning — 166 speakers is not enough data to safely update hundreds of millions of encoder parameters (explicit anti-pattern flagged in the research).
3. **PMI co-occurrence graph augmentation** (`alzspeech.models.PMIGraphClassifier`) — optional second iteration adding a PMI-weighted co-occurrence graph (vs. a healthy-speaker reference corpus) through a GAT layer, concatenated with the BERT `[CLS]` vector (Xiao et al. 2026 template, scoped to the single highest-ablation-value graph).

All three are binary AD-vs-control classifiers first (matching the ADReSSo primary task); MMSE regression is a documented extension point (swap the classification head for a regression head over the same pooled features).

## Guardrails baked into the pipeline (not optional add-ons)

- **Speaker-level cross-validation only.** `alzspeech.cv.speaker_kfold` / `leave_one_speaker_out` take a speaker-id column and guarantee no speaker's segments cross the train/val boundary. There is no segment-level split path in the codebase — this is deliberate, given the documented 28pp accuracy inflation from leakage.
- **Silence-only control classifier.** `alzspeech.models.SilenceControlClassifier` trains on nothing but total-silence-duration / speech-rate summary stats. Any real model's reported accuracy must be compared against this control; the training script always reports both. This directly operationalizes the Clever Hans finding.
- **No bundled data.** TalkBank's DUA forbids redistributing Pitt Corpus / ADReSSo audio or transcripts. The repo never contains real recordings — `data/raw/` is gitignored, and `docs/DATA_ACCESS.md` documents the request process. Tests run against a synthetic data generator (`alzspeech.data.synthetic`) that produces structurally-similar but fake sine-wave audio and template transcripts, so the full pipeline is exercised in CI without any protected data.
- **Reporting discipline.** The eval script always reports accuracy, F1, precision, recall, specificity, and a 95% CI via bootstrap (small-N statistical power is a documented pitfall — headline differences under a few points are frequently noise at n=71).

## Pipeline stages

```
raw audio (DUA-gated, user-supplied)
  -> scripts/prepare_manifest.py    (build manifest.csv: speaker_id, path, label, split)
  -> scripts/extract_features.py    (Whisper ASR + word timestamps -> transcript with pause tokens;
                                      frozen Whisper encoder embeddings; cached to data/processed/)
  -> scripts/train.py               (speaker-level CV; text / fusion / graph model; silence control)
  -> scripts/evaluate.py            (metrics + bootstrap CI + silence-control comparison)
```

## Why not X

- **Fine-tuning a large SSL audio encoder end-to-end** (wav2vec2-large, WavLM-large, Whisper-large): explicitly flagged in the research as costing more compute than frozen-feature approaches while not reliably beating them at this data scale. Not implemented as a default path; left as a documented extension for anyone with a larger (e.g. full Pitt Corpus + other DementiaBank subsets) training set.
- **Training on the full three-graph fusion architecture from the start**: the PMI co-occurrence graph alone captured most of the ablation gain in the source paper; building all three graphs first would add engineering surface area (dependency parsing, semantic-similarity graph tuning) for limited marginal benefit before the simpler models are validated.
- **Zero-training LLM few-shot prompting** (MMSE-Calibrated Few-Shot Prompting, 0.82 acc / 0.86 AUC): documented as the best "free lunch" sanity-check baseline, not implemented as a first-class model here since it depends on a proprietary hosted LLM API rather than being a self-contained open-source artifact — left as a documented, optional comparison script (`scripts/baseline_llm_fewshot.py`, requires the user's own API key).
