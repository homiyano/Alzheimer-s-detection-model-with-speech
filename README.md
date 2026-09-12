# alzspeech

An open-source pipeline for detecting Alzheimer's disease / cognitive decline
from spontaneous speech, built on the **ADReSSo** dataset (INTERSPEECH 2021
challenge, DementiaBank Pitt Corpus).

**Research finding driving this project**: as of this writing there is no
trustworthy pretrained open-source Alzheimer's-speech model anywhere — not on
Hugging Face, not on GitHub (see [docs/research/huggingface_github_survey.md](docs/research/huggingface_github_survey.md)).
So this repo builds three models from scratch on top of strong generic
pretrained backbones (Whisper, BERT), following the current published SOTA
approaches surveyed in [docs/research/sota_papers.md](docs/research/sota_papers.md).

> **Not a diagnostic tool.** This is a research pipeline. See
> [docs/MODEL_CARD.md](docs/MODEL_CARD.md) "Ethical considerations" before
> using any trained output for anything beyond research.

## Why three models

| Pick | Model | Template | Reported ceiling in the literature |
|---|---|---|---|
| 2 | `TextDisfluencyClassifier` | Yuan et al. 2021 — BERT + pause tokens | 89.6% acc (ADReSS) |
| 1 | `GatedFusionClassifier` | "Listening Between the Lines" 2026 — frozen Whisper + text, gated fusion | F1 90.14% (ADReSSo) |
| 3 | `PMIGraphClassifier` | Gated Multi-Graph Fusion 2026, PMI graph component only | 90.00% acc (ADReSSo) |

Full rationale, including why large SSL encoders are **not** fine-tuned
end-to-end (166 training speakers isn't enough data to do that safely), is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Guardrails (not optional)

This dataset has well-documented failure modes that inflate reported accuracy
if you're not careful — see [docs/research/sota_papers.md](docs/research/sota_papers.md#5-known-pitfalls--methodological-issues):

- **Speaker-level CV only.** `alzspeech.cv` makes segment-level leakage
  structurally impossible (up to 28pp accuracy inflation documented in the
  literature when leakage isn't controlled for).
- **Silence-only control, always.** `scripts/train.py` trains and reports
  `SilenceControlClassifier` — a model with access to *only* pause timing, no
  lexical or acoustic content — on identical folds, because silence alone has
  been shown to reach near-100% accuracy on this corpus due to recording
  artifacts (the "Clever Hans" finding).
- **No protected data in this repo, ever.** `data/` is gitignored; tests run
  against `alzspeech.data.synthetic`, a fabricated dataset with no diagnostic
  signal. Real ADReSSo/Pitt Corpus data requires your own TalkBank DUA — see
  [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

## Setup

```bash
python3.10 -m venv .venv   # torch/transformers wheels may lag behind the newest Python
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart (no real data needed)

Exercise the full pipeline against fabricated data to confirm everything
works before you have TalkBank access:

```python
from alzspeech.data.synthetic import generate_synthetic_dataset
generate_synthetic_dataset("data/synthetic_demo", n_speakers_per_class=8)
```

Then run `pytest tests/ -q` — `tests/test_end_to_end.py` does exactly this
(synthetic data → features → train → evaluate) for all three models plus the
silence control.

## Real workflow (requires TalkBank DUA — see [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md))

```bash
# 1. Build a manifest from your ADReSSo download (paths + labels only, no audio copied)
python scripts/prepare_manifest.py --root /path/to/ADReSSo21 --out data/raw/manifest.csv

# 2. Run ASR + feature extraction once, cached to disk
python scripts/extract_features.py --manifest data/raw/manifest.csv --out-dir data/processed

# 3. Speaker-level CV training (always reports the silence control alongside)
python scripts/train.py --config configs/text_baseline.yaml \
    --features-dir data/processed --output-dir outputs/text_baseline

# 4. Held-out test evaluation with bootstrap 95% CI
python scripts/evaluate.py --checkpoint outputs/text_baseline/checkpoint.pt \
    --features-dir data/processed
```

Swap `configs/text_baseline.yaml` for `configs/gated_fusion.yaml` or
`configs/pmi_graph.yaml` to train the other two models. `configs/silence_control.yaml`
is run automatically by `scripts/train.py` unless you pass `--skip-silence-control`
(not recommended).

## Repository layout

```
src/alzspeech/
  cv.py                  speaker-level k-fold / LOSO splitting (leakage-safe by construction)
  metrics.py              accuracy/F1/precision/recall/specificity + bootstrap CI
  data/
    manifest.py            manifest schema + validation
    synthetic.py            DUA-free synthetic data generator (for tests)
  features/
    asr.py                  Whisper transcription wrapper (word timestamps)
    text_features.py         pause-token insertion, disfluency stats
    audio_features.py        frozen Whisper-encoder frame embeddings
    pmi_graph.py             PMI co-occurrence graph construction
  models/
    text_baseline.py         Pick 2: TextDisfluencyClassifier
    gated_fusion.py           Pick 1: GatedFusionClassifier
    pmi_graph.py               Pick 3: PMIGraphClassifier
    silence_control.py          Clever-Hans sanity-check baseline
  training/                 shared train/eval loop, datasets, collate fns

scripts/                   prepare_manifest, extract_features, train, evaluate CLIs
configs/                   one YAML per model
docs/
  research/                 SOTA literature review + HF/GitHub survey (read first)
  ARCHITECTURE.md            design decisions derived from the research
  DATA_ACCESS.md              how to get ADReSSo legally
  MODEL_CARD.md                template to fill in for any released checkpoint
tests/                     unit + end-to-end smoke tests, all DUA-free
```

## Status

See [TASKS.md](TASKS.md) for what's implemented vs. still open. In short:
the full pipeline (data → features → 3 models → CV training → evaluation) is
built and verified end-to-end against synthetic data and real (small)
Hugging Face checkpoints — see the commit history for the CPU smoke-test runs.
**Actually training on real ADReSSo data is the one thing this session could
not do**, since that requires a TalkBank DUA only you can obtain personally.
Once you have data, the four `scripts/*.py` CLIs above are the entire
remaining path to a trained, evaluated model.

## Further reading

- [docs/research/sota_papers.md](docs/research/sota_papers.md) — full SOTA
  literature review, pitfalls, and reimplementation notes for 8+ key papers.
- [docs/research/huggingface_github_survey.md](docs/research/huggingface_github_survey.md)
  — why nothing pretrained exists to build on.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design decisions and explicit
  scope boundaries ("why not X").

## License

Code: MIT (see [LICENSE](LICENSE)). ADReSSo/Pitt Corpus data is **not**
included and is governed by TalkBank's own terms — see
[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).
