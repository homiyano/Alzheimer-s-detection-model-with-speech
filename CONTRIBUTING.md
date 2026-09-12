# Contributing

## Setup

```bash
python3.10 -m venv .venv   # torch/transformers wheels may not yet support newer Pythons
source .venv/bin/activate
pip install -e .[dev]
```

## Running tests

```bash
pytest tests/ -q
```

Tests never require real ADReSSo/Pitt Corpus data — they run against
`alzspeech.data.synthetic` (fabricated audio + transcripts) and tiny
randomly-initialized transformer configs, so they're fast and safe to run
without a TalkBank DUA. If you add a feature, add a test that exercises it
this way rather than requiring real data.

## Guardrails you must not remove

These exist because the literature review (`docs/research/sota_papers.md`)
documents them as concrete, measured failure modes on this exact dataset —
not theoretical concerns:

1. **No segment-level CV splits.** Only `alzspeech.cv.speaker_kfold` /
   `leave_one_speaker_out` may be used to split data; both assert no speaker
   crosses the train/val boundary. Segment-level splits have been shown to
   inflate accuracy by up to 28 points on this dataset.
2. **Always run the silence-only control** (`alzspeech.models.SilenceControlClassifier`)
   alongside any new model and compare accuracy against it — silence alone
   reaches near-100% accuracy on this corpus in at least one published audit.
3. **Never bundle real ADReSSo/Pitt Corpus audio, transcripts, or other
   protected data** in this repository, in a test fixture, or in an example.
   Use `alzspeech.data.synthetic` instead.

## Code style

- No comments explaining *what* code does; only *why*, when non-obvious.
- Prefer editing existing modules over adding new abstractions — see
  `docs/ARCHITECTURE.md` "Why not X" for scope boundaries already decided.
- Run `ruff check .` before submitting (installed via the `dev` extra).

## Adding a new model or feature approach

If you're implementing something from `docs/research/sota_papers.md` that
isn't one of the three current picks, add it as a new module under
`alzspeech/models/`, follow the existing forward-pass/dataset/collate pattern
in `alzspeech/training/`, and add both a unit test (tiny synthetic model) and
an end-to-end smoke test (`tests/test_end_to_end.py` pattern) before opening a
PR.
