# Model Card Template

Copy this file to `outputs/<run_name>/MODEL_CARD.md` and fill it in for every
checkpoint you intend to share, per
[Hugging Face model card conventions](https://huggingface.co/docs/hub/model-cards).
Do not publish a checkpoint without one.

## Model description

- **Model type**: `text` / `fusion` / `graph` (see `docs/ARCHITECTURE.md`)
- **Architecture**: <backbone(s), fusion mechanism, parameter count>
- **Language(s)**: English (ADReSSo is English-only Cookie Theft descriptions)
- **License**: MIT (code); see "Training data" below for data licensing caveats
- **Trained from**: <config file used, e.g. `configs/text_baseline.yaml`>

## Intended use

- **Primary intended use**: research on speech-based Alzheimer's/cognitive-decline
  detection; NOT a diagnostic tool.
- **Out-of-scope use**: clinical diagnosis, screening decisions affecting real
  patients, or any use without a qualified clinician in the loop. See
  "Ethical considerations" below.

## Training data

- **Dataset**: ADReSSo (DementiaBank Pitt Corpus subset), obtained under a
  TalkBank DUA — see `docs/DATA_ACCESS.md`. Not redistributed with this model.
- **N speakers / class balance**: <fill in from your manifest>
- **Preprocessing**: <ASR model used, pause threshold, any filtering>

## Evaluation

Report the **speaker-level CV** results from `scripts/train.py`'s
`cv_results.json`, not just a single train/test split — and always alongside
the silence-only control from the same run:

| Split | Model accuracy | Model F1 | Silence-control accuracy |
|---|---|---|---|
| CV (mean ± std) | | | |
| Held-out test (`scripts/evaluate.py`, with 95% bootstrap CI) | | | |

If the model's accuracy does not clearly exceed the silence-only control's,
say so explicitly rather than omitting the comparison — this is not optional
per `docs/ARCHITECTURE.md`'s guardrails.

**Cross-corpus generalization** (recommended, not run by default): report
accuracy on a second corpus (e.g. DementiaBank Lu/Delaware) if you have access
— see `docs/research/sota_papers.md` pitfall #6 on the documented ADReSSo-only
generalization gap.

## Limitations

- Trained on ~166 speakers — a headline accuracy difference of a few points
  vs. another paper's reported number is likely within noise (see pitfall #8).
- ADReSSo is English, Cookie-Theft-task-specific, and demographically curated;
  performance on other languages, tasks, or populations is untested unless you
  specifically evaluated it.
- ASR transcript quality affects the text/fusion/graph models' input; accuracy
  may vary with ASR system choice (pitfall #5).
- Not validated for clinical use. Do not present model output as a diagnosis.

## Ethical considerations

Alzheimer's/dementia status is highly sensitive health information. A false
positive can cause significant distress; a false negative can delay care. This
model:

- Should only be used in research contexts with informed consent, or by
  qualified clinicians as one input among many — never as a standalone
  diagnostic signal.
- Has not been evaluated for demographic fairness (e.g. across accents,
  dialects, non-native English speakers, or recording conditions) beyond
  ADReSSo's built-in age/gender balancing.
- Reflects the biases and limitations of its training data — a small,
  curated, English-only, single-task corpus.

## Citation

If you use ADReSSo, cite per TalkBank's rules
([talkbank.org/0share/rules.html](https://talkbank.org/0share/rules.html)) and
the original challenge paper (see `docs/research/sota_papers.md` section 1.1).
