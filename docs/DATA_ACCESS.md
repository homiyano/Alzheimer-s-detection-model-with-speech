# Getting Data: ADReSSo / DementiaBank Pitt Corpus

This project trains on the **ADReSSo** dataset (Alzheimer's Dementia Recognition
through Spontaneous Speech, INTERSPEECH 2021), drawn from the **Pitt Corpus**
on **DementiaBank** (part of TalkBank). See
[docs/research/sota_papers.md](research/sota_papers.md) section 1 for full
background.

**No audio, transcripts, or other protected data is bundled in this repository,
and none ever will be** — TalkBank's Data Use Agreement (DUA) prohibits
redistribution. You must request access yourself.

## 1. Request access

1. Register/log in at [talkbank.org](https://talkbank.org) and read the
   [Ground Rules](https://talkbank.org/0share/rules.html).
2. Email **talkbank@cmu.edu** with your name, institutional affiliation, and a
   brief statement of intended use (e.g. "open-source research on speech-based
   Alzheimer's detection").
3. If you are a student/trainee, your faculty advisor must request membership
   and separately request dated access for you — see
   [Data Access Levels](https://talkbank.org/0share/access.html).
4. Once approved, the ADReSSo-specific download is available via
   [talkbank.org/dementia/access/](https://talkbank.org/dementia/access/).

There is no fee. TalkBank generally relies on the contributing institution's
original IRB approval; see [IRB Approval](https://talkbank.org/0share/irb/) if
your institution requires more.

## 2. What you'll get

The ADReSSo 2021 release is organized (approximately; layout has varied
slightly across challenge years) as:

```
ADReSSo21/
  diagnosis/
    train/
      audio/
        ad/*.wav
        cn/*.wav
    test-dist/
      audio/*.wav   (unlabeled; official test labels released separately/later)
```

166 training speakers (87 AD / 79 CN), 71 test speakers. No manual transcripts
are provided for ADReSSo (unlike the earlier ADReSS 2020 release) — this
project generates its own ASR transcripts via `scripts/extract_features.py`.

If you also get access to the **full Pitt Corpus** (not just the ADReSSo
challenge subset) or other DementiaBank English subsets (Lu, Delaware, WLS),
you can extend the training set — see
[docs/research/huggingface_github_survey.md](research/huggingface_github_survey.md)
section 3 for a comparison of alternative corpora and their access processes.

## 3. Build a manifest

```bash
python scripts/prepare_manifest.py --root /path/to/ADReSSo21 --out data/raw/manifest.csv
```

Adjust `--ad-subdir`/`--cn-subdir`/`--splits` if your download's directory
names differ. This only writes file *paths* and labels to `data/raw/manifest.csv`
(gitignored) — never audio content — so the manifest itself is safe to move
around, but `data/raw/` must never be committed or otherwise redistributed.

See [alzspeech.data.manifest](../src/alzspeech/data/manifest.py) for the exact
schema if you need to build a manifest by hand for a different corpus.

## 4. Extract features and train

```bash
python scripts/extract_features.py --manifest data/raw/manifest.csv --out-dir data/processed
python scripts/train.py --config configs/text_baseline.yaml --features-dir data/processed --output-dir outputs/text_baseline
python scripts/evaluate.py --checkpoint outputs/text_baseline/checkpoint.pt --features-dir data/processed
```

See the main [README](../README.md) for the full walkthrough and
[docs/ARCHITECTURE.md](ARCHITECTURE.md) for what each model does.

## Redistribution reminder

- Do not commit anything under `data/raw/`, `data/interim/`, or `data/processed/`
  (all gitignored) to this or any other public repository.
- If you publish model weights fine-tuned on ADReSSo/Pitt Corpus data, check
  your own TalkBank DUA's specific terms before redistributing them — the
  research survey ([docs/research/huggingface_github_survey.md](research/huggingface_github_survey.md)
  section 2) notes this is a legal gray area that several existing Hugging
  Face uploads have not clearly resolved.
