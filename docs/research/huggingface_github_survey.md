# Survey: Open-Source Models, Datasets & Repos for Alzheimer's Speech Detection

_Research pass conducted 2026-09-12. Covers Hugging Face Hub (models + datasets), GitHub, and alternative speech-based dementia corpora._

## 1. Hugging Face Models — Alzheimer's/Dementia-Speech-Specific

**Reality check up front: there is no credible, well-documented, production-quality pretrained Alzheimer's-speech model on the Hub.** What exists is ~30-40 scattered student/hobbyist uploads, most abandoned mid-project.

| Model | Author | Base | Task | Performance | License | Updated | Usability |
|---|---|---|---|---|---|---|---|
| [flax-community/hubert-dementia-screening](https://huggingface.co/flax-community/hubert-dementia-screening) | HF Flax community event | HuBERT | Screening (classification) | Not reported — no model card | Not specified | Jul 2021 | Weights present, zero documentation |
| [patrickvonplaten/hubert-dementia-screening](https://huggingface.co/patrickvonplaten/hubert-dementia-screening) | Patrick von Platen (mirror) | HuBERT | Screening | None documented | None | Jul 2021 | Same — no card |
| [shields/wav2vec2-base-dementiabank](https://huggingface.co/shields/wav2vec2-base-dementiabank) (+ `-15sec`/`-20sec`/`-40sec`/`-timit-and-dementiabank`) | shields | wav2vec2-base | **ASR** (not classification) | WER = 1.0 (non-functional) | Apache 2.0 | Feb 2022 | Uploaded but doesn't transcribe usefully |
| [giyong/wav2vec2-base_ADReSSo](https://huggingface.co/giyong/wav2vec2-base_ADReSSo), `giyong/distil-large-v2_ADReSSo`, `giyong/whisper-large-v3_ADReSSo` | giyong | wav2vec2 / distil-Whisper / Whisper-large-v3 | Audio classification (AD vs. control) | Not reported | None specified | Undated | No eval numbers, unknown training data |
| [shreyasgite/wav2vec2-large-xls-r-300m-dementianet](https://huggingface.co/shreyasgite/wav2vec2-large-xls-r-300m-dementianet) (+`-8sec`) | shreyasgite (DementiaNet author) | wav2vec2-XLS-R-300M | Speech classification | **40.6% accuracy** (near-chance for binary) | Apache 2.0 | Jan 2022 | Not usable as-is |
| [vjsyong/xlm-roberta-dementia_detection](https://huggingface.co/vjsyong/xlm-roberta-dementia_detection) (+`xlm-v`) | vjsyong | XLM-RoBERTa | Text classification | 89.6% train acc (val loss suggests overfitting) | MIT | Apr 2023 | Minimal documentation |
| [mmtg/dementia_ASR_CTC_Finetuned](https://huggingface.co/mmtg/dementia_ASR_CTC_Finetuned) | mmtg | CTC ASR | ASR | No card | Unspecified | Aug 2024 | No documentation |
| [sharad-raj-aiml-emasters/wav2vec2-*-dementia](https://huggingface.co/sharad-raj-aiml-emasters) | Student project | wav2vec2-XLS-R-300M | Classification | Not documented | Unspecified | Oct 2024 | Coursework artifact |
| [annafavaro/bert-base-uncased-finetuned-addresso](https://huggingface.co/annafavaro/bert-base-uncased-finetuned-addresso) | Anna Favaro (published researcher) | BERT-base-uncased | Text classification | Not reported | Apache 2.0 | ~late 2021 | Minimal card |
| [mocherson/AD-BERT](https://huggingface.co/mocherson/AD-BERT) | AD-BERT paper | Bio+ClinicalBERT | **EHR text** MCI→AD (NOT speech) | Not on card | Unspecified | Nov 2022 | Irrelevant to speech |
| [fagbavor/whisper-*-ADReSSo-2021*](https://huggingface.co/fagbavor) (~15 variants) | fagbavor | Whisper base/small/medium/large | ASR fine-tune (transcription only) | Not reported | Unspecified | Undated | Minimal downloads, no eval metrics |
| `AppliedML123/whisper-large-v3_ADReSSo` | AppliedML123 | Whisper-large-v3 | ASR fine-tune | Not documented | Unspecified | Undated | Same pattern |
| [0x7o/roberta-base-ad-detector](https://huggingface.co/0x7o) | 0x7o | RoBERTa | Text classification | Undocumented; paired dataset card is placeholder text | Apache 2.0 (gated) | Apr 2023 | Not trustworthy |

**NTHU-ML-2023-team19** claims 84.5% accuracy (distil-whisper-large-v2 acoustic model) in its GitHub README and references `huggingface.co/NTHU-ML-2023-team19` as the weights host — but that org page currently shows **"None public yet"**. Common pattern: reproducible *code* exists, but trained *weights* are rarely actually published with documentation/metrics.

Note: MRI-based "Alzheimer" models dominate HF search results for "alzheimer" (e.g. `Falah/Alzheimer_MRI`) — irrelevant to speech, just noise in search.

## 2. Hugging Face Datasets

| Dataset | Content | Access | Concern flag |
|---|---|---|---|
| [cheulyop/dementiabank](https://huggingface.co/datasets/cheulyop/dementiabank) | Loader script only (6KB) — requires locally-held, authorized Pitt Corpus files | Requires DementiaBank/TalkBank membership | None — compliant loader pattern |
| [MearaHe/dementiabank](https://huggingface.co/datasets/MearaHe/dementiabank) | Similar small loader/derived-feature repo | Unclear | Verify before use |
| [0x7o/ad_detector](https://huggingface.co/datasets/0x7o/ad_detector) | CSV, 1K-10K rows, gated | Apache 2.0, gated | Placeholder card — can't confirm what "ad" even refers to; low trust |
| `SilpaCS/Alzheimer`, `AzizSouiai/Alzheimer`, `Falah/Alzheimer_MRI`, `radiata-ai/brain-structure` | MRI images | Various | Not speech-relevant |

**No evidence of raw ADReSSo/DementiaBank audio being illicitly re-uploaded to HF or GitHub.** TalkBank's DUA appears to be actively respected. However, **fine-tuned model weights trained on ADReSSo are being freely redistributed** (fagbavor/giyong/shields models) — a legal gray area (does shipping weights fine-tuned on DUA-protected data count as "redistribution"?) worth being cautious about for this project's own release strategy.

## 3. Alternative Speech-Based Dementia/Cognitive-Impairment Datasets

| Corpus | Access | Notes |
|---|---|---|
| **DementiaBank Pitt Corpus (full, English)** | Free registration via [TalkBank](https://dementia.talkbank.org/access/English/Pitt.html), DUA + "bona fide research" restriction | Source of ADReSS/ADReSSo; full corpus (319 subjects) is larger than challenge subsets — **primary recommended data source** |
| DementiaBank other English subsets (Lu, Delaware, WLS, Holland, Kempler) | Same TalkBank DUA | Expand training volume once Pitt access is granted |
| DementiaBank non-English subsets (Mandarin, Greek Dem@Care, Spanish) | Same TalkBank DUA | Useful for multilingual generalization testing |
| **PROCESS Grand Challenge (ICASSP 2025)** | 157 patients (HC/MCI/Dementia), registered participants | Recent successor challenge to ADReSS/ADReSSo |
| **TAUKADIAL (Interspeech 2024)** | Email registration, DementiaBank membership for full access | 507 samples, English+Chinese, MCI vs NC + MMSE regression |
| **NCMMSC2021 AD Recognition Challenge** | Challenge registration | Mandarin, Cookie Theft, 124 speakers |
| **Carolinas Conversation Collection (CCC)** | Restricted DUA (MUSC) | 646 interviews, conversational (not picture-description) — good domain diversity |
| **Dem@Care** | Signed Terms of Use w/ GAADRD (Greece) | 24 AD + 8 control, Greek, multimodal |
| **DementiaNet** ([repo](https://github.com/shreyasgite/dementianet)) | Public figures scraped from YouTube w/ confirmed diagnoses vs healthy 80+ | Genuinely DUA-free but provenance/label-quality caveats |
| **MultiConAD** ([repo](https://github.com/ArezoShakeri/MultiConAD), [paper](https://arxiv.org/abs/2502.19208)) | Unifies 16 existing datasets; individual sources still gated | Useful as a preprocessing/unification pipeline |
| Kaggle "Federated Alzheimer's Speech Dataset" | Kaggle | Unverified provenance/quality — treat cautiously |
| SpeechDx (ADDF) | Approved-researcher gated access | Industry-sponsored, potentially large |

**No fully DUA-free large audio corpus exists for this task.** Pitt Corpus DUA (free, straightforward) is the most practical real path. DementiaNet is the most interesting DUA-free exception but smaller with label-quality caveats.

## 4. GitHub Repositories

| Repo | Stars | Implements | Weights | Verdict |
|---|---|---|---|---|
| [NTHU-ML-2023-team19/ADReSSo](https://github.com/NTHU-ML-2023-team19/ADReSSo) | 10 | Fine-tuning Whisper/distil-Whisper/wav2vec2/HuBERT/WavLM + RoBERTa/BART; **84.51% accuracy** with distil-whisper-large-v2 | Claimed but not actually public | Best *code* found, weights unavailable |
| [billzyx/WavBERT](https://github.com/billzyx/WavBERT) | 24 | wav2vec2 + BERT fusion; **83.1% acc / 4.44 RMSE / 70.91% F1** on ADReSSo progression task (Interspeech 2021 paper) | Uses public fairseq wav2vec2 checkpoint, not AD-specific | Most complete research-grade repo |
| [billzyx/awesome-dementia-detection](https://github.com/billzyx/awesome-dementia-detection) | 31 | Curated paper list | — | Best literature-survey starting point |
| [wazeerzulfikar/alzheimers-dementia](https://github.com/wazeerzulfikar/alzheimers-dementia) | 66 | Ensemble disfluency/acoustic/intervention nets + openSMILE ComParE; **83.3% acc / 4.60 RMSE** ADReSS, 88% full DementiaBank | Not included | Highest-starred, well-documented, no weights |
| [pcuenca/alzheimer](https://github.com/pcuenca/alzheimer) | 37 | SVM/NB-SVM/RF/ULMFiT/BERT text + xResNet-34 spectrogram CNN + iOS demo | Not documented | 2019 hackathon, unmaintained |
| [KarolChlasta/ADReSS-Challenge2020](https://github.com/KarolChlasta/ADReSS-Challenge2020) | 18 | Notebooks, transfer learning on ADReSS 2020 | Unconfirmed | Reference notebooks |
| [Butovens/DementiaVoiceAnalyzer](https://github.com/Butovens/DementiaVoiceAnalyzer) | 0 | openSMILE + LR/kNN/SVM/FFNN/LSTM; 68% acc | None | Educational only |
| [chirag126/Automatic_Alzheimer_Detection](https://github.com/chirag126/Automatic_Alzheimer_Detection) | 11 | Linguistic features on Pitt transcripts | None | Classical-ML baseline |

No repo above 66 stars — a genuinely small, academic-hobbyist field on GitHub; nothing resembling a maintained production library.

## 5. Reusable Pretrained Building-Block Models (non-AD-specific backbones)

**Audio/speech backbones:**
- **WavLM-large** ([microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)) — strongest general-purpose SSL speech backbone on SUPERB for paralinguistic tasks (denoising + masked-prediction objective preserves speaker/paralinguistic info better than HuBERT/wav2vec2). **Recommended default acoustic backbone.**
- **HuBERT-large / wav2vec2-large-XLSR-53** — strong alternatives, extensively used in actual ADReSSo literature (WavBERT, NTHU repo).
- **TRILLsson** — distilled paralinguistic-specialized Conformer, reported SOTA on several paralinguistic tasks; worth evaluating as smaller/faster alternative.

**ASR for disfluent/elderly speech:**
- **Whisper large-v3 / distil-whisper-large-v2** — outperforms wav2vec2 on WER for elderly/noisy speech generally, and is what most serious ADReSSo practitioners actually fine-tune. distil-whisper performed *best* of acoustic classifiers in NTHU's own experiments (84.5% acc).
- Whisper's CER/WER degrades substantially with dysarthria/impairment severity (mild ~5-8% WER → severe ~20-30%+), a known confound: ASR quality itself correlates with diagnostic severity, so classifiers can partly learn "how bad is the transcript" as a proxy signal rather than genuine linguistic markers.

**Text/transcript embeddings:**
- **Sentence-BERT (SBERT)** — used successfully for Cookie Theft transcript semantic features + classical classifiers.
- **Bio_ClinicalBERT** — more relevant for clinical-note text than pure picture-description transcripts.
- Standard BERT-base/RoBERTa-base remain competitive baselines; 2025 papers increasingly report modern open LLM embeddings (Llama-3, Qwen, Gemma) outperforming classic BERT on this task — worth benchmarking.

## Bottom Line

**There is no usable pretrained open-source Alzheimer's-speech-detection model to build on top of.** Every AD/dementia-specific model on Hugging Face is either an ASR fine-tune with no diagnostic output, a classifier with no/near-chance/unverifiable reported performance, or an abandoned hobbyist upload with a placeholder model card.

**This project will train from scratch on ADReSSo/Pitt Corpus, using generic pretrained backbones as feature extractors.** Recommended path:
1. Obtain Pitt Corpus / ADReSSo access via the standard TalkBank DUA (free, the barrier is time/paperwork, not cost).
2. Use **WavLM-large** or **wav2vec2-XLSR** as the acoustic backbone, **Whisper-large-v3 / distil-whisper-large-v2** for transcription — matches what the few serious open repos (WavBERT, NTHU team) do, and their reported 83-85% accuracy is a reasonable target/sanity check.
3. Use SBERT or a modern open LLM embedding model as the text-branch feature extractor.
4. **Do not use any existing "AD-detection" checkpoint from the Hub as a warm start** — none are trustworthy enough. Documented "Clever Hans" confounds exist in this literature (models exploiting recording-condition artifacts rather than genuine linguistic/acoustic markers), so validate any borrowed component carefully.
