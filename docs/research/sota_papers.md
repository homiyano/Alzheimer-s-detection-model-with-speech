# Literature Review: SOTA for Alzheimer's Detection from Speech (ADReSS/ADReSSo)

_Research pass conducted 2026-09-12._

## 1. The ADReSS / ADReSSo Challenges

### 1.1 Origin and lineage

- **ADReSS** — INTERSPEECH 2020 shared task (Luz, Haider, de la Fuente, Fromm, MacWhinney). [arXiv:2004.06833](https://arxiv.org/abs/2004.06833). 34 participating teams.
- **ADReSSo** (ADReSS-only-speech) — INTERSPEECH 2021 follow-up, same organizers. [ISCA](https://www.isca-archive.org/interspeech_2021/luz21_interspeech.html). Key change: **no manual (CLAN-annotated) transcripts provided** — forces genuinely speech-only pipelines.
- Related: **ADReSS-M** (ICASSP 2023 SPGC, cross-lingual English→Greek — [PMC11218814](https://pmc.ncbi.nlm.nih.gov/articles/PMC11218814/)) and **TAUKADIAL** (multilingual MCI detection, Interspeech 2024).

### 1.2 Data source

Both challenges are built from the **Pitt Corpus** (Cookie Theft picture-description task, Boston Diagnostic Aphasia Examination), part of **DementiaBank**, hosted on **TalkBank**. Full Pitt Corpus (244 control + 309 dementia/MCI/vascular-dementia subjects) is larger than the curated ADReSS/ADReSSo subsets.

### 1.3 Dataset composition & splits

| Challenge | Total speakers | Train | Test | Transcripts |
|---|---|---|---|---|
| ADReSS (2020) | 156 | 108 (54 AD / 54 HC) | 48 (24 AD / 24 HC) | Manual (CHAT/CLAN) + audio, age/gender-matched |
| ADReSSo (2021) | 237 | 166 (87 AD / 79 HC) | 71 (35 AD / 36 HC) | **None provided** — ASR must be self-generated |
| ADReSS-M (2023) | 271 | 225 English | 46 Greek | Cross-lingual generalization test |

ADReSS mean MMSE ≈ 17.0 (AD) vs 29.1 (non-AD) in training — correlates almost linearly with linguistic decline (part of why linguistic approaches do well, and a confound — see §5).

### 1.4 Task definitions

1. **AD classification** — binary AD vs. HC (accuracy/F1/precision/recall/specificity).
2. **MMSE regression** — predict continuous MMSE score (0–30), scored by RMSE.
3. **Cognitive decline/progression prediction** (ADReSSo Task 3) — decline trajectory class.

### 1.5 Baselines

- ADReSSo baseline: **78.87% accuracy** (classification), **RMSE 5.28** (MMSE), **68.75% acc / F1 66.67%** (progression). [Luz et al. 2021](https://www.isca-archive.org/interspeech_2021/luz21_interspeech.html)
- ADReSS-M baseline: 73.91% acc, RMSE 4.955; best team 87.0% acc / RMSE 3.727. [PMC11218814](https://pmc.ncbi.nlm.nih.gov/articles/PMC11218814/)

### 1.6 How to legally obtain the data (TalkBank/DementiaBank DUA)

TalkBank tiered access ([Data Access Levels](https://talkbank.org/0share/access.html)):

1. Open access — docs/tools, no login.
2. Registration required — most general corpora.
3. **Approved access** (DementiaBank/AphasiaBank/ADReSSo tier):
   - Register/login at talkbank.org, read the **Ground Rules**.
   - Email **talkbank@cmu.edu** with name, institutional affiliation, intended use.
   - Students: faculty advisor must request membership and separately request dated access for the student.
   - IRB: TalkBank generally relies on the contributing institution's original IRB; clinical sub-banks may request more documentation ([IRB Approval](https://talkbank.org/0share/irb/)).
4. Controlled access — stricter tier (not typically required for DementiaBank).

No fees. **Redistribution is prohibited** — raw audio/transcripts cannot be bundled into this (or any) open-source repo. Code, model weights (subject to DUA terms), and derived numerical features can be released. Cite the corpus per [Basic Rules for Data Usage](https://talkbank.org/0share/rules.html). Request page: [talkbank.org/dementia/access/](https://talkbank.org/dementia/access/).

## 2. State-of-the-Art Approaches

### 2.1 Classical acoustic features (eGeMAPS/openSMILE/ComParE)

- eGeMAPS (88-dim) / ComParE 2016 (6,373-dim functionals) via openSMILE remain standard hand-crafted baselines.
- Pure acoustic ComParE+1NN on ADReSS: ~57.4% accuracy; LDA/DT variants ~62.5% — acoustics alone are weak.
- Edwards et al. 2020: OpenSMILE eGeMAPS/ComParE + phonemic/prosodic features → **F1 = 0.78** on ADReSS.
- Acoustic features generalize better cross-lingually (ADReSS-M) than linguistic ones — disfluency/pause/rhythm features were most portable English→Greek.

### 2.2 Linguistic/NLP approaches on transcripts

- **Balagopalan et al., "To BERT or Not to BERT" (Interspeech 2020)** — [arXiv:2008.01551](https://arxiv.org/abs/2008.01551). Fine-tuned BERT beats hand-engineered feature pipelines on manual ADReSS transcripts — established "linguistics dominates."
- **Yuan et al., "Combining Linguistic Complexity and (Dis)Fluency Features with PLMs" (Interspeech 2021)** — [arXiv:2106.08689](https://arxiv.org/pdf/2106.08689). Encodes filled/unfilled pauses as special tokens in transcript text, fine-tunes BERT/ERNIE. **89.6% accuracy with ERNIE on ADReSS test** — one of the highest manual-transcript results on record.
- RoBERTa fine-tuning on ADReSSo ASR transcripts: **88.7% accuracy** in several comparative studies.
- Lexical richness/syntactic complexity/POS features (TTR, Yngve depth, idea density) remain used as interpretable auxiliary features in fusion systems.

### 2.3 Self-supervised speech representations (wav2vec2/HuBERT/WavLM/Whisper/data2vec)

- **WavBERT (Zhu et al., Interspeech 2021)** — [PMC10102979](https://pmc.ncbi.nlm.nih.gov/articles/PMC10102979/). wav2vec2 for ASR + preserves non-semantic cues (pause tokens + embedding-conversion network feeding acoustic info into BERT's input space); fine-tunes BERT. **83.1% acc, RMSE 4.44, 70.91% F1** on ADReSSo — strongest fully speech-only result at the time.
- **Qin et al., "Exploiting Pre-Trained ASR Models..." (2021)** — [arXiv:2110.01493](https://arxiv.org/pdf/2110.01493). Truncated pretrained ASR acoustic encoder (bottom layers) + FC head, end-to-end on raw audio. +4.6pp over transcript-based RoBERTa on a related Chinese AD challenge — SSL-encoder bottom layers carry strong paralinguistic AD signal.
- **data2vec end-to-end (2022)** — [PMC9856143](https://pmc.ncbi.nlm.nih.gov/articles/PMC9856143/): AUC **0.846** held-out ADReSSo, 0.835 external DementiaBank test.
- **HAFFormer (Dong et al., 2024)** — [arXiv:2405.03952](https://arxiv.org/pdf/2405.03952). Attention-free hierarchical transformer (depthwise-conv token mixer + GEGLU) over frozen wav2vec2 XLS-R embeddings, ~1/20th MACs of standard Transformer. **82.60% acc/F1 on ADReSS-M** vs 73.91% fine-tuned wav2vec2 baseline.
- **Whisper transfer learning w/ transcript prompts (IEEE, 2024)**: **84.51% acc / 84.50% F1** on ADReSSo, 9-12% relative improvement over plain fine-tuning.

### 2.4 Multimodal fusion (acoustic + linguistic [+ vision])

- **Rohanian et al.-style BiLSTM+highway fusion** — [arXiv:2106.15684](https://arxiv.org/pdf/2106.15684). Words + word probabilities + disfluency markers + pause info + acoustic features. **84% accuracy** on ADReSSo, robust to noisy ASR.
- **"Listening Between the Lines" (2026)** — [arXiv:2606.30675](https://arxiv.org/abs/2606.30675). Whisper acoustic embeddings (temporal attention pooling) + LLM-extracted interpretable linguistic features, gated fusion. **F1 89.47% (ADReSS) / 90.14% (ADReSSo)**. Acoustic-only F1 83.08% vs linguistic-only 76.06% on ADReSSo — fusion gains +7 to +14pp.
- **Query-based multimodal + adaptive gated fusion (Frontiers, 2026)**: **87.14% acc, macro-F1 87.05%** on ADReSSo.
- **CogniAlign (2025)** — [arXiv:2506.01890](https://arxiv.org/abs/2506.01890). **87.35% (LOSO) / 90.36% (5-fold CV)** — see §3.

### 2.5 Graph neural network approaches

- **Disfluency co-occurrence graphs + GCN (MCPR 2026)** — token-level co-occurrence graphs with special disfluency nodes, BERT embeddings, GCN classifier; outperforms fine-tuned BERT alone; Integrated-Gradients confirms attention to disfluency nodes. [Springer](https://link.springer.com/chapter/10.1007/978-3-032-28393-1_31)
- **Gated Multi-Graph Fusion via GAT (Xiao et al., 2026)** — [arXiv:2606.31186](https://arxiv.org/html/2606.31186v1). Three graphs (semantic/BERT-cosine, dependency/syntactic, co-occurrence/PMI vs healthy reference corpus) each through GAT + mean pool, combined via per-sample softmax gate. **90.00% acc, 89.86% F1 on ADReSSo** — among the top reported numbers. PMI co-occurrence graph is the single most important component (ablation).

### 2.6 LLM-based / prompting approaches (2023-2026)

- ChatGPT vs Bard zero-shot/CoT (2024) — [PMC11048951](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048951/): Bard F1 71%, GPT-4 F1 62% — beat chance but short of clinical grade untrained.
- Prompt-tuned PLMs (2022) — [arXiv:2210.16539](https://arxiv.org/abs/2210.16539): 84.20% mean acc (manual transcripts) / 82.64% (ASR transcripts) on ADReSS.
- LLaMA2+LoRA prompt fine-tuning (2025) — [arXiv:2501.00861](https://arxiv.org/abs/2501.00861): **81.31% accuracy**, +4.46pp over BERT control.
- GPT-4 as linguistic-feature extractor (2024) — [arXiv:2412.15772](https://arxiv.org/pdf/2412.15772) — LLM-as-feature-extractor pattern.
- CoT reasoning approaches (Interspeech 2025) — [arXiv:2506.01683](https://arxiv.org/pdf/2506.01683).
- **MMSE-Calibrated Few-Shot Prompting (2025)** — [arXiv:2509.19926](https://arxiv.org/abs/2509.19926). Training-free ICL anchored to MMSE-band probabilities. **0.82 acc / 0.86 AUC on ADReSS, zero gradient training** — good near-zero-compute baseline.
- Paired-LLM perplexity difference (2025) — [arXiv:2506.09315](https://arxiv.org/pdf/2506.09315).
- **LLMCare (2025)** — 10 transformer architectures + LLM-generated synthetic augmentation + 110 handcrafted features. **F1 85.7% in-domain (ADReSSo) → 72.8% external (DementiaBank Delaware MCI)** — concrete generalization-gap data point. [github.com/SpeechCARE/LLMCare](https://github.com/SpeechCARE/LLMCare).

### 2.7 Chronological highlight reel (2023-2026)

| Year | Paper | Approach | Best result |
|---|---|---|---|
| 2023 | ADReSS-M SPGC overview | Cross-lingual acoustic+disfluency ensemble | 87.0% acc / RMSE 3.727 |
| 2024 | HAFFormer | Attention-free transformer over XLS-R | 82.6% acc/F1 (ADReSS-M) |
| 2024 | Clever Hans effect paper | Bias audit (silence-only classifier) | ~100% acc from silence alone — bias finding |
| 2024 | Whisper transfer learning | Whisper fine-tune w/ transcript prompts | 84.51% acc / 84.50% F1 (ADReSSo) |
| 2025 | CogniAlign | Word-level audio-text gated cross-attention | 87.35% LOSO / 90.36% 5-fold |
| 2025 | MMSE-Calibrated Few-Shot Prompting | Training-free LLM ICL | 0.82 acc / 0.86 AUC |
| 2026 | Gated Multi-Graph Fusion (GAT) | Tri-graph fusion | **90.00% acc / 89.86% F1 (ADReSSo)** |
| 2026 | Listening Between the Lines | Whisper + LLM-linguistic gated fusion | F1 89.47% (ADReSS) / 90.14% (ADReSSo) |
| 2026 | Cleaner Speech, Weaker Generalization | Methodology critique | N/A — critique, see §5 |

## 3. Reimplementation Detail — Top Papers

1. **Yuan et al. 2021** ([arXiv:2106.08689](https://arxiv.org/pdf/2106.08689)) — 89.6% acc, ADReSS. Insert `[PAUSE]`/filler tokens at detected silence positions into transcripts; fine-tune BERT/ERNIE as sequence classification; optionally concat hand-crafted linguistic-complexity features before the final layer. Cheapest, most reimplementable "linguistics only" SOTA.

2. **Balagopalan et al. 2020** ([arXiv:2008.01551](https://arxiv.org/abs/2008.01551)) — foundational BERT-vs-features comparison; simple `BertForSequenceClassification` fine-tune. Good baseline/ablation reference.

3. **WavBERT (Zhu et al. 2021)** ([PMC10102979](https://pmc.ncbi.nlm.nih.gov/articles/PMC10102979/)) — 83.1% acc / RMSE 4.44 / 70.91% F1, full ADReSSo suite, speech-only. wav2vec2→ASR+frame features; convert CTC "blank" outputs into pause tokens inserted into transcript; small embedding-conversion network maps wav2vec2 embeddings into BERT's embedding space; fine-tune BERT, freeze conversion net. Batch 8, LR 1e-6, Adam, 10 runs averaged. Best template for a genuinely speech-only (ADReSSo-compliant) system.

4. **Gated Multi-Graph Fusion (Xiao et al. 2026)** ([arXiv:2606.31186](https://arxiv.org/html/2606.31186v1)) — 90.00% acc / 89.86% F1, ADReSSo. Whisper→ASR→BERT-base embeddings; build 3 graphs (semantic cosine-sim τ=0.8, syntactic dependency parse, PMI co-occurrence vs healthy reference corpus window n=3 τ=0.3); each through GAT→mean pool; gated fusion (softmax weight network) + concat → classifier. Adam, 100 epochs, batch 8, label smoothing α=0.2. Removing PMI co-occurrence graph hurts most.

5. **CogniAlign (2025)** ([arXiv:2506.01890](https://arxiv.org/abs/2506.01890)) — 87.35% LOSO / 90.36% 5-fold, ADReSSo. Word-level forced-alignment of audio segments to transcript tokens; **asymmetric gated cross-attention** (audio attends over text, since text alone outperforms audio alone); pause tokens in text matched with silent-interval audio embeddings; Integrated Gradients for interpretability.

6. **"Listening Between the Lines" (2026)** ([arXiv:2606.30675](https://arxiv.org/abs/2606.30675)) — F1 90.14%, ADReSSo. Whisper encoder + attention pooling (not naive mean-pool) for acoustics; LLM (prompted, not fine-tuned) extracts interpretable linguistic features (lexical diversity, syntactic complexity, coherence, discourse); gated fusion. Reports per-modality numbers (acoustic F1 83.08%, linguistic F1 76.06%) — useful for incremental validation.

7. **HAFFormer (Dong et al. 2024)** ([arXiv:2405.03952](https://arxiv.org/pdf/2405.03952)) — 82.60% acc/F1, ADReSS-M. Frozen wav2vec2 XLS-R embeddings (3200×1024) → linear projection to 3200×8 (aggressive dim reduction, key tractability trick); 3-stage hierarchical attention-free encoder (depthwise multi-scale conv token mixer + GEGLU channel mixer). AdamW, LR 2e-3, wd 1e-5, batch 8, 80 epochs. ~20x cheaper in MACs than standard Transformer.

8. **MMSE-Calibrated Few-Shot Prompting (Sweidan et al. 2025)** ([arXiv:2509.19926](https://arxiv.org/abs/2509.19926)) — 0.82 acc / 0.86 AUC, ADReSS, zero gradient training. Class-balanced few-shot ICL prompt where each example carries an MMSE-band-derived probability anchor; a variant generates synthetic examples via GPT-5 reasoning over image+transcript+MMSE. Best "free lunch" baseline to sanity-check the data pipeline before training anything.

## 4. Open-Source Code Repositories

| Repo | Implements | Stars | Last push | Notes |
|---|---|---|---|---|
| [wazeerzulfikar/alzheimers-dementia](https://github.com/wazeerzulfikar/alzheimers-dementia) | Multimodal ensemble (acoustic+cognitive/linguistic) for ADReSS classification+MMSE | 66★ | 2022-12 | Most-starred ADReSS-specific repo |
| [billzyx/awesome-dementia-detection](https://github.com/billzyx/awesome-dementia-detection) | Curated paper list | 46★ | 2026-03 (active) | Best resource to track new work |
| [shreyasgite/dementianet](https://github.com/shreyasgite/dementianet) | Longitudinal speech ML pipeline | 38★ | 2022-08 | |
| [vmasrani/dementia_classifier](https://github.com/vmasrani/dementia_classifier) | Classic ML pipeline | 18★ | 2023-07 | Older/classical style |
| [KarolChlasta/ADReSS-Challenge2020](https://github.com/KarolChlasta/ADReSS-Challenge2020) | ADReSS 2020 challenge code | 18★ | 2020-12 (stale) | Historical reference |
| [NiliRahmani/...ADReSSo](https://github.com/NiliRahmani/Alzheimer-s-Dementia-Recognition-through-Spontaneous-Speech) | ADReSSo feature extraction (20s clips) | 15★ | 2023-08 | Readable audio pipeline |
| [probstlukas/gpt3-dementia-detection](https://github.com/probstlukas/gpt3-dementia-detection) | GPT-3 embeddings→classifier | 5★ | 2026-05 | Modern LLM-embedding approach |
| [SpeechCARE/LLMCare](https://github.com/SpeechCARE/LLMCare) | 10 architectures + LLM synthetic augmentation + 110 handcrafted features | 2★ | 2025-12 | Newest/most complete, well-documented |

None redistribute raw ADReSS/ADReSSo audio/transcripts (DUA-restricted) — a DUA-approved copy is required to run any end-to-end.

## 5. Known Pitfalls / Methodological Issues

1. **Tiny dataset, high overfitting risk.** ADReSS: 156 subjects; ADReSSo: 237, 166 train. Any model with more than a few thousand effective parameters overfits trivially — strong papers restrict to frozen SSL features + shallow heads, heavy regularization, LOSO/k-fold CV.

2. **Speaker leakage inflates accuracy dramatically.** Studies with confirmed subject-wise splits: 66-90% accuracy; high-leakage-risk studies (segment-level splits): 95-99% claimed. One study: accuracy dropped **94%→66%** once leakage was eliminated. ([PMC12468286](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12468286/)) — **always split by speaker, never by segment/utterance.**

3. **"Clever Hans" silence artifact.** [arXiv:2406.07410](https://arxiv.org/pdf/2406.07410) — silent segments alone achieve near-100% classification accuracy on Pitt corpus, suggesting recording/processing artifacts leak the label through non-speech audio. **Always sanity-check acoustic models against a silence-only control; be suspicious of acoustic-only numbers above ~85%.**

4. **Demographic/MMSE confounds.** ADReSS is age/gender-balanced, but MMSE correlates so strongly with linguistic markers that some "AD detection" may partly be "MMSE-correlated verbal fluency detection."

5. **ASR transcript quality has a non-monotonic relationship with detection accuracy.** [arXiv:2412.06332](https://arxiv.org/pdf/2412.06332): a 33.9% WER ASR system matched manual-transcript accuracy (~88%) — specific asymmetric error patterns amplify AD-associated disfluencies rather than harming detection broadly. But other setups show text-only-on-ASR losing ~7.8pp vs manual transcripts. Don't assume an ASR pipeline transfers cleanly across ASR systems/populations.

6. **Cross-corpus generalization is weak.** [arXiv:2609.00276](https://arxiv.org/html/2609.00276) (2026) — ADReSSo's speech-enhancement/denoising preprocessing improves in-domain performance but **hurts cross-domain (Lu corpus) generalization**; models trained on unprocessed Pitt audio generalize better. Also induces systematic AD-class prediction bias in large audio-LMs. **Evaluate on raw/unenhanced Pitt extension or another corpus, not just the official test split, if real-world deployment matters.**

7. **LLMCare's external validation gap**: F1 85.7% in-domain → 72.8% external (Delaware MCI cohort) — concrete >12pp generalization gap even for a well-engineered 2025 pipeline.

8. **Small-N statistical power.** Test sets of 46-71 speakers: 3-4 correctly-classified subjects shift accuracy by 4-6pp. Treat 87% vs 90% headline differences as largely noise absent confidence intervals/significance testing — rarely reported in this field.

## Top Picks to Reimplement (single-GPU / Colab-scale budget)

### Pick 1 — Frozen-feature multimodal gated fusion (recommended starting point)
Template: "Listening Between the Lines" simplified. Frozen Whisper-encoder embeddings (small/base, mean+attention pooling) for acoustics — no encoder fine-tuning (avoids overfitting 166 examples). Fine-tune BERT-base/RoBERTa-base on Whisper-transcribed ASR text (optionally with pause-token insertion). Fuse via a small gated linear+sigmoid layer over concatenated pooled vectors. Runs on free-tier Colab GPU in minutes/epoch; off-the-shelf checkpoints (`openai/whisper-base`, `bert-base-uncased`); speech-only/ADReSSo-compliant; published numbers (83-90% F1) are genuine current SOTA. **Build in from day one:** strict speaker-level k-fold/LOSO CV, silence-only control classifier as Clever-Hans sanity check.

### Pick 2 — Text-only fine-tuned transformer with disfluency encoding (cheapest, strong, easiest to debug)
Template: Yuan et al. 2021, or WavBERT's text-side approach for fully speech-only. Whisper/wav2vec2 ASR → transcript with `[PAUSE]`/filler tokens inserted using Whisper word-level timestamps. Fine-tune BERT-base/RoBERTa-base as sequence classifier. Lowest engineering risk, trains in minutes on a T4, published ceiling (88-90% accuracy) nearly matches best multimodal systems. Use as the **baseline** regardless of which other picks are built, for ablation credibility.

### Pick 3 — Graph-augmented linguistic model (higher ceiling, moderate extra engineering)
Template: Gated Multi-Graph Fusion, scoped down. Add just the **PMI co-occurrence graph** (skip semantic/dependency graphs initially — PMI was the highest-value component in ablation) on top of BERT embeddings via `torch_geometric` GAT + mean pool, concatenated with BERT [CLS]. Recommended as a **second iteration** after Picks 1-2 validate the pipeline — PMI reference-corpus choice materially affects results and is easy to get subtly wrong.

**Explicitly avoid given limited compute:** full end-to-end fine-tuning of a large SSL audio encoder (wav2vec2-large, WavLM-large, Whisper-large) on the full 166-speaker training set — costs far more compute than frozen-feature approaches and doesn't reliably beat them; 166 speakers is not enough data to safely update hundreds of millions of encoder parameters.

## Sources

- [ADReSS Challenge (arXiv:2004.06833)](https://arxiv.org/abs/2004.06833) · [ADReSSo Challenge (ISCA)](https://www.isca-archive.org/interspeech_2021/luz21_interspeech.html) · [ADReSS-M Overview (PMC11218814)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11218814/)
- [TalkBank Access Levels](https://talkbank.org/0share/access.html) · [IRB Approval](https://talkbank.org/0share/irb/) · [Data Usage Rules](https://talkbank.org/0share/rules.html) · [DementiaBank Access](https://talkbank.org/dementia/access/)
- [To BERT or Not To BERT (arXiv:2008.01551)](https://arxiv.org/abs/2008.01551)
- [Linguistic Complexity + (Dis)Fluency + PLMs (arXiv:2106.08689)](https://arxiv.org/pdf/2106.08689)
- [WavBERT (PMC10102979)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10102979/)
- [Exploiting Pre-Trained ASR Models (arXiv:2110.01493)](https://arxiv.org/pdf/2110.01493)
- [data2vec end-to-end (PMC9856143)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9856143/)
- [HAFFormer (arXiv:2405.03952)](https://arxiv.org/pdf/2405.03952)
- [Clever Hans Effect (arXiv:2406.07410)](https://arxiv.org/pdf/2406.07410)
- [Acoustic/Lexical/Disfluency/Pause BiLSTM (arXiv:2106.15684)](https://arxiv.org/pdf/2106.15684)
- [Listening Between the Lines (arXiv:2606.30675)](https://arxiv.org/abs/2606.30675)
- [CogniAlign (arXiv:2506.01890)](https://arxiv.org/abs/2506.01890)
- [Graph-Based Disfluent Speech Modeling (Springer, MCPR 2026)](https://link.springer.com/chapter/10.1007/978-3-032-28393-1_31)
- [Gated Multi-Graph Fusion via GAT (arXiv:2606.31186)](https://arxiv.org/html/2606.31186v1)
- [ChatGPT vs Bard (PMC11048951)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11048951/)
- [Prompt Learning with PLMs (arXiv:2210.16539)](https://arxiv.org/abs/2210.16539)
- [LLM Prompt Engineering for AD Detection (arXiv:2501.00861)](https://arxiv.org/abs/2501.00861)
- [Reasoning-Based CoT Approach (arXiv:2506.01683)](https://arxiv.org/pdf/2506.01683)
- [MMSE-Calibrated Few-Shot Prompting (arXiv:2509.19926)](https://arxiv.org/abs/2509.19926)
- [Paired-LLM Perplexity (arXiv:2506.09315)](https://arxiv.org/pdf/2506.09315)
- [Not All Errors Are Equal (arXiv:2412.06332)](https://arxiv.org/pdf/2412.06332)
- [Cleaner Speech, Weaker Generalization (arXiv:2609.00276)](https://arxiv.org/html/2609.00276)
- [Data Leakage Scoping Review (PMC12468286)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12468286/)
- [awesome-dementia-detection](https://github.com/billzyx/awesome-dementia-detection) · [LLMCare](https://github.com/SpeechCARE/LLMCare) · [wazeerzulfikar/alzheimers-dementia](https://github.com/wazeerzulfikar/alzheimers-dementia)
