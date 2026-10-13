# mmpsy-quicklook: Exploratory Analysis of an Adolescent Mental-Health Interview Dataset

Before trying to reproduce the published model, I wanted to look at the data
myself — without importing the paper's assumptions. This repository documents
that first pass: exploratory analyses of transcribed spoken responses from a
public adolescent mental-health dataset, paired with PHQ-9 and GAD-7 scores.

## Data & ethics

- **Source**: [mmpsy-data](https://github.com/shuyeit/mmpsy-data) — the public
  release of the MMPsy corpus, introduced by *Mental-Perceiver: Audio-Textual
  Multi-Modal Learning for Estimating Mental Disorders* (Qin et al., AAAI 2025;
  arXiv:2408.12088). It contains transcripts and audio features of
  Mandarin-speaking adolescents undergoing automated mental-health assessment
  interviews, with self-report PHQ-9/GAD-7 scores.
- **Who the participants are**: minors. The source repository states that the
  data were collected in cooperation with the local education authority's
  annual mental-health screening and anonymized before release; audio is
  shared only as mel-spectrogram features for privacy reasons.
- **My use of it**: transcripts + scale scores only. **This repository
  redistributes none of the raw data.** Everything published here is code,
  aggregate statistics, and figures checked to contain no verbatim responses.
  All analyses are exploratory and group-level; nothing here is intended as,
  or capable of, individual diagnosis.
- The source dataset is released under the MIT License (see the source
  repository); citation below.

## If you're starting with this dataset

Three things I wish I had known on day one:

1. **Interview paths were adaptive** — which topic each student was asked
   depended on their previous answer, so group differences partly reflect what
   was asked, not only what was said. Frame text-level contrasts as
   exploratory.
2. **Response length scales ~2.6× with severity** (275 → 719 chars from Normal
   to Severe) — any count-based metric must be normalized per 1,000 words
   before comparing groups. A raw-count filler-word comparison of mine looked
   monotonic and collapsed under normalization.
3. **The Severe group has only 15 students (1.2%)** — treat Severe-level
   signals with care.

Baseline reference points (shared 5-fold splits): TF–IDF + logistic regression
AUC 0.634; fine-tuned MacBERT 0.745–0.767 across three identical reruns (GPU
nondeterminism — expect ±0.01 drift between runs).

## Questions

1. How far do simple lexical features go toward distinguishing severity
   levels — and where exactly do they stop working?
2. What do richer representations (a fine-tuned Chinese BERT) add over those
   features?
3. Do speakers across severity groups differ in *how* they connect clauses
   (transition words), not just in *what* they say?

## Analyses & findings

| Script | What it does | Key result |
|---|---|---|
| `analysis/01_overview.py` | data profile: distributions, class imbalance, length × severity | mean response length rises monotonically with severity: 275 → 332 → 476 → 719 chars (Normal → Severe) |
| `analysis/02_wordfreq.py` | word-frequency contrasts, extreme groups | Severe-only items include 自杀/逃避/消失/恐惧/压抑; Normal-skewed: 开心/朋友/一起 |
| `analysis/03_baseline.py` | TF–IDF + logistic regression, 5-fold CV | AUC 0.634 — lexical features work at the extremes; Mild (largest group) is the blind spot |
| `analysis/04_bert.py` | MacBERT fine-tuning, same 5-fold splits | classification AUC 0.745–0.767 across three reruns (GPU nondeterminism; the range is reported, not a single run); PHQ-9 regression: Pearson r = 0.551, MAE 2.50 (paper's multi-task version: r = 0.706) |
| `analysis/05_transition.py` | transition-word density across severity (my own hypothesis, exploratory) | per 1k words: 1.34 (Normal) → 2.59 (Moderate) → 2.33 (Severe), with a shift in forms (但是/可是 ↓, 却 ↑); share of speakers using any transition: 27% → 54% |
| `analysis/07_same_word_contexts.py` | observational probe: same word across contexts vs different words (figure in `figures/`) | same-word cross-context similarity stays high (≈0.88–0.97 vs cross-word baseline ≈0.76): word identity dominates, sensitivity to context is graded across words — a limit on reading stance off embeddings |

Result files under `results/` are working notes written in Chinese; the tables
and numbers are self-explanatory.

Interpretation stays deliberately **exploratory**: interview paths were
adaptive (which topic a student was asked depended on their previous answer),
and response length scales ~2.6× with severity, so every count-based contrast
is reported per-1,000-words. A raw-count version of a filler-word comparison
initially looked monotonic and did not survive density normalization — kept as
a methodological note in script 02.

## Reproduce

```bash
# obtain the data from the source repository (see Data & ethics) into data/, then:
python analysis/01_overview.py
python analysis/02_wordfreq.py
python analysis/03_baseline.py
python analysis/04_bert.py      # needs GPU + transformers
python analysis/05_transition.py
python analysis/07_same_word_contexts.py
```

Dependencies: pandas, jieba, scikit-learn, matplotlib, transformers, torch.

## Citation

If you use the underlying dataset, please cite the source:

```bibtex
@dataset{qin2024mentalperceiverdataset,
  author = {Jinghui Qin and Changsong Liu and Tianchi Tang and Dahuang Liu and Minghao Wang and Qianying Huang and Yang Xu and Rumin Zhang},
  title = {Mental-Perceiver Dataset: Audio-Textual Multimodal Data for Mental Health Assessment},
  year = {2024},
  url = {https://github.com/shuyeit/mmpsy-data}
}
```

Analysis code in this repository: MIT License.
