# Analysis report — confidence vs. accuracy under drift

**Author:** Victor E. Birkle III  
**Date:** August 2026  
**Scope:** Synthetic intent-classification pilots (not production ASR logs)  
**Companion figures:** `research/figures/`  
**Machine-readable tables:** see CSVs linked below

---

## 1. Executive summary

Two finished synthetic passes tested when reported classifier confidence stops tracking accuracy.

| Pass | Mechanism | Primary result |
|------|-----------|----------------|
| Pilot | Token dropout + vocab injection | **Underconfidence** (confidence below accuracy) |
| Study v2 | Competing-intent cue injection | **Overconfidence** + high-confidence errors |
| Study v2 | ASR keyword swaps | Mostly underconfidence; weak high-conf errors |
| Study v2 | Prior shift | Did not systematically create overconfidence |

**Analytical claim:** on this bag-of-words logistic baseline, *corruption type* — not merely “noise amount” — determines the calibration failure mode.

---

## 2. Methods (analysis-facing)

### Shared model
- Data: synthetic 8-class intent utterances (seed 42)
- Split: 1232 train / 528 test
- Model: `CountVectorizer(1-2gram) + LogisticRegression`
- Confidence: max softmax probability
- Calibration: 10-bin Expected Calibration Error (ECE)
- Gap: mean confidence − accuracy (positive ⇒ overconfidence)

### Pilot corruption
`token dropout (~0.85*level) + random vocab injection` at levels 0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8.

### Study v2 regimes
asr_swap, cue_inject, prior_shift; high-confidence error threshold = 0.8.

### Operational probe (study v2)
Binary target: `|confidence − correctness| ≥ 0.5`.  
Features: `length, corruption_proxy, regime_code, level (confidence excluded)` — confidence excluded by design.

---

## 3. Results tables

### 3.1 Pilot metrics

| Level | Accuracy | Mean conf. | ECE | Conf − Acc |
|------:|---------:|-----------:|----:|-----------:|
| 0.00 | 0.9886 | 0.9506 | 0.0380 | -0.0380 |
| 0.10 | 0.9735 | 0.9213 | 0.0636 | -0.0522 |
| 0.20 | 0.9432 | 0.8406 | 0.1025 | -0.1025 |
| 0.35 | 0.8788 | 0.7596 | 0.1214 | -0.1192 |
| 0.50 | 0.7765 | 0.6545 | 0.1223 | -0.1220 |
| 0.65 | 0.6742 | 0.5631 | 0.1153 | -0.1111 |
| 0.80 | 0.5549 | 0.4720 | 0.0829 | -0.0829 |

Clean baseline: accuracy 0.9886, ECE 0.0380.  
Harsh dropout (@0.8): accuracy 0.5549, gap -0.0829.

CSV: [`pilot_metrics.csv`](./pilot_metrics.csv)

### 3.2 Study v2 — key cells

Worst high-conf-error cell: **cue_inject @ 0.6**

| Metric | Value |
|--------|------:|
| Accuracy | 0.5511 |
| Mean confidence | 0.7510 |
| ECE | 0.2158 |
| Conf − Acc | +0.1999 |
| Errors | 237 |
| P(conf≥0.8 | error) | 0.4262 |

Full regime grid CSV: [`study_v2_metrics.csv`](./study_v2_metrics.csv)

### 3.3 Cross-condition comparison

CSV: [`comparison_summary.csv`](./comparison_summary.csv)

### 3.4 Operational probe

| Item | Value |
|------|------:|
| AUC | 0.7226 |
| N | 7920 |
| Positive rate | 0.1523 |

Coefficients:

| Feature | Coef |
|---------|-----:|
| length | -0.0114 |
| corruption_proxy | 1.6855 |
| regime_code | -0.2404 |
| level | 1.5196 |

CSV: [`study_v2_operational_probe.csv`](./study_v2_operational_probe.csv)

---

## 4. Interpretation

1. **Dropout destroys evidence** → softmax spreads → underconfidence (pilot).
2. **Rival cue injection activates the wrong class** while labels stay true → overconfidence and high-confidence mistakes (study v2).
3. **ASR swaps** on this lexical model mostly reduce confidence with accuracy; they are a weak proxy for the production pathology.
4. **Prior shift alone** is insufficient here to recreate high-confidence errors.
5. **Ops features without confidence** give a moderate detector (AUC 0.7226) — early-warning grade, not solved.

---

## 5. Threats to validity

- Synthetic templates, not real ASR lattices or production transcripts.
- Cue injection is a strong artificial mechanism; it validates the *measurement loop*, not the exact production cause.
- Single model family (linear bag-of-words); neural encoders may differ.
- Operational probe is in-sample across stacked regime rows — treat AUC as descriptive, not a holdout claim.

---

## 6. Reproducibility

```bash
pip install -r research/code/requirements.txt
python research/code/pilot_confidence_drift.py
python research/code/study_high_confidence_errors.py
python research/code/generate_reports.py
```

Source metrics JSON:
- [`../pilot_metrics.json`](../pilot_metrics.json)
- [`../study_v2_metrics.json`](../study_v2_metrics.json)

Figures:
- [`../figures/pilot_ece_vs_corruption.png`](../figures/pilot_ece_vs_corruption.png)
- [`../figures/pilot_reliability.png`](../figures/pilot_reliability.png)
- [`../figures/study_v2_regimes.png`](../figures/study_v2_regimes.png)
- [`../figures/study_v2_reliability.png`](../figures/study_v2_reliability.png)

---

## 7. Next analysis step

Production-faithful ASR/domain corruptions with a **held-out** operational detector, scored against the same ECE / gap / high-conf-error report schema used here.
