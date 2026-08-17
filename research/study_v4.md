# Study v4 — encoder vs BoW on CLINC150

**Status:** encoder also did not reproduce synthetic overconfidence
**Author:** Victor E. Birkle III
**Date:** August 2026
**Code:** [`research/code/study_encoder_clinc150.py`](./code/study_encoder_clinc150.py)
**Metrics:** [`research/study_v4_metrics.json`](./study_v4_metrics.json)
**Data:** `DeepPavlov/clinc150` (CC BY 4.0 (CLINC150 / DeepPavlov HF release)), fingerprint `6369fc73a44085b8`

## Headline

A fine-tuned encoder on the same CLINC150 splits still does **not** reproduce study v2’s overconfidence. Cue-inject@0.6: conf−acc +0.023, P(conf≥0.8|error) 0.021. OOS mean confidence 0.204 (P(high-conf|error) 0.006). MSP already separates in-domain from OOS (BoW AUROC 0.921; encoder 0.947). The synthetic pathology needs a corruption that makes the model confidently wrong, not only a stronger model or an unseen-intent split.

## MSP OOD (ID vs OOS)

| Model | ID mean conf | OOS mean conf | MSP AUROC |
|-------|-------------:|--------------:|----------:|
| BoW logistic | 0.754 | 0.215 | 0.921 |
| DistilBERT | 0.657 | 0.204 | 0.947 |

## BoW logistic

| Condition | n | Acc | Mean conf | ECE | Conf − Acc | P(conf≥0.8\|err) |
|---|--:|----:|----------:|----:|-----------:|------------------:|
| in_domain | 4500 | 0.898 | 0.754 | 0.144 | -0.144 | 0.059 |
| cue_inject @ 0.3 | 4500 | 0.715 | 0.600 | 0.115 | -0.115 | 0.039 |
| cue_inject @ 0.6 | 4500 | 0.535 | 0.516 | 0.068 | -0.019 | 0.091 |
| OOS | 1000 | 0.000 | 0.215 | 0.215 | +0.215 | 0.028 |

## DistilBERT

| Condition | n | Acc | Mean conf | ECE | Conf − Acc | P(conf≥0.8\|err) |
|---|--:|----:|----------:|----:|-----------:|------------------:|
| in_domain | 4500 | 0.924 | 0.657 | 0.267 | -0.267 | 0.006 |
| cue_inject @ 0.3 | 4500 | 0.597 | 0.478 | 0.119 | -0.119 | 0.015 |
| cue_inject @ 0.6 | 4500 | 0.388 | 0.411 | 0.057 | +0.023 | 0.021 |
| OOS | 1000 | 0.000 | 0.204 | 0.204 | +0.204 | 0.006 |

In-domain DistilBERT is more accurate than BoW (0.924 vs 0.898) and more underconfident (gap -0.267). Cue-inject@0.6 has a tiny positive gap (+0.023) but P(high-conf|error) 0.021 — not v2’s 0.43. BoW metrics match study v3 bit-for-bit.

## Run

```bash
uv venv .venv-encoder --python 3.12
uv pip install -r research/code/requirements-encoder.txt
python research/code/study_encoder_clinc150.py
python research/code/generate_reports.py
```
