# Study v3 — real data (CLINC150)

**Status:** Complete · mixed / null vs. synthetic overconfidence  
**Author:** Victor E. Birkle III  
**Date:** August 2026  
**Code:** [`research/code/study_real_data_clinc150.py`](./code/study_real_data_clinc150.py)  
**Metrics:** [`research/study_v3_metrics.json`](./study_v3_metrics.json)  
**Data:** `DeepPavlov/clinc150` (CC BY 4.0), fingerprint `53c91a8fb52c3fe7`

## Headline

On real CLINC150 text, **neither cue injection nor OOS reproduced study v2’s overconfidence.**

Cue injection still drops accuracy, but confidence drops with it. OOS errors are mostly low-confidence (mean conf 0.215; 2.8% of OOS errors have conf ≥ 0.8). MSP AUROC for in-domain vs OOS is **0.921** (study v4 recompute on this model).

## Numbers

| Condition | n | Acc | Mean conf | ECE | Conf − Acc | P(conf≥0.8\|err) |
|-----------|--:|----:|----------:|----:|-----------:|-----------------:|
| in-domain | 4500 | 0.898 | 0.754 | 0.144 | −0.144 | 0.059 |
| cue_inject @ 0.3 | 4500 | 0.715 | 0.600 | 0.115 | −0.115 | 0.039 |
| cue_inject @ 0.6 | 4500 | 0.535 | 0.516 | 0.068 | −0.019 | 0.091 |
| OOS | 1000 | 0.000 | 0.215 | 0.215 | +0.215* | 0.028 |

\*OOS accuracy is 0 by construction (no OOS class in the model). Gap = mean confidence. That is not v2-style overconfidence.

v2 cue_inject@0.6 was conf−acc **+0.20** and P(high-conf\|error) **0.43**.

Ops probe AUC **0.608** (length, OOV rate, regime, cue level; confidence excluded). Weaker than v2.

## Why v2 and v3 differ

Real utterances are more redundant than the synthetic templates, so rival keywords rarely dominate. In-domain v3 is already underconfident. OOS here is unseen-intent text, not a keyword attack.

```bash
python research/code/study_real_data_clinc150.py
```
