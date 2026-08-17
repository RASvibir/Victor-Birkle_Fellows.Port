# Research artifacts

| Note | Status |
|------|--------|
| [Study v4: encoder vs BoW on CLINC150](./study_v4.md) | **Complete** (DistilBERT also did not reproduce v2 overconfidence) |
| [Study v3: CLINC150 real data](./study_v3.md) | **Complete** (mixed / null vs v2 overconfidence) |
| [Analysis report + CSVs](./reports/) | **Complete** |
| [Study v2: high-confidence errors](./study_v2.md) | **Complete** |
| [Pilot: token corruption](./pilot.md) | **Complete** |
| [Plan](./confidence-accuracy-divergence.md) | Living |

## Quick links

- Hub: [index.html](./index.html)
- **Reports:** [reports/](./reports/) · [analysis_report.md](./reports/analysis_report.md)
  - [pilot_metrics.csv](./reports/pilot_metrics.csv)
  - [study_v2_metrics.csv](./reports/study_v2_metrics.csv)
  - [study_v3_metrics.csv](./reports/study_v3_metrics.csv)
  - [study_v4_metrics.csv](./reports/study_v4_metrics.csv)
  - [study_v4_ood.csv](./reports/study_v4_ood.csv)
  - [comparison_summary.csv](./reports/comparison_summary.csv)
  - [study_v2_operational_probe.csv](./reports/study_v2_operational_probe.csv)
- Study v4: [study_v4.html](./study_v4.html) · [code](./code/study_encoder_clinc150.py)
- Study v3: [study_v3.html](./study_v3.html) · [code](./code/study_real_data_clinc150.py)
- Study v2: [study_v2.html](./study_v2.html) · [code](./code/study_high_confidence_errors.py)
- Pilot: [pilot.html](./pilot.html) · [code](./code/pilot_confidence_drift.py)
- Figures: [figures/](./figures/)

## Regenerate reports

```bash
python research/code/generate_reports.py
```
