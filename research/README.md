# Research artifacts

| Note | Status |
|------|--------|
| [Analysis report + CSVs](./reports/) | **Complete** |
| [Study v2: high-confidence errors](./study_v2.md) | **Complete** |
| [Pilot: token corruption](./pilot.md) | **Complete** |
| [Plan: production-faithful pass](./confidence-accuracy-divergence.md) | Next |

## Quick links

- Hub: [index.html](./index.html)
- **Reports:** [reports/](./reports/) · [analysis_report.md](./reports/analysis_report.md)
  - [pilot_metrics.csv](./reports/pilot_metrics.csv)
  - [study_v2_metrics.csv](./reports/study_v2_metrics.csv)
  - [comparison_summary.csv](./reports/comparison_summary.csv)
  - [study_v2_operational_probe.csv](./reports/study_v2_operational_probe.csv)
- Study v2: [study_v2.html](./study_v2.html) · [code](./code/study_high_confidence_errors.py)
- Pilot: [pilot.html](./pilot.html) · [code](./code/pilot_confidence_drift.py)
- Figures: [figures/](./figures/)

## Regenerate reports

```bash
python research/code/generate_reports.py
```
