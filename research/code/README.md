# research/code

## Pilot (underconfidence under dropout)

```bash
python research/code/pilot_confidence_drift.py
```

## Study v2 (high-confidence errors)

```bash
python research/code/study_high_confidence_errors.py
```

## Analysis reports (CSV + HTML/MD)

```bash
python research/code/generate_reports.py
```

Writes `research/reports/` (analysis report, pilot/study/comparison/probe CSVs).

Shared deps:

```bash
pip install -r research/code/requirements.txt
```
