# research/code

## Pilot (underconfidence under dropout)

```bash
python research/code/pilot_confidence_drift.py
```

## Study v2 (high-confidence errors)

```bash
python research/code/study_high_confidence_errors.py
```

## Study v3 (in progress — CLINC150)

Drop the script here when the run exists:

```
research/code/study_real_data_clinc150.py
```

No results in this folder until that file is real.

## Analysis reports (CSV + HTML/MD)

```bash
python research/code/generate_reports.py
```

Writes `research/reports/` (analysis report, pilot/study/comparison/probe CSVs).

Shared deps:

```bash
pip install -r research/code/requirements.txt
```
