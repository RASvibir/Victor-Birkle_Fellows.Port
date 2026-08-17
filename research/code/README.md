# research/code

## Pilot (underconfidence under dropout)

```bash
python research/code/pilot_confidence_drift.py
```

## Study v2 (high-confidence errors)

```bash
python research/code/study_high_confidence_errors.py
```

## Study v3 (CLINC150, BoW)

```bash
python research/code/study_real_data_clinc150.py
```

Requires `datasets` (see requirements.txt). Writes `research/study_v3_metrics.json` and `research/figures/study_v3_*.png`.

## Study v4 (CLINC150, BoW + DistilBERT)

Needs Python 3.10–3.12 (PyTorch wheels; the repo `.venv` may be 3.14):

```bash
uv venv .venv-encoder --python 3.12
uv pip install -r research/code/requirements-encoder.txt
python research/code/study_encoder_clinc150.py
```

Writes `research/study_v4_metrics.json` and `research/figures/study_v4_*.png`. Checkpoint under `research/code/out/` (gitignored).

## Analysis reports (CSV + HTML/MD)

```bash
python research/code/generate_reports.py
```

Also regenerates `research/study_v4.html` / `.md` from the metrics JSON when present.

Shared deps (pilot / v2 / v3):

```bash
pip install -r research/code/requirements.txt
```
