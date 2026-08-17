# research/code

## Pilot

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r research/code/requirements.txt
python research/code/pilot_confidence_drift.py
```

Writes:

- `research/figures/pilot_ece_vs_corruption.png`
- `research/figures/pilot_reliability.png`
- `research/code/out/pilot_metrics.json` (local; committed summary is `research/pilot_metrics.json`)
