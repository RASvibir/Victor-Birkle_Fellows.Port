"""
Generate analysis reports (CSV tables + comparative markdown/HTML) from
committed experiment metrics JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
REPORTS = ROOT / "reports"
PILOT_JSON = ROOT / "pilot_metrics.json"
STUDY_JSON = ROOT / "study_v2_metrics.json"
STUDY_V3_JSON = ROOT / "study_v3_metrics.json"
STUDY_V4_JSON = ROOT / "study_v4_metrics.json"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def load_pilot() -> dict:
    return json.loads(PILOT_JSON.read_text())


def load_study() -> dict:
    return json.loads(STUDY_JSON.read_text())


def load_study_v3() -> dict:
    return json.loads(STUDY_V3_JSON.read_text())


def load_study_v4() -> dict | None:
    if not STUDY_V4_JSON.exists():
        return None
    return json.loads(STUDY_V4_JSON.read_text())


def export_tables(pilot: dict, study: dict, study_v3: dict, study_v4: dict | None = None) -> dict[str, Path]:
    pilot_csv = REPORTS / "pilot_metrics.csv"
    study_csv = REPORTS / "study_v2_metrics.csv"
    study_v3_csv = REPORTS / "study_v3_metrics.csv"
    compare_csv = REPORTS / "comparison_summary.csv"
    probe_csv = REPORTS / "study_v2_operational_probe.csv"

    write_csv(
        pilot_csv,
        pilot["rows"],
        ["level", "accuracy", "mean_confidence", "ece", "gap_conf_minus_acc"],
    )
    write_csv(
        study_csv,
        study["rows"],
        [
            "regime",
            "level",
            "accuracy",
            "mean_confidence",
            "ece",
            "gap_conf_minus_acc",
            "high_conf_error_rate",
            "n_errors",
        ],
    )

    write_csv(
        study_v3_csv,
        study_v3["rows"],
        [
            "regime",
            "level",
            "accuracy",
            "mean_confidence",
            "ece",
            "gap_conf_minus_acc",
            "high_conf_error_rate",
            "n",
            "n_errors",
        ],
    )

    worst = study["worst_cell"]
    pilot_mid = next(r for r in pilot["rows"] if abs(r["level"] - 0.5) < 1e-9)
    pilot_harsh = next(r for r in pilot["rows"] if abs(r["level"] - 0.8) < 1e-9)
    asr_harsh = next(
        r for r in study["rows"] if r["regime"] == "asr_swap" and abs(r["level"] - 0.75) < 1e-9
    )
    prior_harsh = next(
        r for r in study["rows"] if r["regime"] == "prior_shift" and abs(r["level"] - 0.75) < 1e-9
    )

    compare_rows = [
        {
            "condition": "pilot_dropout@0.5",
            "accuracy": pilot_mid["accuracy"],
            "mean_confidence": pilot_mid["mean_confidence"],
            "ece": pilot_mid["ece"],
            "gap_conf_minus_acc": pilot_mid["gap_conf_minus_acc"],
            "high_conf_error_rate": "",
            "failure_mode": "underconfidence",
        },
        {
            "condition": "pilot_dropout@0.8",
            "accuracy": pilot_harsh["accuracy"],
            "mean_confidence": pilot_harsh["mean_confidence"],
            "ece": pilot_harsh["ece"],
            "gap_conf_minus_acc": pilot_harsh["gap_conf_minus_acc"],
            "high_conf_error_rate": "",
            "failure_mode": "underconfidence",
        },
        {
            "condition": f"study_v2_{worst['regime']}@{worst['level']}",
            "accuracy": worst["accuracy"],
            "mean_confidence": worst["mean_confidence"],
            "ece": worst["ece"],
            "gap_conf_minus_acc": worst["gap"],
            "high_conf_error_rate": worst["high_conf_error_rate"],
            "failure_mode": "overconfidence",
        },
        {
            "condition": "study_v2_asr_swap@0.75",
            "accuracy": asr_harsh["accuracy"],
            "mean_confidence": asr_harsh["mean_confidence"],
            "ece": asr_harsh["ece"],
            "gap_conf_minus_acc": asr_harsh["gap_conf_minus_acc"],
            "high_conf_error_rate": asr_harsh["high_conf_error_rate"],
            "failure_mode": "underconfidence",
        },
        {
            "condition": "study_v2_prior_shift@0.75",
            "accuracy": prior_harsh["accuracy"],
            "mean_confidence": prior_harsh["mean_confidence"],
            "ece": prior_harsh["ece"],
            "gap_conf_minus_acc": prior_harsh["gap_conf_minus_acc"],
            "high_conf_error_rate": prior_harsh["high_conf_error_rate"],
            "failure_mode": "weak / mixed",
        },
        {
            "condition": "study_v3_in_domain",
            "accuracy": next(r["accuracy"] for r in study_v3["rows"] if r["regime"] == "in_domain"),
            "mean_confidence": next(r["mean_confidence"] for r in study_v3["rows"] if r["regime"] == "in_domain"),
            "ece": next(r["ece"] for r in study_v3["rows"] if r["regime"] == "in_domain"),
            "gap_conf_minus_acc": next(r["gap_conf_minus_acc"] for r in study_v3["rows"] if r["regime"] == "in_domain"),
            "high_conf_error_rate": next(r["high_conf_error_rate"] for r in study_v3["rows"] if r["regime"] == "in_domain"),
            "failure_mode": "underconfidence",
        },
        {
            "condition": "study_v3_cue_inject@0.6",
            "accuracy": next(r["accuracy"] for r in study_v3["rows"] if r["regime"] == "cue_inject" and r["level"] == 0.6),
            "mean_confidence": next(r["mean_confidence"] for r in study_v3["rows"] if r["regime"] == "cue_inject" and r["level"] == 0.6),
            "ece": next(r["ece"] for r in study_v3["rows"] if r["regime"] == "cue_inject" and r["level"] == 0.6),
            "gap_conf_minus_acc": next(r["gap_conf_minus_acc"] for r in study_v3["rows"] if r["regime"] == "cue_inject" and r["level"] == 0.6),
            "high_conf_error_rate": next(r["high_conf_error_rate"] for r in study_v3["rows"] if r["regime"] == "cue_inject" and r["level"] == 0.6),
            "failure_mode": "null vs v2 overconfidence (gap still negative)",
        },
        {
            "condition": "study_v3_oos",
            "accuracy": next(r["accuracy"] for r in study_v3["rows"] if r["regime"] == "oos"),
            "mean_confidence": next(r["mean_confidence"] for r in study_v3["rows"] if r["regime"] == "oos"),
            "ece": next(r["ece"] for r in study_v3["rows"] if r["regime"] == "oos"),
            "gap_conf_minus_acc": next(r["gap_conf_minus_acc"] for r in study_v3["rows"] if r["regime"] == "oos"),
            "high_conf_error_rate": next(r["high_conf_error_rate"] for r in study_v3["rows"] if r["regime"] == "oos"),
            "failure_mode": "low-confidence errors (not v2 overconfidence)",
        },
    ]
    if study_v4 and "distilbert" in study_v4.get("models", {}):
        enc = study_v4["models"]["distilbert"]
        for r in enc["rows"]:
            if r["regime"] == "cue_inject" and r["level"] != 0.6:
                continue
            label = {
                "in_domain": "study_v4_distilbert_in_domain",
                "cue_inject": "study_v4_distilbert_cue_inject@0.6",
                "oos": "study_v4_distilbert_oos",
            }[r["regime"]]
            compare_rows.append(
                {
                    "condition": label,
                    "accuracy": r["accuracy"],
                    "mean_confidence": r["mean_confidence"],
                    "ece": r["ece"],
                    "gap_conf_minus_acc": r["gap_conf_minus_acc"],
                    "high_conf_error_rate": r["high_conf_error_rate"],
                    "failure_mode": "encoder column (see study v4)",
                }
            )
    write_csv(
        compare_csv,
        compare_rows,
        [
            "condition",
            "accuracy",
            "mean_confidence",
            "ece",
            "gap_conf_minus_acc",
            "high_conf_error_rate",
            "failure_mode",
        ],
    )

    probe = study["operational_probe"]
    probe_rows = [
        {"feature": "auc", "value": probe["auc"]},
        {"feature": "n", "value": probe["n"]},
        {"feature": "positive_rate", "value": probe["positive_rate"]},
        {"feature": "features", "value": probe.get("features", "")},
    ]
    for name, coef in probe["coef"].items():
        probe_rows.append({"feature": f"coef_{name}", "value": coef})
    write_csv(probe_csv, probe_rows, ["feature", "value"])

    written = {
        "pilot_csv": pilot_csv,
        "study_csv": study_csv,
        "study_v3_csv": study_v3_csv,
        "compare_csv": compare_csv,
        "probe_csv": probe_csv,
    }
    if study_v4:
        v4_rows = []
        ood_rows = []
        for name, model in study_v4["models"].items():
            for r in model["rows"]:
                v4_rows.append({"model": name, **r})
            ood = dict(model["ood_detection"])
            ood["model"] = name
            ood_rows.append(ood)
        v4_csv = REPORTS / "study_v4_metrics.csv"
        ood_csv = REPORTS / "study_v4_ood.csv"
        write_csv(
            v4_csv,
            v4_rows,
            [
                "model",
                "regime",
                "level",
                "accuracy",
                "mean_confidence",
                "ece",
                "gap_conf_minus_acc",
                "high_conf_error_rate",
                "n",
                "n_errors",
            ],
        )
        write_csv(
            ood_csv,
            ood_rows,
            ["model", "auroc_id_vs_oos", "id_mean_conf", "oos_mean_conf", "n_id", "n_oos", "method", "note"],
        )
        written["study_v4_csv"] = v4_csv
        written["study_v4_ood_csv"] = ood_csv
    return written


def render_markdown(pilot: dict, study: dict, paths: dict[str, Path]) -> str:
    worst = study["worst_cell"]
    probe = study["operational_probe"]
    pilot_clean = pilot["rows"][0]
    pilot_harsh = next(r for r in pilot["rows"] if abs(r["level"] - 0.8) < 1e-9)

    return f"""# Analysis report — confidence vs. accuracy under drift

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
- Data: synthetic 8-class intent utterances (seed {pilot["seed"]})
- Split: {pilot["n_train"]} train / {pilot["n_test"]} test
- Model: `{pilot["model"]}`
- Confidence: max softmax probability
- Calibration: 10-bin Expected Calibration Error (ECE)
- Gap: mean confidence − accuracy (positive ⇒ overconfidence)

### Pilot corruption
`{pilot["corruption"]}` at levels {", ".join(str(r["level"]) for r in pilot["rows"])}.

### Study v2 regimes
{", ".join(study["regimes"])}; high-confidence error threshold = {study["high_conf_threshold"]}.

### Operational probe (study v2)
Binary target: `|confidence − correctness| ≥ 0.5`.  
Features: `{probe.get("features", "length, corruption_proxy, regime_code, level")}` — confidence excluded by design.

---

## 3. Results tables

### 3.1 Pilot metrics

| Level | Accuracy | Mean conf. | ECE | Conf − Acc |
|------:|---------:|-----------:|----:|-----------:|
{chr(10).join(
    f"| {r['level']:.2f} | {r['accuracy']:.4f} | {r['mean_confidence']:.4f} | {r['ece']:.4f} | {r['gap_conf_minus_acc']:+.4f} |"
    for r in pilot["rows"]
)}

Clean baseline: accuracy {pilot_clean["accuracy"]:.4f}, ECE {pilot_clean["ece"]:.4f}.  
Harsh dropout (@0.8): accuracy {pilot_harsh["accuracy"]:.4f}, gap {pilot_harsh["gap_conf_minus_acc"]:+.4f}.

CSV: [`pilot_metrics.csv`](./pilot_metrics.csv)

### 3.2 Study v2 — key cells

Worst high-conf-error cell: **{worst["regime"]} @ {worst["level"]}**

| Metric | Value |
|--------|------:|
| Accuracy | {worst["accuracy"]:.4f} |
| Mean confidence | {worst["mean_confidence"]:.4f} |
| ECE | {worst["ece"]:.4f} |
| Conf − Acc | {worst["gap"]:+.4f} |
| Errors | {worst["n_errors"]} |
| P(conf≥{study["high_conf_threshold"]} | error) | {worst["high_conf_error_rate"]:.4f} |

Full regime grid CSV: [`study_v2_metrics.csv`](./study_v2_metrics.csv)

### 3.3 Cross-condition comparison

CSV: [`comparison_summary.csv`](./comparison_summary.csv)

### 3.4 Operational probe

| Item | Value |
|------|------:|
| AUC | {probe["auc"]} |
| N | {probe["n"]} |
| Positive rate | {probe["positive_rate"]:.4f} |

Coefficients:

| Feature | Coef |
|---------|-----:|
{chr(10).join(f"| {k} | {v:.4f} |" for k, v in probe["coef"].items())}

CSV: [`study_v2_operational_probe.csv`](./study_v2_operational_probe.csv)

---

## 4. Interpretation

1. **Dropout destroys evidence** → softmax spreads → underconfidence (pilot).
2. **Rival cue injection activates the wrong class** while labels stay true → overconfidence and high-confidence mistakes (study v2).
3. **ASR swaps** on this lexical model mostly reduce confidence with accuracy; they are a weak proxy for the production pathology.
4. **Prior shift alone** is insufficient here to recreate high-confidence errors.
5. **Ops features without confidence** give a moderate detector (AUC {probe["auc"]}) — early-warning grade, not solved.
6. **Study v3 (CLINC150):** cue-inject on real text did **not** reproduce v2 overconfidence (gap stayed negative). OOS errors were mostly low-confidence (mean conf 0.215). In-domain was already underconfident. MSP AUROC for ID vs OOS on the BoW model is in study v4.
7. **Study v4:** same splits, DistilBERT column + MSP AUROC. See [`../study_v4.html`](../study_v4.html).

---

## 5. Threats to validity

- Synthetic templates, not real ASR lattices or production transcripts.
- Cue injection is a strong artificial mechanism; it validates the *measurement loop*, not the exact production cause.
- Study v3 is a linear bag-of-words model; study v4 adds one fine-tuned encoder on the same splits.
- Operational probe is in-sample across stacked regime rows — treat AUC as descriptive, not a holdout claim.

---

## 6. Reproducibility

```bash
pip install -r research/code/requirements.txt
python research/code/pilot_confidence_drift.py
python research/code/study_high_confidence_errors.py
python research/code/study_real_data_clinc150.py
python research/code/study_encoder_clinc150.py
python research/code/generate_reports.py
```

Source metrics JSON:
- [`../pilot_metrics.json`](../pilot_metrics.json)
- [`../study_v2_metrics.json`](../study_v2_metrics.json)
- [`../study_v3_metrics.json`](../study_v3_metrics.json)
- [`../study_v4_metrics.json`](../study_v4_metrics.json)

Figures:
- [`../figures/pilot_ece_vs_corruption.png`](../figures/pilot_ece_vs_corruption.png)
- [`../figures/pilot_reliability.png`](../figures/pilot_reliability.png)
- [`../figures/study_v2_regimes.png`](../figures/study_v2_regimes.png)
- [`../figures/study_v2_reliability.png`](../figures/study_v2_reliability.png)

---

## 7. Next analysis step

The encoder column is the test of whether the CLINC150 null was a model-class artifact. A full ASR pipeline is out of scope until that column is interpreted.
"""


def render_html(md_body_note: str) -> str:
    """Simple HTML shell linking tables; detailed narrative lives in analysis_report.md too."""
    pilot = load_pilot()
    study = load_study()
    worst = study["worst_cell"]
    probe = study["operational_probe"]

    pilot_rows = "".join(
        f"<tr><td>{r['level']:.2f}</td><td>{r['accuracy']:.4f}</td>"
        f"<td>{r['mean_confidence']:.4f}</td><td>{r['ece']:.4f}</td>"
        f"<td>{r['gap_conf_minus_acc']:+.4f}</td></tr>"
        for r in pilot["rows"]
    )
    study_rows = "".join(
        f"<tr><td>{r['regime']}</td><td>{r['level']:.2f}</td><td>{r['accuracy']:.4f}</td>"
        f"<td>{r['mean_confidence']:.4f}</td><td>{r['ece']:.4f}</td>"
        f"<td>{r['gap_conf_minus_acc']:+.4f}</td>"
        f"<td>{r['high_conf_error_rate']:.4f}</td><td>{r['n_errors']}</td></tr>"
        for r in study["rows"]
    )
    coef_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in probe["coef"].items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analysis report · Confidence vs. accuracy · Victor E. Birkle III</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap" rel="stylesheet" />
  <style>
    :root {{ --ink:#1a2332; --soft:#3d4a5c; --muted:#5c6b7a; --line:#d4dce6; --paper:#f7f5f0; --accent:#1f5c4a; --ok:#166534; }}
    body {{ margin:0; font-family:"IBM Plex Sans",sans-serif; color:var(--ink); background:var(--paper); line-height:1.55; }}
    .wrap {{ width:min(860px, calc(100% - 2.5rem)); margin:0 auto; padding:2.5rem 0 4rem; }}
    a {{ color:var(--accent); }}
    h1,h2,h3 {{ font-family:Newsreader,Georgia,serif; font-weight:600; letter-spacing:-0.015em; }}
    h1 {{ font-size:clamp(1.7rem,4vw,2.2rem); line-height:1.2; margin:1rem 0 0.5rem; }}
    h2 {{ font-size:1.25rem; margin:2rem 0 0.55rem; }}
    h3 {{ font-size:1.05rem; margin:1.35rem 0 0.4rem; }}
    .status {{ display:inline-block; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; color:var(--ok); background:#e4efe9; border:1px solid #b7d4c6; padding:0.22rem 0.5rem; }}
    .meta {{ color:var(--muted); font-size:0.9rem; margin-bottom:1.25rem; }}
    p,li {{ color:var(--soft); }}
    table {{ width:100%; border-collapse:collapse; font-size:0.84rem; margin:0.7rem 0 1rem; }}
    th,td {{ border:1px solid var(--line); padding:0.4rem 0.5rem; text-align:left; }}
    th {{ background:#efebe3; }}
    .downloads {{ display:flex; flex-wrap:wrap; gap:0.55rem; margin:1rem 0 1.5rem; }}
    .downloads a {{ display:inline-block; border:1px solid var(--accent); padding:0.4rem 0.7rem; text-decoration:none; font-size:0.85rem; }}
    .downloads a:hover {{ background:#e4efe9; }}
    figure {{ margin:1.2rem 0; }}
    figure img {{ width:100%; border:1px solid var(--line); background:#fff; }}
    figcaption {{ font-size:0.8rem; color:var(--muted); margin-top:0.35rem; }}
    .callout {{ border:1px solid var(--line); background:#fff; padding:1rem 1.1rem; margin:1rem 0; }}
  </style>
</head>
<body>
  <article class="wrap">
    <a href="../">← Investigation hub</a>
    <p class="status">Analysis report</p>
    <h1>Confidence vs. accuracy under drift</h1>
    <p class="meta">Victor E. Birkle III · August 2026 · Synthetic intent-classification analysis</p>

    <div class="downloads">
      <a href="./analysis_report.md">Markdown report</a>
      <a href="./pilot_metrics.csv">Pilot CSV</a>
      <a href="./study_v2_metrics.csv">Study v2 CSV</a>
      <a href="./study_v3_metrics.csv">Study v3 CSV</a>
      <a href="./study_v4_metrics.csv">Study v4 CSV</a>
      <a href="./study_v4_ood.csv">Study v4 OOD CSV</a>
      <a href="./comparison_summary.csv">Comparison CSV</a>
      <a href="./study_v2_operational_probe.csv">Ops probe CSV</a>
    </div>

    <div class="callout">
      <strong>Claim.</strong> Corruption type decides the calibration failure mode on this baseline:
      dropout → underconfidence; competing-intent cue injection → overconfidence and high-confidence errors.
    </div>

    <h2>1. Executive summary</h2>
    <table>
      <thead><tr><th>Pass</th><th>Mechanism</th><th>Primary result</th></tr></thead>
      <tbody>
        <tr><td>Pilot</td><td>Token dropout</td><td>Underconfidence</td></tr>
        <tr><td>Study v2</td><td>Cue injection</td><td>Overconfidence + high-conf errors</td></tr>
        <tr><td>Study v2</td><td>ASR swaps</td><td>Mostly underconfidence</td></tr>
        <tr><td>Study v2</td><td>Prior shift</td><td>Weak for high-conf errors</td></tr>
        <tr><td>Study v3</td><td>CLINC150 cue-inject / OOS</td><td>Did not reproduce v2 overconfidence</td></tr>
        <tr><td>Study v4</td><td>Same splits, DistilBERT + MSP AUROC</td><td><a href="../study_v4.html">Encoder column</a></td></tr>
      </tbody>
    </table>

    <h2>2. Pilot results</h2>
    <figure>
      <img src="../figures/pilot_ece_vs_corruption.png" alt="Pilot ECE vs corruption" />
      <figcaption>Pilot: accuracy, mean confidence, ECE vs. dropout level.</figcaption>
    </figure>
    <table>
      <thead><tr><th>Level</th><th>Accuracy</th><th>Mean conf.</th><th>ECE</th><th>Conf − Acc</th></tr></thead>
      <tbody>{pilot_rows}</tbody>
    </table>

    <h2>3. Study v2 results</h2>
    <figure>
      <img src="../figures/study_v2_regimes.png" alt="Study v2 regimes" />
      <figcaption>Study v2 regimes: accuracy, gap, high-confidence error share.</figcaption>
    </figure>
    <p>
      Worst cell: <strong>{worst["regime"]} @ {worst["level"]}</strong> —
      accuracy {worst["accuracy"]:.4f}, conf−acc {worst["gap"]:+.4f},
      P(conf≥0.8|error) {worst["high_conf_error_rate"]:.4f} (n_errors={worst["n_errors"]}).
    </p>
    <table>
      <thead>
        <tr>
          <th>Regime</th><th>Level</th><th>Acc</th><th>Mean conf</th><th>ECE</th>
          <th>Gap</th><th>P(high-conf|err)</th><th>n_err</th>
        </tr>
      </thead>
      <tbody>{study_rows}</tbody>
    </table>
    <figure>
      <img src="../figures/study_v2_reliability.png" alt="Study v2 reliability" />
      <figcaption>Reliability: clean vs. strongest high-confidence-error cell.</figcaption>
    </figure>

    <h2>4. Operational probe</h2>
    <p>AUC <strong>{probe["auc"]}</strong> (n={probe["n"]}; positive rate {probe["positive_rate"]:.3f}). Confidence excluded from features.</p>
    <table>
      <thead><tr><th>Feature</th><th>Coefficient</th></tr></thead>
      <tbody>{coef_rows}</tbody>
    </table>

    <h2>5. Threats to validity</h2>
    <ul>
      <li>Synthetic templates, not production ASR transcripts.</li>
      <li>Cue injection is artificial; validates measurement, not exact production cause.</li>
      <li>Study v3 is linear bag-of-words; study v4 adds one encoder on the same splits.</li>
      <li>Operational probe is descriptive across stacked regime rows — not a strict holdout claim.</li>
    </ul>

    <h2>6. Reproducibility</h2>
    <p>{md_body_note}</p>
    <p>
      Full narrative:
      <a href="./analysis_report.md">analysis_report.md</a> ·
      metrics JSON:
      <a href="../pilot_metrics.json">pilot</a>,
      <a href="../study_v2_metrics.json">study v2</a>,
      <a href="../study_v3_metrics.json">study v3</a>,
      <a href="../study_v4_metrics.json">study v4</a>
    </p>
  </article>
</body>
</html>
"""


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pilot = load_pilot()
    study = load_study()
    study_v3 = load_study_v3()
    study_v4 = load_study_v4()
    paths = export_tables(pilot, study, study_v3, study_v4)
    md = render_markdown(pilot, study, paths)
    md_path = REPORTS / "analysis_report.md"
    md_path.write_text(md)
    html_path = REPORTS / "index.html"
    html_path.write_text(
        render_html("Regenerate with `python research/code/generate_reports.py`.")
    )
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    if study_v4 and "distilbert" in study_v4.get("models", {}):
        from write_study_v4_pages import write_pages

        h, m = write_pages(study_v4)
        print(f"Wrote {h}")
        print(f"Wrote {m}")
    for p in paths.values():
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
