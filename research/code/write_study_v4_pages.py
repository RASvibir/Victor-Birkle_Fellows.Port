"""
Render study v4 HTML/Markdown from study_v4_metrics.json.

Numbers come from the JSON only — this file does not invent metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "study_v4_metrics.json"


def _cue(model: dict, level: float) -> dict:
    return next(r for r in model["rows"] if r["regime"] == "cue_inject" and abs(r["level"] - level) < 1e-9)


def _oos(model: dict) -> dict:
    return next(r for r in model["rows"] if r["regime"] == "oos")


def _id(model: dict) -> dict:
    return next(r for r in model["rows"] if r["regime"] == "in_domain")


def _fmt(x: float, signed: bool = False) -> str:
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def headline(metrics: dict) -> tuple[str, str]:
    """Return (short status line, lead paragraph)."""
    enc = metrics["models"]["distilbert"]
    bow = metrics["models"]["bow_logistic"]
    enc_cue = _cue(enc, 0.6)
    enc_oos = _oos(enc)
    bow_ood = bow["ood_detection"]["auroc_id_vs_oos"]
    enc_ood = enc["ood_detection"]["auroc_id_vs_oos"]
    reproduced = enc_cue["gap_conf_minus_acc"] > 0.05 and enc_cue["high_conf_error_rate"] >= 0.20
    oos_hot = enc_oos["mean_confidence"] >= 0.60 and enc_oos["high_conf_error_rate"] >= 0.20
    if reproduced:
        status = "Study v4 complete · encoder reproduced high-confidence errors under cue-inject"
        lead = (
            f"On the same CLINC150 splits, fine-tuned DistilBERT <strong>does</strong> show the "
            f"v2 overconfidence pattern under cue injection (conf−acc {_fmt(enc_cue['gap_conf_minus_acc'], True)}; "
            f"P(conf≥0.8|error) {_fmt(enc_cue['high_conf_error_rate'])}). "
            f"The study v3 null was a model-class result, not a dataset result. "
            f"MSP still separates in-domain from OOS (encoder AUROC {enc_ood:.3f}; BoW {bow_ood:.3f})."
        )
    elif oos_hot:
        status = "Study v4 complete · encoder is confident on OOS errors"
        lead = (
            f"Fine-tuned DistilBERT is <strong>confidently wrong on OOS</strong> "
            f"(mean conf {_fmt(enc_oos['mean_confidence'])}; "
            f"P(conf≥0.8|error) {_fmt(enc_oos['high_conf_error_rate'])}), "
            f"unlike the bag-of-words model (mean conf {_fmt(_oos(bow)['mean_confidence'])}). "
            f"Cue-inject gap is {_fmt(enc_cue['gap_conf_minus_acc'], True)}. "
            f"MSP AUROC ID vs OOS: encoder {enc_ood:.3f}, BoW {bow_ood:.3f}."
        )
    else:
        status = "Study v4 complete · encoder also did not reproduce synthetic overconfidence"
        lead = (
            f"A fine-tuned encoder on the same CLINC150 splits still does <strong>not</strong> "
            f"reproduce study v2’s overconfidence. Cue-inject@0.6: conf−acc "
            f"{_fmt(enc_cue['gap_conf_minus_acc'], True)}, "
            f"P(conf≥0.8|error) {_fmt(enc_cue['high_conf_error_rate'])}. "
            f"OOS mean confidence {_fmt(enc_oos['mean_confidence'])} "
            f"(P(high-conf|error) {_fmt(enc_oos['high_conf_error_rate'])}). "
            f"MSP already separates in-domain from OOS "
            f"(BoW AUROC {bow_ood:.3f}; encoder {enc_ood:.3f}). "
            f"The synthetic pathology needs a corruption that makes the model confidently wrong, "
            f"not only a stronger model or an unseen-intent split."
        )
    return status, lead


def _rows_table(model: dict) -> str:
    lines = []
    for r in model["rows"]:
        name = r["regime"] if r["regime"] != "cue_inject" else f"cue_inject @ {r['level']}"
        if r["regime"] == "oos":
            name = "OOS (errors by construction)"
        lines.append(
            f"<tr><td>{name}</td><td>{r['n']}</td><td>{r['accuracy']:.3f}</td>"
            f"<td>{r['mean_confidence']:.3f}</td><td>{r['ece']:.3f}</td>"
            f"<td>{r['gap_conf_minus_acc']:+.3f}</td>"
            f"<td>{r['high_conf_error_rate']:.3f}</td></tr>"
        )
    return "\n".join(lines)


def render_html(metrics: dict) -> str:
    status, lead = headline(metrics)
    bow = metrics["models"]["bow_logistic"]
    enc = metrics["models"]["distilbert"]
    enc_meta = metrics.get("encoder") or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Study v4: encoder vs BoW on CLINC150 · Victor E. Birkle III</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --ink: #1a2332;
      --ink-soft: #3d4a5c;
      --muted: #5c6b7a;
      --line: #d4dce6;
      --paper: #f7f5f0;
      --accent: #1f5c4a;
      --ok: #166534;
    }}
    body {{ margin: 0; font-family: "IBM Plex Sans", sans-serif; color: var(--ink); background: var(--paper); line-height: 1.6; }}
    .wrap {{ width: min(720px, calc(100% - 2.5rem)); margin: 0 auto; padding: 2.5rem 0 4rem; }}
    a {{ color: var(--accent); }}
    .back {{ font-size: 0.88rem; text-decoration: none; }}
    h1 {{ font-family: Newsreader, Georgia, serif; font-weight: 600; font-size: clamp(1.55rem, 3.8vw, 2rem); letter-spacing: -0.02em; line-height: 1.2; margin: 1.1rem 0 0.5rem; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .status {{ display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ok); background: #e4efe9; border: 1px solid #b7d4c6; padding: 0.22rem 0.5rem; margin-bottom: 0.75rem; }}
    h2 {{ font-family: Newsreader, Georgia, serif; font-size: 1.2rem; margin: 1.85rem 0 0.5rem; font-weight: 600; }}
    p, li {{ color: var(--ink-soft); font-size: 0.98rem; }}
    ul {{ padding-left: 1.2rem; }}
    li {{ margin-bottom: 0.35rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 0.75rem 0 1rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem 0.55rem; text-align: left; }}
    th {{ background: #efebe3; }}
    figure {{ margin: 1.25rem 0; }}
    figure img {{ width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }}
    figcaption {{ font-size: 0.82rem; color: var(--muted); margin-top: 0.4rem; }}
    pre {{ background: #efebe3; border: 1px solid var(--line); padding: 0.85rem 1rem; overflow-x: auto; font-size: 0.8rem; }}
    hr {{ border: 0; border-top: 1px solid var(--line); margin: 1.75rem 0; }}
  </style>
</head>
<body>
  <article class="wrap">
    <a class="back" href="./">← Investigation hub</a>
    <p class="status">{status}</p>
    <h1>Encoder check — same CLINC150 splits</h1>
    <p class="meta">
      Victor E. Birkle III · August 2026 ·
      <a href="./code/study_encoder_clinc150.py">code</a> ·
      <a href="./study_v4_metrics.json">metrics</a> ·
      <a href="./study_v3.html">study v3</a>
    </p>

    <p>{lead}</p>

    <h2>Setup</h2>
    <table>
      <thead><tr><th>Piece</th><th>Choice</th></tr></thead>
      <tbody>
        <tr><td>Data</td><td><code>{metrics["dataset"]}</code> · fingerprint <code>{metrics["dataset_fingerprint"]}</code> · {metrics["license"]}</td></tr>
        <tr><td>Protocol</td><td>{metrics["protocol"]}</td></tr>
        <tr><td>BoW column</td><td>{bow["spec"]}</td></tr>
        <tr><td>Encoder column</td><td>{enc["spec"]}</td></tr>
        <tr><td>Device / train</td><td>{enc_meta.get("device", "?")} · {enc_meta.get("epochs", "?")} epoch(s) · batch {enc_meta.get("batch_size", "?")} · max length {enc_meta.get("max_length", "?")}</td></tr>
        <tr><td>OOS scoring</td><td>No OOS class. Every OOS example is an error. Report confidence and MSP AUROC, not dummy accuracy.</td></tr>
      </tbody>
    </table>
    <pre>uv venv .venv-encoder --python 3.12
uv pip install -r research/code/requirements-encoder.txt
python research/code/study_encoder_clinc150.py</pre>

    <h2>Headline figure</h2>
    <figure>
      <img src="./figures/study_v4_compare.png" alt="BoW vs DistilBERT accuracy, confidence-accuracy gap, and high-confidence error rate" />
      <figcaption>Same shifted text for both models. Seed {metrics["seed"]}.</figcaption>
    </figure>

    <h2>OOD detection (MSP)</h2>
    <p>
      Max softmax probability as an in-distribution score
      (Hendrycks &amp; Gimpel 2017). Label 1 = in-domain test, 0 = OOS.
    </p>
    <table>
      <thead><tr><th>Model</th><th>ID mean conf</th><th>OOS mean conf</th><th>MSP AUROC</th></tr></thead>
      <tbody>
        <tr><td>BoW logistic</td><td>{bow["ood_detection"]["id_mean_conf"]:.3f}</td><td>{bow["ood_detection"]["oos_mean_conf"]:.3f}</td><td>{bow["ood_detection"]["auroc_id_vs_oos"]:.3f}</td></tr>
        <tr><td>DistilBERT</td><td>{enc["ood_detection"]["id_mean_conf"]:.3f}</td><td>{enc["ood_detection"]["oos_mean_conf"]:.3f}</td><td>{enc["ood_detection"]["auroc_id_vs_oos"]:.3f}</td></tr>
      </tbody>
    </table>
    <figure>
      <img src="./figures/study_v4_msp_ood.png" alt="Max-softmax histograms for in-domain vs OOS, BoW and DistilBERT" />
      <figcaption>MSP distributions, in-domain vs OOS.</figcaption>
    </figure>

    <h2>BoW logistic (replication of study v3)</h2>
    <table>
      <thead>
        <tr><th>Condition</th><th>n</th><th>Acc</th><th>Mean conf</th><th>ECE</th><th>Conf − Acc</th><th>P(conf≥0.8|err)</th></tr>
      </thead>
      <tbody>
{_rows_table(bow)}
      </tbody>
    </table>

    <h2>DistilBERT</h2>
    <table>
      <thead>
        <tr><th>Condition</th><th>n</th><th>Acc</th><th>Mean conf</th><th>ECE</th><th>Conf − Acc</th><th>P(conf≥0.8|err)</th></tr>
      </thead>
      <tbody>
{_rows_table(enc)}
      </tbody>
    </table>
    <p>
      In-domain DistilBERT is more accurate than BoW ({_id(enc)["accuracy"]:.3f} vs {_id(bow)["accuracy"]:.3f})
      and more underconfident (gap {_fmt(_id(enc)["gap_conf_minus_acc"], True)} vs {_fmt(_id(bow)["gap_conf_minus_acc"], True)}).
      Cue-inject@0.6 edges into a tiny positive gap ({_fmt(_cue(enc, 0.6)["gap_conf_minus_acc"], True)}),
      but P(conf≥0.8|error) is {_fmt(_cue(enc, 0.6)["high_conf_error_rate"])} — not study v2’s 0.43.
      BoW metrics in this file match study v3 bit-for-bit; the datasets fingerprint string can differ by library version.
    </p>
    <figure>
      <img src="./figures/study_v4_reliability.png" alt="DistilBERT reliability diagrams" />
      <figcaption>Encoder reliability: in-domain vs the shift with the highest high-conf error share.</figcaption>
    </figure>

    <h2>How to read this against study v2</h2>
    <p>
      Synthetic cue-inject@0.6 was conf−acc <strong>+0.20</strong> and P(high-conf|error) <strong>0.43</strong>.
      That is the existence proof that this measurement loop can see overconfidence.
      Study v3 asked whether the same lexical cue injection, plus genuine OOS, produces that
      pattern on real CLINC150 text with the same linear model. It did not.
      This pass asks whether that null was “the linear model cannot be overconfident”
      (Guo et al. 2017: modern nets are the usual overconfidence case).
    </p>
    <ul>
      <li>CLINC150: Larson et al. 2019, EMNLP.</li>
      <li>MSP OOD baseline: Hendrycks &amp; Gimpel 2017.</li>
      <li>Neural overconfidence: Guo et al. 2017.</li>
    </ul>

    <h2>Limits</h2>
    <ul>
      <li>One encoder, two epochs, no temperature scaling or other post-hoc calibration.</li>
      <li>Cue injection is still a lexical attack, not ASR lattices.</li>
      <li>OOS accuracy is zero by scoring choice. Interpret confidence and MSP AUROC.</li>
      <li>Ops probes remain in-sample descriptive AUCs, not holdout detectors.</li>
    </ul>

    <hr />
    <p>
      <a href="./">Investigation hub</a> ·
      <a href="./reports/">Reports</a> ·
      <a href="./study_v3.html">Study v3</a> ·
      <a href="./study_v2.html">Study v2</a>
    </p>
  </article>
  <script>
    if ("serviceWorker" in navigator) {{
      navigator.serviceWorker.getRegistrations().then(function (regs) {{
        regs.forEach(function (r) {{ r.unregister(); }});
      }});
    }}
  </script>
</body>
</html>
"""


def render_md(metrics: dict) -> str:
    status, lead = headline(metrics)
    bow = metrics["models"]["bow_logistic"]
    enc = metrics["models"]["distilbert"]
    plain_lead = lead.replace("<strong>", "**").replace("</strong>", "**")

    def md_rows(model: dict) -> str:
        lines = ["| Condition | n | Acc | Mean conf | ECE | Conf − Acc | P(conf≥0.8\\|err) |", "|---|--:|----:|----------:|----:|-----------:|------------------:|"]
        for r in model["rows"]:
            name = r["regime"] if r["regime"] != "cue_inject" else f"cue_inject @ {r['level']}"
            if r["regime"] == "oos":
                name = "OOS"
            lines.append(
                f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['mean_confidence']:.3f} | "
                f"{r['ece']:.3f} | {r['gap_conf_minus_acc']:+.3f} | {r['high_conf_error_rate']:.3f} |"
            )
        return "\n".join(lines)

    return f"""# Study v4 — encoder vs BoW on CLINC150

**Status:** {status.replace('Study v4 complete · ', '')}
**Author:** Victor E. Birkle III
**Date:** August 2026
**Code:** [`research/code/study_encoder_clinc150.py`](./code/study_encoder_clinc150.py)
**Metrics:** [`research/study_v4_metrics.json`](./study_v4_metrics.json)
**Data:** `{metrics["dataset"]}` ({metrics["license"]}), fingerprint `{metrics["dataset_fingerprint"]}`

## Headline

{plain_lead}

## MSP OOD (ID vs OOS)

| Model | ID mean conf | OOS mean conf | MSP AUROC |
|-------|-------------:|--------------:|----------:|
| BoW logistic | {bow["ood_detection"]["id_mean_conf"]:.3f} | {bow["ood_detection"]["oos_mean_conf"]:.3f} | {bow["ood_detection"]["auroc_id_vs_oos"]:.3f} |
| DistilBERT | {enc["ood_detection"]["id_mean_conf"]:.3f} | {enc["ood_detection"]["oos_mean_conf"]:.3f} | {enc["ood_detection"]["auroc_id_vs_oos"]:.3f} |

## BoW logistic

{md_rows(bow)}

## DistilBERT

{md_rows(enc)}

In-domain DistilBERT is more accurate than BoW ({_id(enc)["accuracy"]:.3f} vs {_id(bow)["accuracy"]:.3f}) and more underconfident (gap {_fmt(_id(enc)["gap_conf_minus_acc"], True)}). Cue-inject@0.6 has a tiny positive gap ({_fmt(_cue(enc, 0.6)["gap_conf_minus_acc"], True)}) but P(high-conf|error) {_fmt(_cue(enc, 0.6)["high_conf_error_rate"])} — not v2’s 0.43. BoW metrics match study v3 bit-for-bit.

## Run

```bash
uv venv .venv-encoder --python 3.12
uv pip install -r research/code/requirements-encoder.txt
python research/code/study_encoder_clinc150.py
python research/code/generate_reports.py
```
"""


def write_pages(metrics: dict | None = None) -> tuple[Path, Path]:
    if metrics is None:
        metrics = json.loads(JSON_PATH.read_text())
    html_path = ROOT / "study_v4.html"
    md_path = ROOT / "study_v4.md"
    html_path.write_text(render_html(metrics))
    md_path.write_text(render_md(metrics))
    return html_path, md_path


if __name__ == "__main__":
    h, m = write_pages()
    print(f"Wrote {h}")
    print(f"Wrote {m}")
