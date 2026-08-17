"""
Study v2: can we produce high-confidence errors under production-shaped shifts?

Builds on the pilot (token dropout → underconfidence). Here we try three
label-preserving mechanisms that can activate the wrong class strongly:

1. asr_swap      — confuse intent keywords with near-neighbor tokens that
                   belong to other intents (ASR/homophone proxy)
2. cue_inject    — splice a strong competing-intent keyword into the utterance
                   while keeping the true label
3. prior_shift   — train balanced; evaluate on a skewed test prior, optionally
                   with mild cue injection

Reports accuracy, mean confidence, ECE, confidence−accuracy gap, and the
share of errors that are high-confidence (conf ≥ 0.8). Also fits a tiny
operational-feature probe: can length / corruption proxy / regime flag
predict |confidence − correctness|?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Reuse corpus builders from the pilot module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_confidence_drift import (  # noqa: E402
    ECE_BINS,
    FIG_DIR,
    INTENTS,
    N_PER_CLASS,
    SEED,
    TEMPLATES,
    build_corpus,
    expected_calibration_error,
    make_utterance,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
HIGH_CONF = 0.8

# Intent-critical tokens → confusable tokens that lean toward another intent.
ASR_SWAPS: dict[str, list[str]] = {
    "balance": ["ballet", "ballot", "billing"],
    "cancel": ["candle", "council", "channel"],
    "hours": ["ours", "houses", "powers"],
    "password": ["passport", "passed", "pastor"],
    "refund": ["refuse", "refill", "reform"],
    "shipping": ["shopping", "slipping", "skipping"],
    "support": ["sport", "report", "supposed"],
    "upgrade": ["update", "upbraid", "upstate"],
    "account": ["amount", "a count", "accent"],
    "order": ["odor", "older", "border"],
    "please": ["pleas", "fleece", "police"],
}

# Strong class cues used for competing-intent injection
CLASS_CUES: dict[str, list[str]] = {
    "balance": ["balance", "funds", "checking"],
    "cancel": ["cancel", "unsubscribe", "terminate"],
    "hours": ["hours", "open", "closing"],
    "password": ["password", "login", "reset"],
    "refund": ["refund", "moneyback", "reimburse"],
    "shipping": ["shipping", "tracking", "delivery"],
    "support": ["support", "agent", "representative"],
    "upgrade": ["upgrade", "premium", "tier"],
}


def fit_pipeline(x_train: list[str], y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def asr_swap_text(text: str, level: float, rng: np.random.Generator) -> str:
    if level <= 0:
        return text
    tokens = text.split()
    out = []
    for t in tokens:
        key = t.lower()
        if key in ASR_SWAPS and rng.random() < level:
            choices = ASR_SWAPS[key]
            out.append(choices[int(rng.integers(0, len(choices)))])
        else:
            out.append(t)
    return " ".join(out)


def cue_inject_text(text: str, true_idx: int, level: float, rng: np.random.Generator) -> str:
    if level <= 0:
        return text
    tokens = text.split()
    true_name = INTENTS[true_idx]
    true_cues = set(CLASS_CUES[true_name])
    # Strip true-class cues with probability ∝ level so rival keywords can dominate
    if rng.random() < level:
        stripped = [t for t in tokens if t.lower() not in true_cues]
        if stripped:
            tokens = stripped

    others = [i for i in range(len(INTENTS)) if i != true_idx]
    rival = others[int(rng.integers(0, len(others)))]
    cues = CLASS_CUES[INTENTS[rival]]
    n_inject = 1 + int(level >= 0.3) + int(level >= 0.6)
    for _ in range(n_inject):
        cue = cues[int(rng.integers(0, len(cues)))]
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), cue)
    return " ".join(tokens)


def resample_prior(
    texts: list[str], labels: np.ndarray, level: float, rng: np.random.Generator
) -> tuple[list[str], np.ndarray]:
    """Skew test distribution toward a few classes as level rises."""
    if level <= 0:
        return texts, labels
    # Weight: boost first 3 classes, suppress the rest, strength∝level
    weights = np.ones(len(INTENTS))
    weights[:3] *= 1.0 + 4.0 * level
    weights[3:] *= max(0.05, 1.0 - 0.9 * level)
    weights /= weights.sum()

    n = len(texts)
    by_class: dict[int, list[int]] = {i: [] for i in range(len(INTENTS))}
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    chosen: list[int] = []
    targets = rng.choice(len(INTENTS), size=n, p=weights)
    for y in targets:
        pool = by_class[int(y)]
        if not pool:
            # fallback any
            chosen.append(int(rng.integers(0, n)))
        else:
            chosen.append(pool[int(rng.integers(0, len(pool)))])
    return [texts[i] for i in chosen], labels[np.array(chosen)]


def transform_batch(
    texts: list[str],
    labels: np.ndarray,
    regime: str,
    level: float,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns corrupted texts, labels, and a scalar corruption proxy per row."""
    out_texts: list[str] = []
    proxies: list[float] = []
    y = labels.copy()

    if regime == "prior_shift":
        texts, y = resample_prior(texts, y, level, rng)

    for t, label in zip(texts, y):
        if regime == "asr_swap":
            c = asr_swap_text(t, level, rng)
        elif regime == "cue_inject":
            c = cue_inject_text(t, int(label), level, rng)
        elif regime == "prior_shift":
            # Mild competing cues on top of prior skew — otherwise prior alone
            # mostly changes prevalence, not per-example confidence errors.
            c = cue_inject_text(t, int(label), level * 0.5, rng)
        else:
            raise ValueError(regime)
        out_texts.append(c)
        # proxy: relative token edit intensity
        proxies.append(abs(len(c.split()) - len(t.split())) / max(1, len(t.split())) + (0.0 if c == t else level))
    return out_texts, y, np.array(proxies, dtype=float)


def summarize(
    pipe: Pipeline,
    texts: list[str],
    y_true: np.ndarray,
    proxies: np.ndarray,
    regime: str,
    level: float,
) -> dict:
    proba = pipe.predict_proba(texts)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = pred == y_true
    ece, bin_acc, bin_conf, bin_count = expected_calibration_error(conf, correct.astype(float))
    errors = ~correct
    n_err = int(errors.sum())
    high_conf_err = float(((errors) & (conf >= HIGH_CONF)).sum() / max(1, n_err)) if n_err else 0.0
    return {
        "regime": regime,
        "level": level,
        "accuracy": float(accuracy_score(y_true, pred)),
        "mean_confidence": float(conf.mean()),
        "ece": float(ece),
        "gap": float(conf.mean() - accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "n_errors": n_err,
        "high_conf_error_rate": high_conf_err,  # among errors
        "mean_proxy": float(proxies.mean()),
        "mean_length": float(np.mean([len(t.split()) for t in texts])),
        "confidences": conf,
        "correct": correct.astype(float),
        "proxies": proxies,
        "lengths": np.array([len(t.split()) for t in texts], dtype=float),
        "reliability": {
            "bin_acc": bin_acc.tolist(),
            "bin_conf": bin_conf.tolist(),
            "bin_count": bin_count.tolist(),
        },
    }


def plot_regimes(rows: list[dict], path: Path) -> None:
    regimes = sorted({r["regime"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
    metrics = [
        ("accuracy", "Accuracy"),
        ("gap", "Confidence − accuracy"),
        ("high_conf_error_rate", f"P(conf≥{HIGH_CONF} | error)"),
    ]
    colors = {"asr_swap": "#1f5c4a", "cue_inject": "#9a3412", "prior_shift": "#1d4ed8"}
    for ax, (key, title) in zip(axes, metrics):
        for regime in regimes:
            series = [r for r in rows if r["regime"] == regime]
            series = sorted(series, key=lambda r: r["level"])
            ax.plot(
                [r["level"] for r in series],
                [r[key] for r in series],
                "o-",
                color=colors[regime],
                label=regime,
                linewidth=2,
            )
        ax.set_xlabel("Shift strength")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        if key != "gap":
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Study v2: seeking high-confidence errors", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_pair(clean: dict, worst: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, row, title in [
        (axes[0], clean, f"Clean · ECE={clean['ece']:.3f}"),
        (axes[1], worst, f"{worst['regime']}@{worst['level']:.2f} · ECE={worst['ece']:.3f}"),
    ]:
        centers = np.linspace(0.05, 0.95, ECE_BINS)
        acc = np.array(row["reliability"]["bin_acc"])
        conf = np.array(row["reliability"]["bin_conf"])
        count = np.array(row["reliability"]["bin_count"])
        mask = count > 0
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
        ax.bar(centers[mask], acc[mask], width=0.08, color="#1f5c4a", alpha=0.55)
        ax.plot(conf[mask], acc[mask], "o", color="#9a3412")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Confidence")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Reliability: clean vs. strongest high-conf-error regime", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def operational_probe(rows: list[dict]) -> dict:
    """Can operational features alone (no labels, no raw confidence) flag mismatch risk?"""
    xs = []
    ys = []
    for r in rows:
        if r["level"] < 0.15:
            continue
        conf = r["confidences"]
        correct = r["correct"]
        target = np.abs(conf - correct)
        label = (target >= 0.5).astype(int)
        length = r["lengths"]
        proxy = r["proxies"]
        regime_code = {"asr_swap": 0, "cue_inject": 1, "prior_shift": 2}[r["regime"]]
        for i in range(len(conf)):
            # Deliberately exclude confidence — the portfolio question is whether
            # operational signals can flag problems without privileged labels.
            xs.append([length[i], proxy[i], regime_code, r["level"]])
            ys.append(int(label[i]))
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc": None, "note": "degenerate labels — probe skipped"}
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(x, y)
    proba = clf.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, proba))
    coef = {
        name: float(c)
        for name, c in zip(
            ["length", "corruption_proxy", "regime_code", "level"],
            clf.coef_[0],
        )
    }
    return {
        "auc": round(auc, 4),
        "coef": coef,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "features": "length, corruption_proxy, regime_code, level (confidence excluded)",
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    texts, labels = build_corpus(rng)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    pipe = fit_pipeline(x_train, y_train)

    regimes = ["asr_swap", "cue_inject", "prior_shift"]
    rows: list[dict] = []
    for regime in regimes:
        for level in LEVELS:
            # fresh RNG stream per cell for reproducibility across edits
            cell_rng = np.random.default_rng(SEED + hash((regime, level)) % 10_000)
            corrupted, y_eval, proxies = transform_batch(x_test, y_test, regime, level, cell_rng)
            rows.append(summarize(pipe, corrupted, y_eval, proxies, regime, level))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = FIG_DIR / "study_v2_regimes.png"
    fig2 = FIG_DIR / "study_v2_reliability.png"
    plot_regimes(rows, fig1)

    clean = next(r for r in rows if r["regime"] == "cue_inject" and r["level"] == 0.0)
    # Pick the cell with highest high-conf error rate among non-trivial shifts
    candidates = [r for r in rows if r["level"] >= 0.3 and r["n_errors"] >= 10]
    worst = max(candidates, key=lambda r: r["high_conf_error_rate"])
    plot_reliability_pair(clean, worst, fig2)

    probe = operational_probe(rows)

    summary = {
        "seed": SEED,
        "question": "Which label-preserving shifts produce high-confidence errors?",
        "model": "CountVectorizer(1-2gram) + LogisticRegression",
        "regimes": regimes,
        "high_conf_threshold": HIGH_CONF,
        "worst_cell": {
            "regime": worst["regime"],
            "level": worst["level"],
            "accuracy": round(worst["accuracy"], 4),
            "mean_confidence": round(worst["mean_confidence"], 4),
            "ece": round(worst["ece"], 4),
            "gap": round(worst["gap"], 4),
            "high_conf_error_rate": round(worst["high_conf_error_rate"], 4),
            "n_errors": worst["n_errors"],
        },
        "operational_probe": probe,
        "rows": [
            {
                "regime": r["regime"],
                "level": r["level"],
                "accuracy": round(r["accuracy"], 4),
                "mean_confidence": round(r["mean_confidence"], 4),
                "ece": round(r["ece"], 4),
                "gap_conf_minus_acc": round(r["gap"], 4),
                "high_conf_error_rate": round(r["high_conf_error_rate"], 4),
                "n_errors": r["n_errors"],
            }
            for r in rows
        ],
        "figures": [
            "research/figures/study_v2_regimes.png",
            "research/figures/study_v2_reliability.png",
        ],
    }
    out_json = OUT_DIR / "study_v2_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "study_v2_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {fig1}")
    print(f"Wrote {fig2}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
