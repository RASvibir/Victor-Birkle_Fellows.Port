"""
Study v3: confidence vs. accuracy on CLINC150 (real utterances + OOS).

Train a bag-of-words logistic baseline on in-domain CLINC150 only, then evaluate:

1. in_domain — clean test (labels 0–149)
2. cue_inject — Study v2 mechanism on real in-domain test utterances
3. oos — built-in out-of-scope test utterances (label is None in DeepPavlov/clinc150);
         the model has no OOS class, so every OOS example is an error. The question
         is how confident those errors are.

Does not invent numbers. Writes JSON + figures for the writeup to consume.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_confidence_drift import ECE_BINS, FIG_DIR, SEED, expected_calibration_error  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
HIGH_CONF = 0.8
CUE_LEVELS = [0.0, 0.3, 0.6]
DATASET_ID = "DeepPavlov/clinc150"
STOP = {
    "a", "an", "the", "to", "of", "and", "or", "for", "in", "on", "is", "it",
    "my", "me", "i", "you", "please", "can", "do", "what", "how", "where",
    "when", "with", "this", "that", "be", "at", "from", "are", "was",
}


def load_clinc():
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    revision = str(ds["train"]._fingerprint)

    def split_rows(split: str):
        in_utts, in_y, oos_utts = [], [], []
        for row in ds[split]:
            utt = row["utterance"]
            lab = row["label"]
            if lab is None:
                oos_utts.append(utt)
            else:
                in_utts.append(utt)
                in_y.append(int(lab))
        return in_utts, np.array(in_y, dtype=int), oos_utts

    train_x, train_y, train_oos = split_rows("train")
    val_x, val_y, val_oos = split_rows("validation")
    test_x, test_y, test_oos = split_rows("test")
    return {
        "revision": revision,
        "n_train_in": len(train_x),
        "n_train_oos": len(train_oos),
        "n_val_in": len(val_x),
        "n_val_oos": len(val_oos),
        "n_test_in": len(test_x),
        "n_test_oos": len(test_oos),
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "test_oos": test_oos,
        "val_x": val_x,
        "val_y": val_y,
    }


def class_cues(texts: list[str], labels: np.ndarray, top_k: int = 8) -> dict[int, list[str]]:
    df = Counter()
    per: dict[int, Counter] = defaultdict(Counter)
    for t, y in zip(texts, labels):
        toks = {w.lower() for w in t.split() if w.lower() not in STOP and w.isalpha()}
        df.update(toks)
        per[int(y)].update(toks)
    n_docs = max(1, len(texts))
    cues: dict[int, list[str]] = {}
    for y, ctr in per.items():
        scored = []
        for w, c in ctr.items():
            # class frequency / document frequency — prefer distinctive tokens
            scored.append((c / max(1, df[w]) * np.log1p(c), w))
        scored.sort(reverse=True)
        cues[int(y)] = [w for _, w in scored[:top_k]]
    return cues


def cue_inject(text: str, true_y: int, level: float, cues: dict[int, list[str]], rng: np.random.Generator, n_classes: int) -> str:
    if level <= 0:
        return text
    tokens = text.split()
    own = set(cues.get(true_y, []))
    if rng.random() < level:
        stripped = [t for t in tokens if t.lower() not in own]
        if stripped:
            tokens = stripped
    others = [i for i in range(n_classes) if i != true_y]
    rival = int(others[int(rng.integers(0, len(others)))])
    rival_cues = cues.get(rival) or ["please"]
    n_inject = 1 + int(level >= 0.3) + int(level >= 0.6)
    for _ in range(n_inject):
        cue = rival_cues[int(rng.integers(0, len(rival_cues)))]
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), cue)
    return " ".join(tokens)


def summarize(pipe: Pipeline, texts: list[str], y_true: np.ndarray, regime: str, level: float, train_vocab: set[str]) -> dict:
    proba = pipe.predict_proba(texts)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = pred == y_true
    ece, bin_acc, bin_conf, bin_count = expected_calibration_error(conf, correct.astype(float))
    n_err = int((~correct).sum())
    high = float(((~correct) & (conf >= HIGH_CONF)).sum() / max(1, n_err)) if n_err else 0.0
    oov = np.array(
        [
            sum(1 for w in t.lower().split() if w not in train_vocab) / max(1, len(t.split()))
            for t in texts
        ],
        dtype=float,
    )
    lengths = np.array([len(t.split()) for t in texts], dtype=float)
    return {
        "regime": regime,
        "level": level,
        "accuracy": float(accuracy_score(y_true, pred)),
        "mean_confidence": float(conf.mean()),
        "ece": float(ece),
        "gap": float(conf.mean() - accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "n_errors": n_err,
        "high_conf_error_rate": high,
        "confidences": conf,
        "correct": correct.astype(float),
        "lengths": lengths,
        "oov_rate": oov,
        "reliability": {
            "bin_acc": bin_acc.tolist(),
            "bin_conf": bin_conf.tolist(),
            "bin_count": bin_count.tolist(),
        },
    }


def plot_conditions(rows: list[dict], path: Path) -> None:
    labels = [f"{r['regime']}" + (f"@{r['level']}" if r["regime"] == "cue_inject" else "") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    specs = [
        ("accuracy", "Accuracy", (0, 1.05)),
        ("gap", "Confidence − accuracy", None),
        ("high_conf_error_rate", f"P(conf≥{HIGH_CONF} | error)", (0, 1.05)),
    ]
    x = np.arange(len(rows))
    colors = ["#1f5c4a" if r["regime"] == "in_domain" else "#9a3412" if r["regime"] == "cue_inject" else "#1d4ed8" for r in rows]
    for ax, (key, title, ylim) in zip(axes, specs):
        ax.bar(x, [r[key] for r in rows], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    fig.suptitle("Study v3 (CLINC150): in-domain vs cue-inject vs OOS", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(clean: dict, other: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, row in [(axes[0], clean), (axes[1], other)]:
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
        ax.set_title(f"{row['regime']} · ECE={row['ece']:.3f} · gap={row['gap']:+.3f}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Study v3 reliability: in-domain vs strongest shift", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def operational_probe(rows: list[dict]) -> dict:
    xs, ys = [], []
    code = {"in_domain": 0, "cue_inject": 1, "oos": 2}
    for r in rows:
        if r["regime"] == "in_domain":
            continue
        target = (np.abs(r["confidences"] - r["correct"]) >= 0.5).astype(int)
        for i in range(len(target)):
            xs.append([r["lengths"][i], r["oov_rate"][i], code[r["regime"]], r["level"]])
            ys.append(int(target[i]))
    x = np.array(xs, float)
    y = np.array(ys, int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc": None, "note": "degenerate labels — not completed", "n": int(len(y))}
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(x, y)
    auc = float(roc_auc_score(y, clf.predict_proba(x)[:, 1]))
    return {
        "auc": round(auc, 4),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "features": "length, oov_rate vs train vocab, regime_code, cue_level (confidence excluded)",
        "coef": {
            "length": float(clf.coef_[0][0]),
            "oov_rate": float(clf.coef_[0][1]),
            "regime_code": float(clf.coef_[0][2]),
            "cue_level": float(clf.coef_[0][3]),
        },
        "note": "in-sample descriptive probe, not a holdout claim",
    }


def public_row(r: dict) -> dict:
    return {
        "regime": r["regime"],
        "level": r["level"],
        "accuracy": round(r["accuracy"], 4),
        "mean_confidence": round(r["mean_confidence"], 4),
        "ece": round(r["ece"], 4),
        "gap_conf_minus_acc": round(r["gap"], 4),
        "high_conf_error_rate": round(r["high_conf_error_rate"], 4),
        "n": r["n"],
        "n_errors": r["n_errors"],
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_clinc()
    n_classes = int(data["train_y"].max()) + 1
    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
        ]
    )
    pipe.fit(data["train_x"], data["train_y"])
    cues = class_cues(data["train_x"], data["train_y"])
    train_vocab = {w.lower() for t in data["train_x"] for w in t.split()}

    rows = []
    rows.append(summarize(pipe, data["test_x"], data["test_y"], "in_domain", 0.0, train_vocab))
    for level in CUE_LEVELS:
        if level == 0.0:
            continue
        injected = [
            cue_inject(t, int(y), level, cues, rng, n_classes)
            for t, y in zip(data["test_x"], data["test_y"])
        ]
        rows.append(summarize(pipe, injected, data["test_y"], "cue_inject", level, train_vocab))

    oos_y = np.full(len(data["test_oos"]), -1, dtype=int)
    rows.append(summarize(pipe, data["test_oos"], oos_y, "oos", 0.0, train_vocab))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1 = FIG_DIR / "study_v3_conditions.png"
    fig2 = FIG_DIR / "study_v3_reliability.png"
    plot_conditions(rows, fig1)
    shifted = [r for r in rows if r["regime"] != "in_domain"]
    worst = max(shifted, key=lambda r: r["high_conf_error_rate"])
    plot_reliability(rows[0], worst, fig2)

    probe = operational_probe(rows)
    summary = {
        "seed": SEED,
        "dataset": DATASET_ID,
        "dataset_fingerprint": data["revision"],
        "license": "CC BY 4.0 (CLINC150 / DeepPavlov HF release)",
        "question": "On real CLINC150 utterances, does cue-inject and/or OOS produce high-confidence errors?",
        "model": "CountVectorizer(1-2gram) + LogisticRegression (same class as pilot/v2)",
        "train": {"n_in_domain": data["n_train_in"], "n_oos_held_out_of_train": data["n_train_oos"], "n_classes": n_classes},
        "high_conf_threshold": HIGH_CONF,
        "oos_scoring": "trained without OOS class; OOS examples counted as errors (true label -1)",
        "worst_shift": {
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
        "rows": [public_row(r) for r in rows],
        "figures": [
            "research/figures/study_v3_conditions.png",
            "research/figures/study_v3_reliability.png",
        ],
    }
    (OUT_DIR / "study_v3_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "study_v3_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")


if __name__ == "__main__":
    main()
