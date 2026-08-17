"""
Pilot: confidence vs. accuracy under controlled input corruption.

Synthetic 8-class "intent" text classification. Train a logistic regression
on clean bag-of-words features, then evaluate under increasing feature noise
and label-independent token dropout. Report accuracy, mean confidence, and
Expected Calibration Error (ECE) at each corruption level.

This is a toy pilot — not a production ASR/NLU study. See
research/confidence-accuracy-divergence.md for the fuller question.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
N_CLASSES = 8
N_PER_CLASS = 220
CORRUPTION_LEVELS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
ECE_BINS = 10

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT_DIR = Path(__file__).resolve().parent / "out"

INTENTS = [
    "balance",
    "cancel",
    "hours",
    "password",
    "refund",
    "shipping",
    "support",
    "upgrade",
]

# Small vocabulary templates per intent — deliberate lexical overlap across
# classes so the baseline is competent but not trivially perfect.
TEMPLATES = {
    "balance": [
        "what is my account balance please",
        "show checking balance now",
        "how much money do i have left on account",
        "check savings balance today please",
        "tell me my available funds for account",
        "account status for available money",
    ],
    "cancel": [
        "cancel my subscription please",
        "i want to cancel the order on account",
        "stop recurring billing now please",
        "please cancel this service today",
        "end my membership account today",
        "cancel account upgrade request please",
    ],
    "hours": [
        "what are your store hours today",
        "when are you open today please",
        "are you closed on sunday help",
        "opening hours for the branch today",
        "what time do you close please",
        "support hours for store opening",
    ],
    "password": [
        "i forgot my password help please",
        "reset password for my account",
        "change login password please now",
        "cannot remember account password help",
        "help with password recovery please",
        "account support for password reset",
    ],
    "refund": [
        "i need a refund please help",
        "request refund for purchase order",
        "money back for defective item please",
        "how do i get a refund today",
        "refund status for my order please",
        "cancel order and request refund",
    ],
    "shipping": [
        "where is my shipping package please",
        "track shipment status please help",
        "when will my order arrive today",
        "shipping delay on delivery please",
        "update on package tracking order",
        "order support for shipping status",
    ],
    "support": [
        "talk to customer support now please",
        "i need human support help today",
        "connect me with an agent please",
        "support ticket for billing issue",
        "speak with support representative now",
        "account help from support agent",
    ],
    "upgrade": [
        "upgrade my current plan please",
        "switch to premium upgrade today",
        "how do i upgrade service please",
        "move account to higher tier plan",
        "upgrade options for membership please",
        "account upgrade from support please",
    ],
}

FILLERS = [
    "um",
    "please",
    "asap",
    "thanks",
    "hey",
    "quick",
    "urgent",
    "again",
    "maybe",
    "just",
]


def make_utterance(rng: np.random.Generator, intent: str) -> str:
    base = TEMPLATES[intent][int(rng.integers(0, len(TEMPLATES[intent])))]
    tokens = base.split()
    # Mild natural variation: drop/swap filler words; sometimes splice a
    # fragment from another intent to create lexical bleed.
    if rng.random() < 0.55:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), FILLERS[int(rng.integers(0, len(FILLERS)))])
    if rng.random() < 0.4 and len(tokens) > 3:
        i = int(rng.integers(0, len(tokens) - 1))
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    if rng.random() < 0.28:
        other = INTENTS[int(rng.integers(0, len(INTENTS)))]
        frag = TEMPLATES[other][int(rng.integers(0, len(TEMPLATES[other])))].split()
        take = frag[: int(rng.integers(1, min(3, len(frag)) + 1))]
        tokens.extend(take)
    if rng.random() < 0.2 and len(tokens) > 4:
        del tokens[int(rng.integers(0, len(tokens)))]
    return " ".join(tokens)


def build_corpus(rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for idx, intent in enumerate(INTENTS):
        for _ in range(N_PER_CLASS):
            texts.append(make_utterance(rng, intent))
            labels.append(idx)
    return texts, np.array(labels, dtype=int)


def corrupt_text(text: str, level: float, rng: np.random.Generator, vocab: list[str]) -> str:
    """Token dropout + random vocab injection — proxies ASR / channel noise."""
    if level <= 0:
        return text
    tokens = text.split()
    kept = [t for t in tokens if rng.random() > level * 0.85]
    if not kept:
        kept = [tokens[int(rng.integers(0, len(tokens)))]]
    n_inject = int(round(level * 3))
    for _ in range(n_inject):
        kept.insert(int(rng.integers(0, len(kept) + 1)), vocab[int(rng.integers(0, len(vocab)))])
    return " ".join(kept)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        count = int(mask.sum())
        bin_count[i] = count
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        bin_acc[i] = acc
        bin_conf[i] = conf
        ece += (count / len(confidences)) * abs(acc - conf)
    return ece, bin_acc, bin_conf, bin_count


def evaluate_level(
    pipe: Pipeline,
    texts: list[str],
    y_true: np.ndarray,
    level: float,
    rng: np.random.Generator,
    noise_vocab: list[str],
) -> dict:
    corrupted = [corrupt_text(t, level, rng, noise_vocab) for t in texts]
    proba = pipe.predict_proba(corrupted)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = pred == y_true
    ece, bin_acc, bin_conf, bin_count = expected_calibration_error(conf, correct.astype(float))
    return {
        "level": level,
        "accuracy": float(accuracy_score(y_true, pred)),
        "mean_confidence": float(conf.mean()),
        "ece": float(ece),
        "gap": float(conf.mean() - accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "reliability": {
            "bin_acc": bin_acc.tolist(),
            "bin_conf": bin_conf.tolist(),
            "bin_count": bin_count.tolist(),
        },
        "confidences": conf,
        "correct": correct.astype(float),
    }


def plot_ece_vs_corruption(rows: list[dict], path: Path) -> None:
    levels = [r["level"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(levels, [r["accuracy"] for r in rows], "o-", color="#1f5c4a", label="Accuracy", linewidth=2)
    ax.plot(levels, [r["mean_confidence"] for r in rows], "s--", color="#9a3412", label="Mean confidence", linewidth=2)
    ax.plot(levels, [r["ece"] for r in rows], "^:", color="#1a2332", label="ECE", linewidth=2)
    ax.set_xlabel("Corruption level (token dropout + injection)")
    ax.set_ylabel("Metric value")
    ax.set_title("Pilot: confidence vs. accuracy under corruption")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reliability(clean: dict, harsh: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, row, title in [
        (axes[0], clean, f"Reliability @ corruption={clean['level']:.2f}"),
        (axes[1], harsh, f"Reliability @ corruption={harsh['level']:.2f}"),
    ]:
        centers = np.linspace(0.05, 0.95, ECE_BINS)
        acc = np.array(row["reliability"]["bin_acc"])
        conf = np.array(row["reliability"]["bin_conf"])
        count = np.array(row["reliability"]["bin_count"])
        mask = count > 0
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Perfect")
        ax.bar(centers[mask], acc[mask], width=0.08, color="#1f5c4a", alpha=0.55, label="Accuracy")
        ax.plot(conf[mask], acc[mask], "o", color="#9a3412", label="Bin mean")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Confidence")
        ax.set_title(f"{title}\nECE={row['ece']:.3f}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Pilot reliability diagrams (clean vs. harsh)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    texts, labels = build_corpus(rng)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=SEED, stratify=labels
    )

    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)

    noise_vocab = sorted({w for t in x_train for w in t.split()})
    # Add distractors not strongly tied to a single intent
    noise_vocab.extend(FILLERS)

    rows = [evaluate_level(pipe, x_test, y_test, level, rng, noise_vocab) for level in CORRUPTION_LEVELS]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ece_path = FIG_DIR / "pilot_ece_vs_corruption.png"
    rel_path = FIG_DIR / "pilot_reliability.png"
    plot_ece_vs_corruption(rows, ece_path)
    # Clean vs a mid-harsh level that still has signal
    harsh = next(r for r in rows if abs(r["level"] - 0.5) < 1e-9)
    plot_reliability(rows[0], harsh, rel_path)

    summary = {
        "seed": SEED,
        "n_classes": N_CLASSES,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "model": "CountVectorizer(1-2gram) + LogisticRegression",
        "corruption": "token dropout (~0.85*level) + random vocab injection",
        "rows": [
            {
                "level": r["level"],
                "accuracy": round(r["accuracy"], 4),
                "mean_confidence": round(r["mean_confidence"], 4),
                "ece": round(r["ece"], 4),
                "gap_conf_minus_acc": round(r["gap"], 4),
            }
            for r in rows
        ],
        "figures": [str(ece_path.relative_to(ROOT.parent)), str(rel_path.relative_to(ROOT.parent))],
    }
    out_json = OUT_DIR / "pilot_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {ece_path}")
    print(f"Wrote {rel_path}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
