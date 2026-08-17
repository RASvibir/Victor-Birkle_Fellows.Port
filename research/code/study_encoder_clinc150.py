"""
Study v4: same CLINC150 protocol as study v3, second model class.

Keeps the bag-of-words logistic column and adds a fine-tuned encoder
(default: DistilBERT). Question: does the v2 high-confidence-error
pathology appear once the model can be overconfident?

Also reports max-softmax-probability AUROC for in-domain vs OOS
(Hendrycks & Gimpel 2017 style). OOS remains errors by construction
(no OOS class in training).

Does not invent numbers. Writes JSON + figures for the writeup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_confidence_drift import ECE_BINS, FIG_DIR, SEED, expected_calibration_error  # noqa: E402
from study_real_data_clinc150 import (  # noqa: E402
    CUE_LEVELS,
    DATASET_ID,
    HIGH_CONF,
    class_cues,
    cue_inject,
    load_clinc,
    operational_probe,
    public_row,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
CKPT_DIR = OUT_DIR / "distilbert_clinc150_seed42"
MODEL_ID_DEFAULT = "distilbert-base-uncased"


def msp_ood_auroc(id_conf: np.ndarray, oos_conf: np.ndarray) -> float:
    """Higher max-softmax → more in-distribution. Label 1 = ID, 0 = OOS."""
    scores = np.concatenate([id_conf, oos_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(oos_conf))])
    return float(roc_auc_score(labels, scores))


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EncoderClf:
    def __init__(self, model, tokenizer, device, max_length: int = 64, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        self.model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i : i + self.batch_size])
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(chunks)


def train_encoder(
    train_x,
    train_y,
    val_x,
    val_y,
    n_classes: int,
    model_id: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    retrain: bool,
) -> EncoderClf:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = pick_device()
    if CKPT_DIR.exists() and not retrain:
        print(f"Loading encoder checkpoint from {CKPT_DIR}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
        model.to(device)
        model.eval()
        return EncoderClf(model, tokenizer, device, max_length=max_length)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    class TextCls(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = [int(y) for y in labels]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(
                self.texts[i],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=n_classes)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    train_loader = DataLoader(TextCls(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextCls(val_x, val_y), batch_size=max(8, batch_size), shuffle=False)

    steps = max(1, epochs * len(train_loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "model_id": model_id,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "train_steps": steps,
                "n_train": len(train_x),
                "n_val": len(val_x),
            }
        ),
        flush=True,
    )

    best_val = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * len(batch["labels"])
            n_seen += len(batch["labels"])
            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} "
                    f"loss={running / max(1, n_seen):.4f}",
                    flush=True,
                )

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                pred = logits.argmax(dim=-1)
                correct += int((pred == batch["labels"]).sum().item())
                total += len(batch["labels"])
        val_acc = correct / max(1, total)
        print(f"epoch {epoch + 1} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    (CKPT_DIR / "train_meta.json").write_text(
        json.dumps({"val_acc": best_val, "epochs": epochs, "model_id": model_id, "seed": SEED}, indent=2)
        + "\n"
    )
    del optimizer, scheduler
    if str(device) == "mps":
        torch.mps.empty_cache()
    return EncoderClf(model, tokenizer, device, max_length=max_length)


def eval_model(pipe, texts_by_regime: list[tuple], train_vocab: set[str]) -> list[dict]:
    rows = []
    for texts, y, regime, level in texts_by_regime:
        rows.append(summarize(pipe, texts, y, regime, level, train_vocab))
    return rows


def ood_block(rows: list[dict]) -> dict:
    id_row = next(r for r in rows if r["regime"] == "in_domain")
    oos_row = next(r for r in rows if r["regime"] == "oos")
    return {
        "method": "max softmax probability (Hendrycks & Gimpel 2017)",
        "auroc_id_vs_oos": round(msp_ood_auroc(id_row["confidences"], oos_row["confidences"]), 4),
        "id_mean_conf": round(float(id_row["mean_confidence"]), 4),
        "oos_mean_conf": round(float(oos_row["mean_confidence"]), 4),
        "n_id": id_row["n"],
        "n_oos": oos_row["n"],
        "note": "score=MSP; label 1=in-domain, 0=OOS. Not a claim about dummy OOS accuracy.",
    }


def plot_compare(models: dict[str, list[dict]], path: Path) -> None:
    names = list(models)
    conditions = [f"{r['regime']}" + (f"@{r['level']}" if r["regime"] == "cue_inject" else "") for r in models[names[0]]]
    x = np.arange(len(conditions))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    specs = [
        ("accuracy", "Accuracy", (0, 1.05)),
        ("gap", "Confidence − accuracy", None),
        ("high_conf_error_rate", f"P(conf≥{HIGH_CONF} | error)", (0, 1.05)),
    ]
    colors = ["#1f5c4a", "#1d4ed8"]
    for ax, (key, title, ylim) in zip(axes, specs):
        for i, name in enumerate(names):
            vals = [r[key] for r in models[name]]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=name, color=colors[i % 2])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    axes[0].legend(fontsize=8)
    fig.suptitle("Study v4: BoW logistic vs encoder on CLINC150", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_msp_ood(models: dict[str, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(10.2, 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, models.items()):
        id_row = next(r for r in rows if r["regime"] == "in_domain")
        oos_row = next(r for r in rows if r["regime"] == "oos")
        ax.hist(id_row["confidences"], bins=20, range=(0, 1), density=True, alpha=0.55, color="#1f5c4a", label="in-domain")
        ax.hist(oos_row["confidences"], bins=20, range=(0, 1), density=True, alpha=0.55, color="#9a3412", label="OOS")
        auroc = msp_ood_auroc(id_row["confidences"], oos_row["confidences"])
        ax.set_title(f"{name}\nMSP AUROC={auroc:.3f}")
        ax.set_xlabel("Max softmax")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Study v4: MSP separates in-domain from OOS", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(clean: dict, other: dict, path: Path, title: str) -> None:
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
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def model_public(name: str, spec: str, rows: list[dict]) -> dict:
    shifted = [r for r in rows if r["regime"] != "in_domain"]
    worst = max(shifted, key=lambda r: r["high_conf_error_rate"])
    return {
        "name": name,
        "spec": spec,
        "ood_detection": ood_block(rows),
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
        "operational_probe": operational_probe(rows),
        "rows": [public_row(r) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--bow-only", action="store_true", help="Skip encoder (debug)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    data = load_clinc()
    n_classes = int(data["train_y"].max()) + 1
    cues = class_cues(data["train_x"], data["train_y"])
    train_vocab = {w.lower() for t in data["train_x"] for w in t.split()}

    injected = {}
    for level in CUE_LEVELS:
        if level == 0.0:
            continue
        injected[level] = [
            cue_inject(t, int(y), level, cues, rng, n_classes)
            for t, y in zip(data["test_x"], data["test_y"])
        ]
    oos_y = np.full(len(data["test_oos"]), -1, dtype=int)
    regimes = [
        (data["test_x"], data["test_y"], "in_domain", 0.0),
        (injected[0.3], data["test_y"], "cue_inject", 0.3),
        (injected[0.6], data["test_y"], "cue_inject", 0.6),
        (data["test_oos"], oos_y, "oos", 0.0),
    ]

    bow = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
        ]
    )
    bow.fit(data["train_x"], data["train_y"])
    bow_rows = eval_model(bow, regimes, train_vocab)
    print("BoW done", json.dumps(ood_block(bow_rows)), flush=True)

    models_rows = {"bow_logistic": bow_rows}
    models_public = {
        "bow_logistic": model_public(
            "bow_logistic",
            "CountVectorizer(1-2gram) + LogisticRegression (study v3 class)",
            bow_rows,
        )
    }

    encoder_meta = None
    if not args.bow_only:
        enc = train_encoder(
            data["train_x"],
            data["train_y"],
            data["val_x"],
            data["val_y"],
            n_classes,
            args.model_id,
            args.epochs,
            args.batch_size,
            args.max_length,
            args.retrain,
        )
        enc_rows = eval_model(enc, regimes, train_vocab)
        models_rows["distilbert"] = enc_rows
        models_public["distilbert"] = model_public(
            "distilbert",
            f"{args.model_id} fine-tuned, seed {SEED}, epochs={args.epochs}",
            enc_rows,
        )
        encoder_meta = {
            "model_id": args.model_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": str(enc.device),
            "checkpoint": str(CKPT_DIR.relative_to(Path(__file__).resolve().parents[2]))
            if CKPT_DIR.exists()
            else None,
        }
        print("Encoder done", json.dumps(ood_block(enc_rows)), flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_compare = FIG_DIR / "study_v4_compare.png"
    fig_ood = FIG_DIR / "study_v4_msp_ood.png"
    fig_rel = FIG_DIR / "study_v4_reliability.png"
    plot_compare(models_rows, fig_compare)
    plot_msp_ood(models_rows, fig_ood)
    focus_name = "distilbert" if "distilbert" in models_rows else "bow_logistic"
    focus = models_rows[focus_name]
    shifted = [r for r in focus if r["regime"] != "in_domain"]
    worst = max(shifted, key=lambda r: r["high_conf_error_rate"])
    plot_reliability(focus[0], worst, fig_rel, f"Study v4 reliability ({focus_name})")

    summary = {
        "seed": SEED,
        "dataset": DATASET_ID,
        "dataset_fingerprint": data["revision"],
        "license": "CC BY 4.0 (CLINC150 / DeepPavlov HF release)",
        "question": "On the same CLINC150 splits as study v3, does a fine-tuned encoder produce v2-style high-confidence errors under cue-inject or OOS?",
        "protocol": "Train without OOS class. Cue-inject uses the same lexical-cue procedure as study v3, on the same test utterances. Both models see identical shifted text.",
        "citations": [
            "Larson et al. 2019, An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (CLINC150)",
            "Hendrycks & Gimpel 2017, A Baseline for Detecting Misclassified and Out-of-Distribution Examples (MSP)",
            "Guo et al. 2017, On Calibration of Modern Neural Networks",
        ],
        "high_conf_threshold": HIGH_CONF,
        "encoder": encoder_meta,
        "models": models_public,
        "figures": [
            "research/figures/study_v4_compare.png",
            "research/figures/study_v4_msp_ood.png",
            "research/figures/study_v4_reliability.png",
        ],
    }
    payload = json.dumps(summary, indent=2) + "\n"
    (OUT_DIR / "study_v4_metrics.json").write_text(payload)
    (ROOT / "study_v4_metrics.json").write_text(payload)
    print(payload)
    print(f"Wrote {fig_compare}")
    print(f"Wrote {fig_ood}")
    print(f"Wrote {fig_rel}")


if __name__ == "__main__":
    main()
