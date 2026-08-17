# Confidence vs. accuracy under production drift

**Status:** Method and scope published. Empirical run and charts not yet complete — this page is the investigation plan and the place results will land.

**Author:** Victor E. Birkle III  
**Date started:** August 2026  
**Related experience:** Data research / conversational AI pipeline work at OpenCity (2022–2024)

---

## Question

In a production ASR → NLU intent-classification pipeline, under what measurable conditions does **reported model confidence diverge from realized accuracy** — and can those conditions be detected from operational signals alone (latency, audio quality proxies, utterance length, class prior shift) without privileged access to ground truth at inference time?

This is deliberately narrow. It is not a claim about frontier model alignment. It is an empirical diagnostics question about a failure mode I watched in production systems: high-confidence wrong answers under distribution shift.

---

## Why this question

At OpenCity I spent substantial time cleaning multimodal training data, building regression checks for conversational models, and writing log parsers that surfaced anomalous application state. The recurring pattern was not “the model is always wrong,” but **confidence that stayed high while accuracy slipped** when input conditions changed (noise, domain shift, class imbalance).

Fellows-style research readiness, for me, looks like: take that fuzzy production memory, turn it into a scoped experiment, measure something, and write down what failed.

---

## Method (planned)

1. **Define a fixed intent taxonomy** (small: 8–12 classes) and a synthetic or public speech/text corpus with controlled corruptions (SNR drop, truncation, out-of-domain phrases).
2. **Train or freeze a simple baseline classifier** (e.g. logistic regression or small transformer on embeddings) so the experiment stays reproducible in Python without a large GPU budget.
3. **Record, per utterance:** predicted label, max softmax / calibrated confidence, true label, and operational features (length, estimated SNR or text-noise proxy, prior shift flag).
4. **Primary metric:** reliability diagrams and Expected Calibration Error (ECE) stratified by operational bins — not just aggregate accuracy.
5. **Secondary question:** which operational features best predict |confidence − correctness| with a holdout set? Report AUC / Brier, and what did *not* predict well.
6. **Negative results welcome.** If confidence tracks accuracy under these corruptions, say so.

## What will ship here

| Artifact | Location |
|----------|----------|
| This writeup (updated with findings) | `research/confidence-accuracy-divergence.md` |
| Experiment code | `research/code/` (forthcoming) |
| Charts | `research/figures/` (forthcoming) |
| One-paragraph result summary | Linked from the portfolio index |

## What this is not

- Not a pitch for employment.
- Not a claim of prior ML research publications.
- Not safety theater. If the result is weak or null, the writeup will say that.

---

## Next actions

1. Lock dataset + baseline model choice.
2. Implement ECE / reliability plot pipeline in Python.
3. Run stratified corruption sweeps; publish figures and an honest discussion.
