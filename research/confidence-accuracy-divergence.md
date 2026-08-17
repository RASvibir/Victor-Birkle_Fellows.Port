# Confidence vs. accuracy under production drift

**Status:** Pilot + study v2 + study v3 + study v4 complete  
**Author:** Victor E. Birkle III  
**Date started:** August 2026  
**Related experience:** Data research / conversational AI pipeline work at OpenCity (2022–2024)

**Artifacts:**
- Pilot: [`pilot.md`](./pilot.md) · underconfidence under token dropout
- Study v2: [`study_v2.md`](./study_v2.md) · overconfidence under competing-intent cue injection (synthetic)
- Study v3: [`study_v3.md`](./study_v3.md) · CLINC150 — cue-inject/OOS did **not** reproduce v2 overconfidence (BoW)
- Study v4: [`study_v4.md`](./study_v4.md) · same splits — DistilBERT also did **not** reproduce it; MSP AUROC ≥ 0.92

---

## Question

In a production ASR → NLU intent-classification pipeline, under what measurable conditions does **reported model confidence diverge from realized accuracy** — and can those conditions be detected from operational signals alone (latency, audio quality proxies, utterance length, class prior shift) without privileged access to ground truth at inference time?

---

## Results so far

| Pass | Mechanism | Calibration failure |
|------|-----------|---------------------|
| Pilot | Token dropout + vocab injection | Underconfidence (conf &lt; acc) |
| Study v2 | Competing-intent cue injection | **Overconfidence** (conf−acc ≈ +0.20; ~43% of errors still ≥0.8 conf) |
| Study v2 | ASR keyword swaps | Mostly underconfidence; weak high-conf errors |
| Study v2 | Prior shift alone | Did not systematically create overconfidence |
| Study v2 | Ops probe (no confidence feature) | AUC 0.72 for large mismatch |
| Study v3 | CLINC150 in-domain (BoW) | Underconfident already (gap −0.14) |
| Study v3 | Cue inject on real text @ 0.6 | **Null vs v2** — gap −0.019; P(high-conf\|err) 0.091 |
| Study v3 | OOS (no OOS class in model) | Low-confidence errors (mean conf 0.215; P(high-conf\|err) 0.028) |
| Study v3/v4 | MSP AUROC ID vs OOS (BoW) | **0.921** |
| Study v4 | DistilBERT cue-inject @ 0.6 | Gap +0.023; P(high-conf\|err) **0.021** (not v2) |
| Study v4 | DistilBERT OOS | Mean conf 0.204; P(high-conf\|err) 0.006 |
| Study v4 | MSP AUROC ID vs OOS (DistilBERT) | **0.947** |

**Takeaway:** synthetic cue injection produced overconfidence; the same mechanism on real CLINC150 text did not, for either bag-of-words or DistilBERT. Max-softmax already separates OOS. The production pathology needs a corruption that makes the model confidently wrong.

---

## Why this question

At OpenCity I spent substantial time cleaning multimodal training data, building regression checks for conversational models, and writing log parsers that surfaced anomalous application state. The recurring pattern was not “the model is always wrong,” but **confidence that stayed high while accuracy slipped** when input conditions changed.

---

## Method — study v4 (complete)

Same CLINC150 protocol as study v3. Two models on identical shifted text: BoW+logistic and fine-tuned `distilbert-base-uncased` (2 epochs, seed 42). Report ECE, conf−acc gap, P(high-conf|error), and MSP AUROC for in-domain vs OOS.

## Deliverables

| Artifact | Status |
|----------|--------|
| Pilot + study v2 code/charts/writeups | Done |
| Study v3 (CLINC150 BoW) | Done |
| Study v4 (encoder column + MSP AUROC) | Done |
| This plan | Living |

## What this is not

- Not a pitch for employment.
- Not a claim of prior ML publications.
- Not safety theater — negatives are published (pilot underconfidence; prior-shift bust; v3/v4 null vs synthetic overconfidence).
