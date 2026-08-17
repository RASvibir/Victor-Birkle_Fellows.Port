# Confidence vs. accuracy under production drift

**Status:** Pilot + study v2 complete · production-faithful pass next  
**Author:** Victor E. Birkle III  
**Date started:** August 2026  
**Related experience:** Data research / conversational AI pipeline work at OpenCity (2022–2024)

**Artifacts:**
- Pilot: [`pilot.md`](./pilot.md) · underconfidence under token dropout
- Study v2: [`study_v2.md`](./study_v2.md) · overconfidence under competing-intent cue injection

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

**Takeaway:** corruption type decides the failure mode. The production-shaped question is no longer “can we measure anything?” — it is “which real ASR/domain mechanisms look more like cue injection than like dropout?”

---

## Why this question

At OpenCity I spent substantial time cleaning multimodal training data, building regression checks for conversational models, and writing log parsers that surfaced anomalous application state. The recurring pattern was not “the model is always wrong,” but **confidence that stayed high while accuracy slipped** when input conditions changed.

---

## Method — production-faithful pass (planned)

1. Public or production-like speech/text with controlled **ASR/channel** corruptions and natural domain shift (not only synthetic cue injection).
2. Simple reproducible baseline; fixed seed.
3. Per utterance: prediction, confidence, label, operational features.
4. Reliability / ECE stratified by bins; secondary: ops features → `|confidence − correctness|` on a holdout.
5. Compare explicitly against pilot + study v2 baselines.

## Deliverables

| Artifact | Status |
|----------|--------|
| Pilot + study v2 code/charts/writeups | Done |
| This plan | Living |
| Production-faithful pass | Next |

## What this is not

- Not a pitch for employment.
- Not a claim of prior ML publications.
- Not safety theater — negatives are published (pilot underconfidence; prior-shift bust).

