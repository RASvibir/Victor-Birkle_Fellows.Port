# Confidence vs. accuracy under production drift

**Status:** Pilot + study v2 complete · study v3 (CLINC150) in progress  
**Author:** Victor E. Birkle III  
**Date started:** August 2026  
**Related experience:** Data research / conversational AI pipeline work at OpenCity (2022–2024)

**Artifacts:**
- Pilot: [`pilot.md`](./pilot.md) · underconfidence under token dropout
- Study v3: [`study_v3.md`](./study_v3.md) · CLINC150 real data — **in progress, no findings yet**

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

## Method — study v3 (in progress)

CLINC150 (public, CC BY 4.0, DeepPavlov Hugging Face release): real utterances and a built-in out-of-scope class as a distribution-shift condition, not synthetic corruption. Same metrics as prior passes. Results unpublished until the run exists.

## Deliverables

| Artifact | Status |
|----------|--------|
| Pilot + study v2 code/charts/writeups | Done |
| Study v3 (CLINC150) | In progress |
| This plan | Living |

## What this is not

- Not a pitch for employment.
- Not a claim of prior ML publications.
- Not safety theater — negatives are published (pilot underconfidence; prior-shift bust).

