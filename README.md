# Glial regulator — a learned, context-reading modulator inside a frozen LLM

**Status: research in progress. The infrastructure is done and verified; the science is
partly negative, and that is the point of this repository.**

This project asks whether a small, learned, *context-reading* glial module — inspired by
astrocytes in biological tissue — can regulate specific neurons of a frozen language
model: suppress a failure mode when it appears, stay silent when it does not, and be
switched off on demand.

Everything here was built and run on Kaggle T4s by an agent, with every decision,
deviation and negative result recorded in `lab/LOG.md`.

## What is here

| piece | state | where |
|---|---|---|
| Glial field: slow-state reader + per-behaviour heads → bounded per-neuron gains | implemented, unit-tested, Kaggle-verified | `src/glial.py` |
| Integration into Qwen3.5-0.8B via hooks on `mlp.down_proj` inputs | verified (identity at init, hooks live, dtype/device tested) | `src/glial.py`, `tests/` |
| Gradient attribution search over all 86k neurons | implemented; **negative result** on the loop behaviour | `src/attribute.py`, ADR-0001 |
| Push-screen protocol (candidate → dose ladder → held-out tests → Fisher + random controls) | implemented, not yet exercised | `src/push_screen.py` |
| Training loop for the glial field (frozen base, segmented docs, capability guard) | implemented, smoke-tested | `src/train_glial.py`, `src/stageA.py` |
| Diagnostics: context-blindness of the read signal; authority of a bounded gain | see `lab/` | `src/diagnose_context.py` |

## The two results so far

**1. First-order attribution does not recover the loop circuit (negative).** The
Neuron-Zoo method ranks neurons by how much a push changes `log p(trait) − log p(plain)`
in one backward pass; on their 4B model it placed a known persona neuron 42nd of 193,536.
We replicated the method on Qwen3.5-0.8B and tested it against ground truth we own (64
loop neurons per layer in L15/L19, verified by an exact weight edit that lifted
closed-thinking from 11% to 83%). Overlap: **2/10 (L15) and 0/10 (L19)**; a
finite-difference re-check gave 1/10 and 0/10. Interpretation: the loop is an emergent
long-generation behaviour, and next-token sensitivity does not expose its circuit —
consistent with neuron-lens finding 4 (lens and steering agree because both average
over contexts). See `docs/adr/0001-attribution-does-not-transfer-to-loops.md`.

**2. A document-level glial read is context-blind (measured).** Pooled residuals at
layers 4/8/12 are **92.3% similar across documents** (pairwise cosine). The reader
therefore cannot be context-dependent with that signal, which explains the training
failure below.

## Why the glial training failed (and what it taught)

Training the loop head against the static edit's healthy completions drove the head to
**saturate at the gain bound** (mean |g| 0.496 against σ = 0.50) while the loss barely
moved. Two contributing causes, one fixed and one diagnosed:

- *Fixed:* an unbounded gain scale with huge gradients. Mitigations now in place —
  head temperature, normalised state, gradient clipping, a lower learning rate, and a
  **saturation guard that skips the evaluation entirely** when the head pins at the
  bound (it saved ~4 GPU-hours on a run that could not have succeeded).
- *Diagnosed:* the read signal is nearly constant across documents (result 2 above),
  so the optimum is a constant gate. A constant gate is context-blind by construction
  and degenerates the model — which is why the first evaluation attempt generated the
  full 16k tokens on every problem and burned ~950 s each.

The redesign implied by these findings is in `docs/adr/0002-*.md`: read **chunk-local**
residuals (so the gain can change within a generation, which is what "about to loop"
requires), and give the regulator enough authority to matter — a bounded per-neuron
multiplier may simply be too weak to emulate a ×6 removal of a direction.

## Reproducing

`REPLICATE.md` has the exact commands. Requirements: Kaggle CLI credentials, ~30 min
GPU per session. Nothing heavy runs on a local machine; CPU kernels are used for gates.

```bash
python3 tests/test_glial_gain.py        # unit: gains preserve dtype, touch only targets
python3 tests/test_payload_layout.py     # unit: kernel payload paths match the driver
python3 build_stageA.py && kaggle kernels push -p kernel_stageA
```

## Credit

- The weight-space reproduction this project builds on (and the 128 verified loop
  neurons): Takehi, Teoh, Zhang & Khattab, *Sparks of Weight-Space Programming*
  (arXiv / blog, Nov 2026) — MIT CSAIL. Their logs: `RikiyaT/lm_understand-logs`.
- Biology: Araque & Navarrete (2010) on astrocyte processing; the artificial-glial
  taxonomy from Alvarez-Gonzalez et al. (2023). Method references:
  `justinrhowell/neuron-steering` (attribution search), `shayansanjideh/neuron-lab`
  (the same protocol on Kaggle T4s), `stdlogits/neuron-lens` (lenses), `decoderesearch/qwen-3.5-saes`
  (SAEs for this model family), `lyramakesmusic/neuronary` (recipe format).
- Base model: Qwen/Qwen3.5-0.8B (Apache-2.0).
