# Glial regulator — design

**Question.** Can a learned, context-reading glial module regulate specific MLP neurons
*inside* a frozen language model — suppressing behaviors selectively — and does one
shared glial field regulate multiple behaviors without cross-talk?

**Why this is not another adapter, and not a hand-written ratchet.** Grounded in:
- Araque & Navarrete 2010 (playbook): glia read populations, act nonlinearly, have
  their own slower dynamics, and their effect's sign depends on the target — the
  evidence bar is selective stimulation + selective loss-of-function.
- Alvarez-Gonzalez et al. 2023 (systematic review): three artificial-glial designs —
  ANGN (a modulator per neuron), MLP+PGC (pulse chain = slow temporal state),
  SONG-NET (glia as selectors). We take ANGN's per-neuron modulation + PGC's slow state.
- neuron-lens §3/§4: a weight-only read predicts *steering* (r>0.9 late) but NOT what a
  neuron does when it fires in a given context (r 0.05-0.11 mid-network). Static
  vectors are context-blind; a glial controller is the context-aware counterpart.
- NanoCore pre-mortem E8: do not ship a hand-coded knob and call it learned. Every arm
  is learned or explicitly labelled as a control, at matched capacity.

## Architecture

```
        ┌─────────────── frozen Qwen3.5-0.8B ───────────────┐
x ──► layers 0..L ──► [MLP out · (1 + g)] ──► ... ──► logits
                          ▲
        per-neuron gain   │  heads (per behavior)
        ┌─────────────────┴──────────────┐
        │  shared glial field (small MLP) │  reads mean-pooled residual at layers 8,12,16,20
        │  + slow state h_t = decay·h_{t-1} + read(act_t)   (PGC's slower dynamics)
        └─────────────────────────────────┘
```

- `g = tanh(W_head h_glial) * σ` — bounded per-neuron gain on the selected MLP output
  columns (multiplicative on the neuron's write; init g=0 → identical to base).
- Slow state carried across tokens inside one sequence (decay ~0.9); zeroed between
  documents. This is what makes it glial rather than a per-token adapter.
- Base weights frozen. Trainable: glial reader (~0.5M) + per-behavior heads
  (64-128 dims each) ≈ <1M params.

## Behaviors and hosts

Stage 1 identification now follows the attribution-search protocol in
`docs/NEURON-IDENTIFICATION.md` (gradient attribution search + push screen + Fisher
validation), replacing the earlier activation-based candidate filter — which is a
proven failure mode (the target neuron ranked ~19,000th by activation contrast).

| behavior | host | target neurons | metric |
|---|---|---|---|
| loop | `Qwen/Qwen3.5-0.8B` (unedited) | L15/L19 64 neurons each (ground truth from wsp) | closed-thinking rate, accuracy on the 18 paired problems |
| sycophancy | `KidIkaros/qwen3.5-0.8b-loop-fix` (ours) | layers 12-20, learned selection | agreement rate with a wrong user suggestion (~100 GSM8K-derived pairs) |

## Arms (all evaluated on both behaviors)

1. **glial** — shared field + per-behavior heads, trained jointly.
2. **control-glia** — identical architecture/capacity, target neurons randomized, same
   data and step budget (the matched-capacity control the review's own comparisons use).
3. **static** — for loop: our published edit (83% closed); for syco: none exists → the
   arm is "no intervention", stated as such.
4. **base / loop-fix** — unedited references.

## Tests

- **Selectivity**: glial gain magnitude should be near-zero on healthy generations
  (measure mean |g| on non-looping vs looping contexts).
- **Knockout (Araque's loss-of-function)**: zero the glial → the behavior must return.
  A regulator that survives its own ablation is a weight edit in disguise.
- **Interference (the headline)**: disable head_syco, re-evaluate loop; disable
  head_loop, re-evaluate syco. Cross-talk is the number we care about.
- **Capability bleed**: a small MMLU/IFEval slice before/after; a regulator that fixes
  behavior at the cost of general ability is a failure.

## Pre-registered bands (to be fixed before the run)

- SUCCESS: glial closes the behavior gap to >= 70% of the static edit's effect on loop,
  syco agreement drops >= 30% relative, capability bleed <= 2pts, and interference
  (cross-behavior degradation) <= 10% relative.
- PARTIAL: behavior fixed but interference high → report the trade-off; that IS the
  biology result (one glial field, many synapses, sign depends on target).
- FAIL: behavior not fixed, or knockout does not restore it, or bleed > 5pts.

## Cost

- Design + code: free. Training: ~2-3h T4 after the Sunday reset (base frozen, backward
  through <1M params). Evals reuse the existing 18-problem harness + a sycophancy set.
- Quota ledger: 0h spent on this project so far. Reserve floor: 8h of the 30h week.
