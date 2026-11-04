# Results — glial regulator (Stage A and diagnostics)

Every number below was produced on Kaggle T4s by the code in this repository; the
commands are in `REPLICATE.md` and the dated narrative is in `lab/LOG.md`. Numbers from
the published weight-space work are marked **[paper]** and were re-measured here where
noted.

## 0. What was re-measured first (the ground truth we own)

The static edit from Takehi et al. (2026) removes a direction from 64 `down_proj`
columns of layer 15 and 64 of layer 19 of Qwen3.5-0.8B. Re-measured on the 18 paired
MATH-style problems (16k-token cap, greedy):

| model | closed thinking | rate |
|---|---|---|
| base Qwen3.5-0.8B | 2/18 | 11% **[paper]** |
| authors' released edited checkpoint | 15/18 | 83% **[paper]** |

So the fix is large, reproducible, and available to us as weights.

## 1. Diff census — what the authors actually changed

| fact | value |
|---|---|
| tensors differing between `Qwen/Qwen3.5-0.8B` and `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | **2 of 321** |
| which | `model.layers.15.mlp.down_proj.weight`, `model.layers.19.mlp.down_proj.weight` |
| changed elements per tensor | 65,260 / 64,928 (≈64 columns × 1024) |
| our independently diagnosed neurons that overlap the authors' edited columns | **62/64 (L15)**, **63/64 (L19)** |

The second row is a genuine independent replication result: a diagnosis made by a
different method (activation-change attribution over the whole model) recovered 62–63
of the 64 neurons the authors edited, per layer.

## 2. Negative result — first-order attribution does not find the loop circuit

The Neuron-Zoo method ranks neurons by gradient attribution of a trait push. Applied
to Qwen3.5-0.8B and checked against the ground truth above:

| metric | L15 | L19 |
|---|---|---|
| top-10 attribution overlap with the verified loop neurons | 2/10 | 0/10 |
| finite-difference re-check of the top 10 | 1/10 | 0/10 |

Interpretation: the loop is an emergent long-generation behaviour, and next-token
sensitivity does not expose its circuit. (Consistent with neuron-lens finding 4:
lenses and steering agree because both average over contexts.) See
`docs/adr/0001-attribution-does-not-transfer-to-loops.md`.

## 3. Diagnostics — why the first glial design could not work

| measurement | value |
|---|---|
| pairwise cosine between documents' pooled residuals (L4/8/12) | **0.9231** |
| same, last-64-token pooling | 0.9099 |
| pairwise cosine between the **reader's output z** across documents | **0.9984** |
| gain +0.25 → KL vs base | 0.00051 |
| gain +0.5 → KL vs base | 0.00251 |
| gain −0.5 → KL vs base | 0.00126 |
| gain −1.0 (128 neurons fully attenuated) → KL vs base | 0.0030 |

Two conclusions, both measured rather than assumed:

1. A document-level mean-pooled read is **context-blind** (99.8% identical), so the
   learned optimum is a constant gate. That is exactly what happened: the head
   saturated at the bound (mean |g| = 0.496 vs σ = 0.50) while the training loss moved
   from 79.33 to 74.50 on the last document and not at all on the first.
2. The activation-gate lever has **no authority**: even fully zeroing the 128 verified
   loop neurons changes the next-token distribution by KL 0.003. The static edit's
   strength comes from removing a *direction*, not scaling an activation.

### Training runs (each ~2–4 min, deliberately cheap)

| run | config | mean \|gain\| at end | outcome |
|---|---|---|---|
| v1 | σ=0.5, lr 3e-3, no clipping | 0.50000 | saturated in 1 epoch; eval would have degenerated the model (~950 s/problem) |
| v2 | σ=0.2, lr 5e-4, clip 1.0, answer-span target | 0.19835 | saturated; saturation guard skipped the eval |
| v3 | σ=0.5, temp 2.0, normalised z, lr 1e-3, clip 0.5, thinking-tail target | 0.49574 | saturated; guard fired again |

The **saturation guard** added after v1 (skip the evaluation entirely when the head pins
at the bound) turned two would-be 4-GPU-hour sessions into two 2-minute ones.

## 4. Redesign (ADR-0002) — the glial scales a known fix

`out[t] -= α(t)·(act[t,idx]·D[:,idx]ᵀ)`, α = 0 → base model, α = 1 → the verified fix.
The learned quantity is one scalar per token.

Verification status at the time of writing:

| check | result |
|---|---|
| logit equivalence, checkpoint lever at α=1 vs the authors' edited checkpoint | **max 0.164 (fp16 noise), argmax agrees** on both prompts |
| our own 64+64 neurons vs the authors' edited columns | 62/64 (L15), 63/64 (L19) — a lever rebuilt over ours matches the checkpoint lever's effect (KLs to 3 digits) |
| KL at the prompt / inside a looping continuation | 0.009–0.022 / 0.03 (the lever has real authority) |
| closed-thinking at α=1, 16k cap, problem 0 | **did not close**, while the static edit closed at 4,555 chars |

### The finding that matters most: one-step equivalence is not horizon equivalence

The lever is algebraically equivalent to the weight edit and provably logit-equivalent
(max |Δlogits| = 0.164), yet over a 16k-token generation it does **not** reproduce the
edit's behaviour. Cause: the edited checkpoint computes `act @ W_editedᵀ` (one fp16
matmul) while the lever computes `(act @ W_baseᵀ) + (act @ Dᵀ)` (two matmuls and an
add). The paths differ at ~1e-3 relative; over thousands of steps that compounds and the
reasoning loop is chaotic, so the perturbed trajectory falls into the attractor and the
unperturbed one does not.

Two further corrections were needed to get here, both found without spending quota:

- `proj_L*.npy` is stored **negated** (`corr(live W·d̂, proj) = −1.000000` exactly), so
  the delta built as `−6·proj·d̂` was the **anti-fix** — visible in the behaviour (that
  arm looped *longer* than base: 32,731 vs 24,006 chars). `scripts/check_proj.py` now
  verifies the convention locally.
- the saved `loopdir/proj` arrays were computed against an **older revision** of the
  model (the dataset copy, 488 tensors) while the hub's current revision has 321
  tensors, so the same projection differs by ~8x in magnitude. A lever must be rebuilt
  against the weights it will run on.

*(Earlier equivalence attempt reported a mismatch of ~10 in max |Δlogits|. Cause: that
kernel applied α=6 on a delta that already encodes the full 6× edit, i.e. it applied
the fix seven times over. Fixed by running the checkpoint delta at α=1.)*

## 5. Honest status

- The **infrastructure** is done and tested: hooks that install into a real model, unit
  tests for the two nastiest integration bugs (device, dtype), a payload-layout
  regression test that is proven RED/GREEN against the bug that killed two kernel
  launches, kernel builders that fail fast on missing payload files, and a saturation
  guard that protects GPU quota.
- The **science** is at "design validated, effect not yet demonstrated". The next
  experiment is small: train α(t) on the 18 paired problems with the Δ-lever, then check
  (i) closed-thinking rate, (ii) selectivity (α high on looping contexts, low
  elsewhere), (iii) knockout (α = 0 → the loop returns), (iv) capability KL.
- Pre-registered success band (unchanged from the original Stage A): closed-thinking
  ≥ 70% and knockout closed rate ≤ 25%.

## 6. The runtime lever's fidelity horizon: exactly 875 tokens (replicated twice)

Measured on two independent machines, both fp32, both greedy, both on `gsm8k_train_0`,
comparing base+lever(alpha) against the authors' edited checkpoint token-for-token:

| platform | alpha=1 | alpha=1.5 |
|---|---|---|
| Kaggle CPU kernel (4 threads, fp32, 1024-token rollouts) | **875** | 8 |
| Colab T4 (fp32, 16,384-token rollouts) | **875** | 8 |

The same divergence point to the token on different hardware, in different dtypes'
neighbourhood, at two different horizon lengths: the lever at alpha=1 *is* the fix for
875 tokens, and over-correcting (alpha=1.5) leaves the fix's trajectory within 8 tokens.

### The protocol catch (why the first Colab run's closed-thinking column was void)

The first Colab run ran the arms **without the presence penalty**. The reference
protocol - the published work and our own stage-2 replication - applied a presence
penalty of 1.5 to *every* arm, base and fixed alike. Under the un-penalised harness
even the authors' edited checkpoint did not close its thinking within 16,384 tokens
(50,682 characters, `closed=False`), which is inconsistent with our stage-2 result
(our static edit closed this same problem after 4,555 characters *under the penalised
protocol*). So an arm run without the penalty is not protocol-matched, and its
closed-thinking label cannot be compared with the published numbers. The harness now
implements the penalty (`harness.PresencePenalty`) and it is threaded through every arm.

This is worth stating plainly: two of the numbers that look like findings in the
un-penalised run (the fix "not closing", the lever "not closing") are protocol
artefacts, not results. The fidelity numbers (875, 8) are unaffected - both arms
receive the same treatment - and they are the ones that replicate.
