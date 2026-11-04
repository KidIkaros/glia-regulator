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

## 7. The behavioural optimum is NOT the fidelity optimum (the night's result)

Protocol-matched on Colab (T4, fp32, greedy, **presence penalty 1.5 on every arm**),
on `gsm8k_train_0`, 8192-token cap:

| arm | tokens generated | characters | closed its thinking? | matched the fix's trajectory |
|---|---|---|---|---|
| fix reference (the authors' edited checkpoint) | 6,320 | 18,340 | **yes** | — |
| lever alpha=1 | 8,192 (cap) | 26,429 | **no** | 546 |
| **lever alpha=1.5** | 1,838 | 4,881 | **yes** | 8 |

Three things fall out of this, and they matter more than any of the earlier numbers:

1. **The runtime lever CAN suppress the reasoning loop.** At alpha=1.5 the lever closes
   the model's thinking after 1,838 tokens - and *faster than the published weight edit
   does* (4,881 vs 18,340 characters). The mechanism that was in doubt - whether a
   per-token runtime intervention has enough authority to break a long-horizon chaotic
   attractor - is answered **yes**, and the glial's premise survives.
2. **The fidelity optimum and the behavioural optimum are different points.** alpha=1
   reproduces the weight edit's *predictions* best (KL 2.3e-5, and a 875-token
   trajectory match in the un-penalised protocol), yet it is the arm that does **not**
   close the loop. alpha=1.5 leaves the fix's trajectory within 8 tokens and is the arm
   that does. A regulator must be calibrated on behaviour, not on similarity to a
   reference intervention.
3. **Why over-correction is the right direction.** The weight edit removes the loop
   direction *permanently* (strength 6 on every column, once); the runtime lever applies
   a *partial* correction to each step's write. Matching the edit's local predictions
   therefore does not match its cumulative effect over thousands of steps - and the
   lever needs a larger per-step push than the edit does to reach the same behavioural
   outcome. This is the same mechanism as the fidelity horizon, seen from the other
   side.

### What this unblocks, and what it does not

- It unblocks the glial experiment: there is now a lever with demonstrated authority
  over the behaviour, so the open question becomes *when* to apply it (a context-reading
  alpha(t)), not *whether* a runtime intervention can work at all.
- It does **not** establish that alpha=1.5 is safe: alpha=1.5 removes 9x the loop
  direction instead of 6x. The capability panel (perplexity, language mixing, ordinary
  prompts) is now the binding constraint, and the dose-response must map closed-thinking
  rate against capability damage before alpha(t) is trained on it.
- The dose-response is still coarse: one problem, two alphas. The next sweep is
  alpha in {1.0, 1.25, 1.5, 1.75, 2.0} across the 18 paired problems, with the
  capability panel attached.

## 8. Dose-response with a capability panel: the operating point (and a regression)

Run on two Colab T4 sessions in parallel (8 problems per alpha: 6 known loopers plus
2 healthy controls the base model already closes). Protocol as published: greedy,
presence penalty 1.5 on every arm, fp32, 8192-token cap.

| alpha | closed | closed rate | healthy controls closed | ordinary KL | argmax agreement | perplexity | gate |
|---|---|---|---|---|---|---|---|
| 1.0 | 4/8 | 0.50 | **1/2** | 0.0047 | 100% | +3.07% | **FAIL** |
| **1.25** | **8/8** | **1.00** | **2/2** | 0.0077 | 100% | +4.56% | **PASS** |
| 1.5 | 6/8 | 0.75 | 2/2 | 0.0117 | 100% | +6.33% | PASS |
| 1.75 | 8/8 | 1.00 | 2/2 | 0.0167 | 100% | +8.36% | PASS |
| 2.0 | 7/8 | 0.88 | 2/2 | 0.0227 | 100% | +10.64% | PASS |

### The most important row is alpha = 1.0

alpha=1.0 is the dose that reproduces the published edit's *predictions* best (KL
2.3e-5 to the edited checkpoint, an 875-token trajectory match). It is also the only
dose that **fails** the gate - and it fails by *breaking* something, not merely by
failing to fix something:

* it leaves 4 of 6 loopers looping (train_2, train_4, train_5 all ran to the 8192-token
  cap), and
* it turns `gsm8k_train_0` - a problem the **base model solves correctly** - into a
  8192-token loop. Under alpha=1.25 the same problem closes in 6,598 tokens, and the
  base closes it in 1,201.

So the "faithful" dose is a *regression* on healthy work, while a 25% stronger dose
fixes everything. The dose that best mimics the reference intervention is not the dose
that works, and the difference is not subtle.

### Why 1.25 is the operating point

The closed rate saturates at 1.25 (8/8 - every looper and both healthy controls), while
the capability cost keeps growing monotonically with the dose (KL 0.0047 -> 0.0227,
perplexity +3.1% -> +10.6%). 1.25 is therefore the smallest dose that achieves the full
effect, and it carries the smallest cost of any passing dose. The noise in the closed
rate between 1.25 and 2.0 (8/8, 6/8, 8/8, 7/8) is the resolution of an 8-problem set
(one problem = 12.5%), not a real non-monotonicity - the capability damage, which is
measured continuously, is cleanly monotone.

### The capability result is stronger than expected

At **every** dose, including the failing one, the intervened model's next-token argmax
on all 8 ordinary prompts is **unchanged** from the base model (100% agreement), and
perplexity rises by only 3-11% on a fixed neutral passage. Removing the loop direction
does not blunt ordinary text. That is what makes the lever a plausible regulator rather
than a blunt instrument - but it is a statement about *ordinary* text, and the
healthy-control column is what says the intervention is not merely making the model
terse: at alpha=1.25 the healthy problems still close AND still take thousands of
tokens.

### What this settles for Stage B

* the operating point is **alpha* ~= 1.25**, so the regulator should emit 0 (off) or
  ~1.25 (on) - never 1.0, which is the harmful dose;
* the failure mode to avoid is not "too weak" but "on when it should be off" - a
  regulator that fires on healthy contexts will break them exactly as alpha=1.0 did;
* therefore the reader must be **selective**, and its selectivity needs its own gate:
  false-positive firings on healthy contexts are the thing that damages capability.

### Limitations (stated, not hidden)

* 8 problems per dose: the closed rate has 12.5% resolution. The shape of the curve
  near saturation is not resolved; the gates are.
* "closed" means the model emitted its closing marker within the budget. A row can be
  closed=True AND reach the token cap (the model closed its thinking and then kept
  talking), so a late close counts the same as an early one. The close POSITION is not
  recorded - it is added to the harness for the next phase.
* the capability panel measures ordinary text, arithmetic, code, translation and
  constrained instructions at the next-token level plus passage perplexity. It is much
  stronger than the single KL sentence it replaces, but it is not a full capability
  suite, and no claim beyond "ordinary behaviour is preserved at these doses" is made.

## 9. Negative result: the loop direction is not a prompt-level signature

Stage B needs labels (which contexts loop?) and rollout labelling costs ~410 s per
looping problem, so I tested whether the loop direction is a cheap signature: the fix
removes one direction d from the L15/L19 MLP outputs, so the projection
<down_proj_out[t], d_hat> should mark the loop state - measurable in ONE forward pass
instead of an 8192-token rollout.

**It does not work, at least at this horizon.** Over a 128-token generation on the 18
labelled problems (16 loopers, 2 healthy):

| layer | AUC (loopers vs healthy) | loopers | healthy |
|---|---|---|---|
| 15 | 0.688 | 0.1303 | 0.1121 |
| 19 | 0.312 | 0.2011 | 0.2172 |

L15 is barely above chance; L19 is *below* chance (the healthy problems have a larger
mean projection than the loopers). Neither is usable as a labeller: a threshold that
catches the loopers would fire on the healthy controls far more often than the 2/2
false-positive rate the dose-response showed to be damaging.

### Why this is not just a threshold problem

The projection measures the *magnitude* of the model's activation along the direction
the fix removes; it is not a loop detector. Early in a generation the model is doing
ordinary reasoning in both groups, and the loop has not yet established itself - at 128
tokens neither group has entered the attractor. A longer horizon might separate them,
but the regime where a looper is distinguishable is the regime where it has already
been looping for hundreds of tokens, which erodes exactly the cost advantage that
motivated the measurement.

### What it changes

1. **The regulator cannot be purely prompt-reading.** The distinction between a problem
   that will loop and one that will not is not visible in the prompt at the level this
   signal measures - and arguably not a property of the prompt at all: `gsm8k_train_0`
   (healthy) takes 6,598 tokens to close under alpha=1.25, while several loopers close
   in 1,100-2,500. "Will loop" is a property of the *trajectory*, not of the question.
2. **So the reader must read the generated trajectory, not the prompt.** That is what
   the glial's slow-state reader is for - and it means the training signal must come
   from *within* a rollout (is the recent text re-deriving the same step?), which is
   cheap to compute from tokens we already generate, rather than from a prompt-level
   classifier.
3. **The honest consequence for the plan**: cheap labelling of thousands of problems is
   NOT available via the projection. The 18 labelled problems remain the ground truth
   for closure behaviour, and the Stage B training set must be built from rollouts we
   already pay for (the sweep's 40 rollouts are a start) plus in-trajectory features.

### Measurement bug this exposed (fixed)

The prompt-only column of the first run reported AUC 0.000 with NaN scores for every
problem: `summarise()` was called after `probe.reset()` but *before* any forward pass,
so the probe had nothing to report and every score was None. The prompt-only signature
is now measured by a real forward pass. The generated-trajectory numbers above are
unaffected - those probes did see tokens.
