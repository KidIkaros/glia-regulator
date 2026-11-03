# ADR-0002 — The glial must scale a known fix, not invent a mechanism

Date: 2026-11-03 · Status: accepted (design), verification in progress

## Context

Stage A of the glial experiment was supposed to answer: can a small learned module
reading the model's own context decide *when* to suppress a failure mode (the reasoning
loop) in a frozen Qwen3.5-0.8B? The first implementation gated the 128 verified loop
neurons multiplicatively — a per-neuron gain in [−σ, σ] applied to the neuron's
activation before `down_proj`.

It failed, and the failure was diagnostic rather than accidental:

1. **The gate saturated.** With σ = 0.5 the head's output hit the bound by epoch 5
   (mean |g| = 0.496) while the loss barely moved (79.33 → 74.50 on the last document;
   the first document's loss was *identical* to three decimals for 8 epochs, because a
   saturated constant gate produces a constant model).
2. **The read signal is context-blind.** Pairwise cosine between documents' pooled
   residuals (layers 4/8/12) = **0.9231**; the reader's own output `z` across documents
   = **0.9984**. A reader fed near-identical inputs can only emit near-identical gains.
   That is why the optimum was a constant gate.
3. **The lever had no authority.** Sweeping a constant gain on the 128 neurons and
   measuring KL against the base model: +0.25 → 0.00051, +0.5 → 0.00251, −0.5 →
   0.00126, −1.0 → 0.0030. Even *fully zeroing* 128 neurons barely moves the
   next-token distribution.

The static edit that actually fixes the loop is a different kind of operation: it
removes a **direction** from the neurons' write columns at strength 6, `W[:,j] −=
6·(W[:,j]·d̂)·d̂`, over 64 neurons in L15 and 64 in L19.

## Decision

Use the known fix as the **lever**, and let the glial learn only **when** to pull it:

```
out[t] += α(t) · ( act[t, idx] · D[:, idx]ᵀ )
```

with `D = W_edited − W_base` on the edited `down_proj` columns (or our own verified
equivalent `D[:,j] = −6·c_j·d̂`). α(t) is one scalar per token, produced by the
slow-state reader.

**Sign convention:** `D` is the *change* the authors made, so it already carries the
sign of the edit (the edit removes the loop direction, so `D` is negative along `d̂`).
The correction is therefore **added** to the layer output. Subtracting it applies the
loop direction twice — a bug we hit and fixed (it showed up as a ~16-logit mismatch
against the edited checkpoint, i.e. an anti-fix).

Properties that make this the right shape for the experiment:

- **α = 0 is the base model** (the loop). **α = 1 is the verified fix** (closed-thinking
  11% → 83%). So the lever's authority is exactly the authority of a fix we already
  measured, not a guess.
- The learned quantity is a single scalar per token — small, interpretable, and hard to
  saturate into nonsense the way a 128-dim gain vector did.
- Selectivity becomes meaningful: a good glial raises α during a loop and leaves it near
  0 elsewhere, which is exactly the "regulator" claim.

## Consequences

- The glial's read must be **local in time** (per token or per chunk), not a
  document-level mean — the document-level mean was 99.8% identical across documents.
  This is the substantive redesign; the previous implementation's pooling was the
  reason it could not be context-dependent.
- The objective must be one the mechanism can actually move: with a lever of known
  authority, the gradient of "does the model close its thinking?" with respect to α is
  meaningful, whereas with a 0.003-KL lever it was not.
- Capability preservation must be checked explicitly, because α = 1 changes behaviour
  globally; a regulator that only ever chooses α = 1 is a static edit in disguise. That
  is why selectivity and knockout are part of the pre-registered gates.

## Verification status

- Diff census: the authors' checkpoint differs from the base in **exactly 2 tensors**
  (`model.layers.15.mlp.down_proj.weight`, `model.layers.19.mlp.down_proj.weight`,
  ~64 changed columns each) — so a down_proj-only lever can reproduce the whole edit.
- Our independently diagnosed neuron set overlaps the authors' edited columns
  **62/64 (L15) and 63/64 (L19)**.
- Logit equivalence at α = 1 and the behavioural comparison are running
  (`kernel_arms`, `glia-lever-arms`).
