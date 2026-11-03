# Neuron identification for Qwen3.5-0.8B — research synthesis

Date: 2026-11-03. Sources: three parallel research agents + primary sources (all URLs inline).
Files: `qwen35-sae-findings.md` (SAE release), `neuronary_methodology_findings.md` (push-search
methodology) in this directory.

## TL;DR for our project

1. **Identification = gradient attribution search** (justinrhowell/neuron-steering, MIT):
   one backward pass ranks ALL 86k neurons/layer by their first-order push effect on
   log p(trait answers) − log p(plain answers). Verified: the persona neuron that
   activation-based ranking placed at ~19,000th ranked **42nd of 193,536** with this
   method. ~100 lines of MIT code, ~7–10 min per trait on a 4B; less on our 0.8B.
2. **SAEs exist for our exact model**: layers 5/11/17 of the INSTRUCT Qwen3.5-0.8B
   (Matryoshka BatchTopK, d_sae 65,536, attached to resid_post). Usable post-hoc to
   LABEL what the glial regulates (feature decoder → down_proj attribution recipe below).
   Not usable for candidate selection (see the negative result below).
3. **Candidate selection by activation statistics is a proven failure mode**: the
   hostile-persona neuron ranked ~19,000th by firing-contrast and ~14,000th by
   write-alignment; top-activating neurons are word-inserters, not behavior switches.
   My original "activation-diff candidate filter" idea is dead. Use attribution search.
4. **SAEs are not needed for candidate selection either** — the working pipelines both
   use pushes + gradient attribution, no SAEs. SAEs are for interpretation afterward.
5. **Lens port (R/S Jacobians) is NOT worth it for stage 1**: the attribution search
   gives the causal ranking directly and cheaper. Revisit only if we want to label
   neurons in words (and SAE feature explanations may suffice for that).

## Thread 1 — the SAE release (verified from the repo itself)

Repo: https://huggingface.co/decoderesearch/qwen-3.5-saes (no license declared —
research use only). Layers **5, 11, 17**, in two variants:
- `qwen-3.5-0.8b/` = trained on **Qwen/Qwen3.5-0.8B (the instruct/chat model)** with
  lmsys-chat-1m + chat formatting → **this is the one for our host model**.
- `qwen-3.5-0.8b-base/` = Qwen/Qwen3.5-0.8B-Base on Pile.
- (4B variants cover layers 3/15/27.)

Architecture `btk-mat-layer-N-k-100` = **Matryoshka BatchTopK SAE**: `d_in=1024`,
`d_sae=65,536`, matryoshka widths [2048, 16384, 65536], k=100 (training-only; at
inference a per-feature threshold vector applies — measured constant 0.0491 across all
features). Hook point: **resid_post after layer N** (`hook_name: "model.layers.N"`).
Tensors (F32): `W_enc [1024, 65536]`, `W_dec [65536, 1024]`, `b_enc [65536]` (unused at
inference), `b_dec [1024]`, `threshold [65536]`. File ≈537 MB.

Load-and-use recipe (from qwen35-sae-findings.md):
```
pre = (x - b_dec) @ W_enc          # x = resid_post at layer N, [T, 1024]
f   = relu(pre) * (pre > threshold)
write_direction(feature i) = W_dec[i]        # [1024] residual-space direction
neuron_attribution(feature i) = W_dec[i] @ down_proj.weight[:, j]   # [3584] per-neuron
```
Caveat: resid_post features can also write via attention, so the down_proj attribution
is the MLP-shared fraction, not the whole effect.

Neuronpedia hosts only the instruct model, only layer 11 (`11-res-matryoshka-65k`), with
`neuron_alignment_{indices,values,l1}` per feature and auto-labels (gemini-2.5-flash-lite)
via POST `/api/explanation/search-model`. Feature endpoint:
`GET /api/feature/qwen3.5-0.8b/11-res-matryoshka-65k/{index}`.

## Thread 2 — the push-search methodology (the part to replicate)

Both working pipelines avoid priors entirely:

**Pipeline A — brute sweep** (neuron-explorer): push every neuron up AND down
(~10 rollouts each), rank by words distinctive to that unit's rollouts. Expensive;
produced the negative result above.

**Pipeline B — gradient attribution search** (justinrhowell/neuron-steering, MIT):
```
1. calibration: r_L = median over fixed calibration text of ||resid leaving layer L||
   per token, SKIPPING token 0 (attention sink inflates its norm)   [4B: 24.8 at L22]
2. attribution: one backward pass of
       log p(trait answers) - log p(plain answers)
   w.r.t. each neuron's push direction; rank all neurons        [hostile neuron: 42/193536]
3. screen: steer top-60 at dose d, score trait replies          [~7-10 min per trait, 4B]
4. validate: best-3 neurons, pre-registered held-out tests, 3-seed random same-layer
   controls at the same dose, sign-flip arm, dose ladder, blind judge
```
Push mechanics: pre-hook on `mlp.down_proj` INPUT —
`act[n] += dose * r_L / ||W_down[:, n]||` on **every token, prefill + generation**.
Doses add across neurons; fluent zone ≈ 0.5–0.8 (dose 1 ≈ a full residual push).
"Output changed" is NOT evidence — random pushes garble 16/24 replies while scoring
trait 0/24; the control arms are what make a recipe real.

**Eval harness to reuse: shayansanjideh/neuron-lab** — the whole pipeline already ran on
**free Kaggle T4s** (fp32, 2×T4): batch notebooks + blind-judge harness + Fisher-exact
follow-ups. Also: `lyramakesmusic/neuronary` (recipe schema, stdlib Python client,
`GET https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes`).

**Neuronary coverage**: 261 recipes — 4B: 247, 27B: 10, **0.8B: exactly 1** (layer 0,
score 0, unreproduced). A 0.8B recipe catalog would be novel. Reproducibility caveat:
only 1 of the 4B author's neurons reproduced cleanly standalone — expect mix-dependent
candidates.

## Thread 3 — lenses (partial; agent was stopped for running locally)

From the neuron-lens README (fetched earlier, no execution needed): R-lens = RelP-
approximated Jacobian of the final residual w.r.t. layer-L residual, averaged over
corpus + later positions → predicts steering (r > 0.9 from L22; finding 3). S-lens =
backward to input embeddings ("which tokens put this there"). Finding 4: lens/steering
agree because both average over contexts — per-context effects differ (r 0.05–0.11
mid-network). Port cost to our hybrid model: unresolved (agent was stopped); assume a
day of work + GPU. Decision: **deferred** — attribution search covers stage 1 without it.

## Updated Stage 1 protocol (replaces the candidate-filter section in DESIGN.md)

```
1. calibration text (fixed) -> r_L for target layers              [CPU/free]
2. attribution search per behavior (language drift, ALL CAPS):
   one backward pass each -> ranked neuron lists                  [GPU ~10 min/behavior]
3. push screen: top-40 neurons at dose 0.6 and 0.8, 20 held-out
   behavior prompts + 6 ordinary prompts, batched generation      [GPU ~1.5h]
4. score: objective string checks (language ID / case) -> paired
   Fisher exact vs unsteered + vs 3 random-neuron controls        [CPU/free]
5. side-effect panel: ppl delta + language-mixing rate on a small
   corpus (neuronary's warning system)                            [GPU ~15 min]
output: verified 0.8B recipes (neuronary format) for 2 behaviors
```

Then Stage 2 unchanged: glial field over the verified neurons, trained context-dependently,
knockout + interference tests (see DESIGN.md).
