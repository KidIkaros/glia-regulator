# Lab log

## 2026-11-03 15:2x — Session A prep (glia-attr-v1) · pre-launch · quota 26.36/3.64h
Intent: validate attribution search against the 128 known loop neurons (GATE 1: >=3 of
top-10 across L15+L19), then run the language-drift push screen (24 candidates, dose 0.8,
20 held-out drift prompts + 4 ordinary, Fisher p<0.001/40 + 3 random controls, ppl panel).
Pre-registered bands: recipe qualifies on objective string scoring; kill rule = GATE 1
fails twice -> fall back to known neurons, postpone drift recipe.

## 2026-11-03 15:5x — glia-attr-v1 FAILED+KILLED · deviation logged
v1 autograd construction used `gains*0` to attach the graph -> exact-zero gradient ->
all scores zero -> arbitrary top-10. GATE 1's 5/20 overlap was coincidental (known L15
neurons start at low indices). Killed the session; fixed: gains perturb the scale
vector directly. Repushing as v3. Cost of the mistake: ~0.16h quota.

## 2026-11-03 15:5x→16:1x — glia-attr-v1 (v1-v2) · GATE 1 GENUINE FAIL · quota 26.36→26.89h
Sequence: v1 broken autograd (gains*0 → zero grads, arbitrary topk, killed), v2 fixed
construction (fp32 driver + additive push + mean-centered objective). Pipeline then ran
CORRECTLY (finite, structured grads: 512/3584 nonzero in toy replication; in-kernel
attribution completed in ~10s) — and GATE 1 genuinely FAILED: attribution top-10
overlap with the wsp-verified loop neurons = 2/10 (L15) + 0/10 (L19); finite-difference
re-check: 1/10 + 0/10.
DECISION (per plan kill rule): identification-by-attribution does not transfer to the
loop behavior. Recorded as a genuine negative result with analysis: the loop is an
emergent long-generation behavior; first-order next-token sensitivity does not recover
its circuit (consistent with neuron-lens finding 4). Falling back to the KNOWN 128
neurons for glial Stage A. Language-drift recipe postponed to next week's quota.
Raw artifacts: kernel log archived; gate1.json not retrieved (kernel ERROR, outputs
likely still retrievable via kernels output before any overwrite).

## 2026-11-03 16:2x — Session B prep (glia-stageA-v1) · pre-launch · quota 26.89/3.11h
Intent per kill-rule fallback: glial Stage A with the KNOWN 128 loop neurons (attribution
identification failed — recorded as negative result). Train loop head only (3 epochs over
18 paired docs, segment 512, fp16, capability-KL as post-hoc spot-check), then eval the
18 paired problems at paper sampling, selectivity via per-prompt mean |gain|, knockout on
6 problems (expect collapse to <=25% closed). Pre-registered bands: SUCCESS = closed
>= 70% (i.e. >=70% of static edit's 83% effect), knockout <= 25% closed.

## 2026-11-03 16:4x — Session B v2 launch (glia-stageA-v1 v2) · quota 27.07/2.93h
Ticket T5. Fix: payload-path lookup moved to the payload ROOT (v1 died on
FileNotFoundError s1_pairs.json). Added tests/test_payload_layout.py, proven RED by
reverting the fix and GREEN after restore — this failure class can no longer reach a
GPU session. Tickets T1-T7 and ADR-0001 committed (git 10b0c6f).

## 2026-11-03 18:2x — glia-stageA v2 (anti-saturation attempt) · quota 28.19/1.81h
Optimization fix (sigma 0.5, head temperature 2.0, normalized z, lr 1e-3, clip 0.5,
target = healthy completion's last 6000 chars incl. thinking tail, 6 epochs).
Result: GATE 2 identity PASS; loss moved slightly (last-doc 79.33 -> 74.50) but the
first-doc loss stayed EXACTLY constant (55.440 every epoch); mean|gain| climbed
0.129 -> 0.4957 and pinned at the bound (sigma=0.50) by epoch 5; SATURATION GUARD
fired and skipped the eval (quota preserved, ~2 min cost).
DIAGNOSIS (to be tested): the glial reads MEAN-POOLED residuals over a whole document;
those pooled vectors are nearly identical across documents (same prompt shape, same
model), so the reader has almost no context to be context-dependent WITH -> the head
saturates at a constant, context-BLIND gate. Fix is architectural (chunk-local
context -> per-chunk gains), not another knob.

## 2026-11-03 19:0x — diagnostics + lever redesign · quota 28.25/1.75h
DIAGNOSTIC (kernel glia-diag-v1, 3 pushes, ~0.1h total):
  Q1  pooled residuals (L4/8/12) pairwise cosine across 12 documents = 0.9231
      (last-64-token pooling 0.9099) -> the read signal barely varies by document.
  Q1b READER OUTPUT z pairwise cosine across documents = 0.9984 -> the glial was
      context-blind BY CONSTRUCTION; the head saturated because the optimum is a
      constant gate, and a constant gate degenerates the model.
  Q2  AUTHORITY of an activation gate on the 128 verified loop neurons:
      gain +0.25 -> KL 0.00051; +0.5 -> 0.00251; -0.5 -> 0.00126; -1.0 -> 0.0030.
      Even fully zeroing 128 neurons barely moves the next-token distribution.
      => the activation-gate lever is structurally too weak.
DESIGN CONSEQUENCE (ADR-0002): use the authors' EXACT edit as the lever and learn only
its per-token strength. DeltaLever: out -= alpha(t) * (act[t,idx] @ D[:,idx]^T), where
D = W_edited - W_base for the edited down_proj columns. alpha=0 -> base (the loop),
alpha=1 -> numerically the authors' edited checkpoint (closed-thinking 11% -> 83%).
The glial's ONLY learned quantity is WHEN to apply the fix (one scalar per token).
Kernel glia-lever-check pushed: (A) logit equivalence alpha=1 vs edited checkpoint,
(B) authority curve KL(alpha), (C) closed-thinking at alpha=0 vs 1 on 6 problems.

## 2026-11-03 19:5x — sign bug in the lever; Kaggle OAuth refresh token expired
- Diff census (kernel glia-diff-census): the authors' checkpoint differs from the base
  in EXACTLY 2 tensors (layers.15/19.mlp.down_proj.weight, ~64 columns each). Our
  independently diagnosed neurons overlap their edited columns 62/64 (L15) and 63/64
  (L19) -> an independent replication of their neuron set.
- Logit equivalence FAILED twice: first because the kernel ran the checkpoint delta at
  alpha=6 (the delta already encodes the full 6x edit -> applied 7x over); then, after
  fixing that, because the lever SUBTRACTED D = W_edited - W_base while D already
  carries the edit's sign (the edit REMOVES the loop direction) -> the lever applied
  the loop direction twice. Fixed: the correction is ADDED (src/lever.py).
- Kaggle CLI stopped authenticating mid-session: the OAuth access token expired
  (23:54Z) and the stored refresh token no longer refreshes (401 on every call).
  scripts/kaggle_refresh.py + scripts/kg.py written; a browser re-login is needed.
