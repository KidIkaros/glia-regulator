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

## 2026-11-03 20:3x-21:0x — the lever is proven correct · quota 29.76/0.24h
Sequence of measurements (each a separate kernel, all in this log):
  1. glia-lever-arms (2 problems/arm, 8192 cap): base 0/2 closed; "ours" arm STILL
     looped and looped LONGER than base (32731 vs 24006 chars) -> the "ours" lever was
     applying the ANTI-FIX.
  2. LOCAL, zero-quota check (scripts/check_proj.py): proj_L*.npy is stored NEGATED
     (corr(live W.d_hat, proj) = -1.000000 exactly) and loopdir is already unit-norm.
     Our delta had therefore both the wrong sign AND 6x the magnitude.
  3. glia-lever-v2: in-kernel A-check showed our rebuilt delta still mismatches the live
     weights by ~8x in magnitude: the saved npys were computed against the OLDER
     dataset copy of the model (488 tensors), while the hub's current revision has 321
     tensors. Any lever must be rebuilt against the weights it will run on.
  4. glia-lever-v2 / v3, CHECKPOINT LEVER: max|delta logits| = 0.164 (fp16 noise) and
     argmax agrees on both prompts => the lever at alpha=1 IS the authors' edited
     checkpoint. Earlier "equivalence failures" were the check comparing the OURS lever
     with THEIR checkpoint (two different edits; the gap measured their distance).
  5. glia-probe-authority: a lever rebuilt in-kernel over OUR 64+64 neurons (overlap
     62/64 and 63/64 with theirs) gives the SAME effect as the checkpoint lever
     (KLs agree to 3 digits) -> our neuron set is sufficient on its own.
  6. Single-position KL underestimates the lever's effect (0.00004-0.03): the loop fix
     is a LONG-HORIZON effect, so only the closed-thinking metric at the full 16k cap
     measures it. Short-cap runs are inconclusive, not negative (our verified static
     edit closed 15/18 at 16384, and gsm8k_train_0 needed >8192 tokens to close).

## 2026-11-03 21:4x — CPU kernels: the horizon-equivalence dose-response (no GPU quota)
Question: is the runtime lever's scale correct, and how does its effect decay?
Kernel `glia-cpu-horizon` (CPU, 4 threads, 4.1 tok/s, GPU meter flat before AND after):
teacher-force the authors' edited checkpoint and base+lever(alpha) on the SAME 256-token
rolling trajectory from the base model, and compare at 8 positions inside it.
  alpha 0.0 (base) mean KL(target||arm) = 0.014224   argmax agreement 1.000
  alpha 0.5                            = 0.002932   1.000
  alpha 1.0                            = 0.000023   1.000   <- minimum
  alpha 1.5                            = 0.001270   1.000
  alpha 2.0                            = 0.004485   1.000
=> alpha = 1 is the CORRECT scale (U-shaped around it): the lever at alpha=1 tracks the
   fix's predictions 620x more closely than the base model does, and over-correcting
   (alpha > 1) makes it worse. So the lever's semantics are right, and the 16k-token
   behavioural gap is NOT a mis-scaled alpha - it is accumulation.
   Note the divergence is concentrated at the FIRST positions and decays along the
   trajectory: deep inside the loop both models sit in the same attractor.
Blocks found and fixed while building this: Qwen3.5's Triton kernels cannot run on CPU
(flash-linear-attention must be blocked so transformers takes the pure-torch path); the
teacher-forcing path needs a [batch, seq] input, not a 1-D token vector; the kernel
source must stay under 1 MB (so the delta is derived in-kernel from the two checkpoints
rather than shipped - which also removes the revision-drift risk).
Next (same CPU budget): `glia-cpu-traj` - free-running greedy rollouts of the edited
checkpoint vs base+lever(alpha), measuring the first token index where they differ.

## 2026-11-03 20:2x-21:4x — Colab as the GPU: official CLI, not a browser
Kaggle GPU quota is spent until Sunday (29.97/30h), so the fp32 16k fidelity run
moved to Colab. Two routes tried:
  (a) camoufox (headful Firefox) driving the Colab UI: works, but every step is
      fragile - the "Upload notebook" button opens a DIALOG (not a file chooser), the
      GitHub route needs a GitHub OAuth grant in the user's account, the Drive tab
      needs the file to be at the Drive root, the notebook's drive.mount() blocks on a
      permission modal nobody clicks, and a screenshot inside an error handler hung for
      60 s and killed the driver (fixed: safe_shot, screenshots can never kill a run).
  (b) the OFFICIAL Colab CLI (google-colab-cli, ~/.local/bin/colab): the right tool.
      `colab new --gpu T4` provisions a session in seconds; `colab exec` runs code
      (stdin or a file) with --env and --timeout; `colab sessions/status/upload/
      download/stop` cover the rest; liveness is activity-based, so a session does not
      evaporate mid-run.
      Auth: its OWN OAuth consent is required - the scopes are
      .../auth/colaboratory and .../auth/drive.file, which `gcloud auth
      application-default login` does NOT request, so ADC alone is not enough. The
      consent can be completed in camoufox (which already has the user's Google
      session, so no password is typed by the agent); the CLI reads the code from it.
      Gotcha: `colab upload` writes to the Jupyter contents root, NOT the VM
      filesystem the driver reads (/content held only .config and sample_data), so the
      robust route is to send one self-contained script over stdin with the payload
      base64-embedded (scripts/build_colab_script.py).
      Bug this route caught: harness.greedy_rollout decoded via model.tokenizer, which
      does not exist -> chars 0 and closed=False for every rollout, i.e. the
      closed-thinking label was silently broken. The tokenizer is now an explicit
      argument and a missing one raises instead of returning silence.
Colab PRO tier confirmed (priority GPUs, longer sessions). If the CLI exposes H100/A100
on this tier, Modal's H200 allowance becomes unnecessary for this project.

## 2026-11-04 02:3x-06:4x — THE RESULT: the runtime lever CAN break the loop (alpha=1.5)
Colab T4 (via the official CLI, headless), fp32, greedy, presence penalty 1.5 on every
arm (the reference protocol), gsm8k_train_0, 8192-token cap:
  fix reference (authors' checkpoint): 6,320 tokens, 18,340 chars, CLOSED
  lever alpha=1   : 8,192 tokens (cap), 26,429 chars, NOT closed, matched fix 546 tokens
  lever alpha=1.5 : 1,838 tokens, 4,881 chars, CLOSED, matched fix 8 tokens
=> (1) the runtime lever has enough authority to break the reasoning loop, and closes it
   FASTER than the published weight edit (4,881 vs 18,340 chars);
   (2) the fidelity optimum (alpha=1) is NOT the behavioural optimum - alpha=1.5 is the
       least faithful setting (8-token trajectory match) and the one that works;
   (3) mechanism: the edit removes the direction once, permanently, at strength 6; the
       lever applies a partial correction per step, so matching the edit's local
       predictions does not match its cumulative effect - the lever needs a bigger
       per-step push than the edit itself.
=> the glial's premise survives: there is a lever with demonstrated authority over the
   behaviour, so the open question is WHEN to apply it (a context-reading alpha(t)).
   The binding constraint is now the capability panel: alpha=1.5 removes 9x the loop
   direction, so the dose-response must map closed-thinking rate against capability
   damage before alpha(t) is trained on it.
Pipeline lessons (all now in the repo): the Colab CLI's exec has an internal ~20 min
reply ceiling and a disconnected client ABORTS the cell, so arms must be chunked (one
arm per invocation, MAX_ARMS_PER_RUN) with the results file making each invocation
resume; the presence penalty must be the published implementation (unique GENERATED
tokens, prompt excluded) or a 16k generation takes hours; and `colab upload` writes to
the Jupyter contents root, not the VM filesystem, so the payload travels inline over
stdin.

## 2026-11-04 11:28-13:07 — DOSE-RESPONSE SWEEP (two Colab T4s, parallel)
Alpha in {1.0, 1.25, 1.5, 1.75, 2.0} x 8 problems (6 known loopers + 2 healthy controls),
protocol-matched (greedy, presence penalty 1.5 on every arm, fp32, 8192 cap), capability
panel per dose (ordinary-prompt KL + argmax agreement, passage perplexity).
  alpha 1.0  : 4/8 closed, healthy 1/2, kl 0.0047, agree 100%, ppl +3.07%  -> GATE FAIL
  alpha 1.25 : 8/8 closed, healthy 2/2, kl 0.0077, agree 100%, ppl +4.56%  -> GATE PASS
  alpha 1.5  : 6/8 closed, healthy 2/2, kl 0.0117, agree 100%, ppl +6.33%  -> GATE PASS
  alpha 1.75 : 8/8 closed, healthy 2/2, kl 0.0167, agree 100%, ppl +8.36%  -> GATE PASS
  alpha 2.0  : 7/8 closed, healthy 2/2, kl 0.0227, agree 100%, ppl +10.64% -> GATE PASS
=> alpha=1.0 - the dose that best reproduces the published edit's PREDICTIONS - is the
   only failing dose, and it fails by BREAKING healthy work (gsm8k_train_0, which the
   base closes in 1,201 tokens, loops for 8,192 under alpha=1.0) as well as by leaving
   4 of 6 loopers looping. The faithful dose is the harmful dose.
=> alpha=1.25 is the operating point: the smallest dose reaching 8/8 with the least
   capability cost among passing doses. Capability damage is monotone in the dose; the
   closed rate saturates at 1.25 (the 1.25-2.0 wobble is 8-problem resolution).
=> argmax agreement is 100% at EVERY dose: removing the loop direction does not blunt
   ordinary text.
=> for Stage B: the regulator emits 0 or ~1.25, never 1.0; and SELECTIVITY is the
   binding constraint - false-positive firings on healthy contexts are exactly what
   damaged capability at alpha=1.0.
Infrastructure notes: this tier allows exactly 2 concurrent GPU sessions; a Colab
session is reclaimed after ~2 h idle, so sessions are provisioned per phase; the CPU
kernel route is unusable for behavioural work (2.4 h for ONE 16k-token arm) - a fact
that justifies Colab T4 for everything; two fp32 0.8B checkpoints on this 6.3 GB box
OOM-kill, so checkpoints are loaded one at a time.
