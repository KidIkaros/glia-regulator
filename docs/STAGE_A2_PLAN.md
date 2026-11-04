# Stage A2 execution plan (the calm version)

Written 2026-11-04 after the slowdown request. Nothing launches without a line in this
file. Facts marked [V] are verified by a tool result in this session; anything else is
an estimate and says so.

FRAMING (see docs/CONTEXT.md): Stage A2 is not an interpretability milestone - it is
METHOD VALIDATION for the Companion program. The 128 loop neurons are the clean testbed
(a causally verified edit) on which the find -> locate -> regulate chain is proven
before it is pointed at personality traits, which are messier: distributed, and not
cleanly editable. If the method cannot be validated here, personality work needs a
different instrument, and the program should know that now.

## Where we actually are

Established (earlier phases, all committed):

* The lever works: alpha=1.25 closes 8/8 problems including both healthy controls, at
  KL 0.0077 and perplexity +4.6% against base. alpha=1.0 is the harmful dose (breaks a
  healthy problem). [V - sweep_results.json]
* The missing piece is identity: we know the 128 neurons' geometry and causal effect,
  not their function. Stage A2 exists to fix that.
* The identity toolkit (neuron-lens) cannot run on any GPU we can reach for free:
  its Triton kernels need 96 KB shared memory, sm_75 caps at 64 KB. Verified failures:
  Colab T4 (2x InductorError, then OutOfResources), local CPU (2 OOM kills, 1 timeout
  that did not finish 2 documents in 40 min). Kaggle T4/P100 are sm_75/sm_60 - same
  verdict by architecture, no run needed. The only qualifying machine on this account
  is the Modal H200 (sm_90). [V - logs + dmesg + colab/H100 entitlement refusal]

## The phases

| # | what | hardware | cost | output | decides |
|---|------|----------|------|--------|---------|
| 0 | fork: paired rollouts (base vs alpha=1.25), storing text, close position, and per-token activations of the 128 neurons | Colab T4 (free, already approved) | ~2 h wall, chunks of 2 problems | fork.jsonl + npz activations | does the 128-channel space separate loop from fix trajectories? = Stage-B reader's input features |
| 1 | lens fits R + S for Qwen3.5-0.8B, 200 docs each | Modal H200 (PENDING APPROVAL) | <= 45 min GPU bounded by function timeout; fits are resumable on a Volume | two .pt lens fits on the volume | nothing by itself - it is the substrate for phase 2 |
| 2 | reads: d through R and S; each of the 128 neurons' writes through R; cosine with d | same H200 session (minutes) or local CPU once fits exist | minutes | reads.json: token rankings per direction/neuron | which subset of the 64 carries the loop semantics - the names |
| 3 | analysis + writeup | local | none | names, or the calibrated negative + fallback design | Stage B design; whether single-neuron regulation is meaningful or collapses into direction-composition control |

Kill criteria (written now, before any result):
* if phase 1 fails on H200 for a NEW reason after the smoke pass, Stage A2 is paused and
  the fallback (patching their Triton kernels for sm_75, or weight-side control without
  lens reads) becomes the plan - not another hardware hunt.
* if phase 2's reads come back weak at our depth (their findings predict this is
  possible: "L13-L20 is the weakest stretch for every weight-only input read"), that is
  a recorded negative, and the fallback is written down with the evidence.

## The H200 ask, precisely

One Modal app, one function, two fits:

    modal run scripts/modal_lens_fit.py --smoke     # ~10 min bound, --docs 2, validates the whole path
    modal run scripts/modal_lens_fit.py             # the real thing: R then S, 200 docs, <= 45 min hard timeout

* every intermediate is persisted on a Modal Volume (corpus, fit checkpoints) - a kill
  or a crash loses nothing; a re-run resumes;
* the script is vetted line-by-line in scripts/modal_lens_fit.py's docstring (VET
  section), including: exact pip pins, GPU string, timeout, volume mounts, revision
  handling of the base model, and what happens on each failure;
* cost: the allowance is 6 h/month with none used; smoke + full <= ~1 h worst case
  (< 20% of allowance). [V - allowance fact from memory; GPU-hour pricing NOT asserted]
* what I will do first if approved: run --smoke only, report, then run the full fit on
  your confirmation. Two approvals, not one blanket.

## Threads deliberately NOT in this plan

* H-Net AUDIO_PHASE - runs Sunday on the Kaggle quota reset, already designed in
  hnet-kaggle. Unaffected by anything here.
* qwentensor - bookmarked as the weight-side fallback proof-of-concept (docs/
  stageA2-identity.md related-work note). No work scheduled.
* H-Net chunker identity transplant (neuron-lens on the router) - a follow-on that
  reuses phase 1-2's fits only if the lens methodology proves out on our model first.
