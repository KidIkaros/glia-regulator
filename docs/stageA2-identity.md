# Stage A2 — the identity foothold

## Why this comes now

We built and validated a control system (the lever, dose-response, alpha*=1.25) for a
mechanism we have never characterised. We know the 128 edited neurons' GEOMETRY (64
columns each in L15/L19 down_proj, one direction per layer at min cos 0.999, strength 6)
and their CAUSAL effect (removing it closes the loop), but not their FUNCTION: what
feature they respond to, what the loop direction encodes, or why removing it breaks
semantic re-derivation.

Three failed measurements (gate saturation; zeroing all 128 neurons -> ~0.003 KL; the
projection detector AUC 0.688/0.312) all failed to find the loop in the activations.
The reading they converge on: the loop lives in how downstream weights read these
neurons - in the direction, not in the units. So the identity question is:

    what does the loop direction encode?

## The toolkit (the links that were shared for exactly this)

`stdlogits/neuron-lens` reads every MLP neuron of Qwen3-8B three ways:

* **R-lens** - forward Jacobian under RelP rules, read through W_U: what a direction
  makes the model say LATER (the write side);
* **S-lens** - the same estimator backwards, read through W_E: which input tokens put a
  direction there (the input side);
* activations over real corpora + exact token drivers + labels.

Their findings calibrate our expectations for L15/L19 of 28 (mid-to-late depth):

* the R-lens read of a neuron's WRITE predicts its steering effect from L5 onward
  (92-100% of random poles) - so the R-lens read of d is the most trustworthy read we
  can make;
* the S-lens recovers token detectors only in early layers (L0-8) and "L13-L20 is the
  weakest stretch for every weight-only input read" - so an S-lens read of d may come
  back weak, and that is a known limitation of the method, not a failure of ours;
* finding 4 is the warning that matters for Stage B: what a mid-network neuron does in
  ONE context is mostly specific to that context - no single vector carries it. Our
  direction is not a neuron but the aggregate write of 64 of them with a verified causal
  effect, so it has a better chance than a random mid-network pole; but the same finding
  says the reader for Stage B should expect context-specificity, not a single feature.

`logitsml/flyweight` is the identity STANDARD, not a tool for us: every one of its
166,700 fly neurons has a name (P9, DNa02, the giant fiber), a checked role, and a
wiring audit before any of it is used. Our neurons have no names. The deliverable of
this stage is names.

## What we build

1. **Fit both lenses for Qwen3.5-0.8B** (no released lens covers it; their
   `02_fit.py` already handles Qwen3.5's DeltaNet layers, and the accumulator for a
   0.8B/28-layer model is ~60 MB - it fits on one T4 with room to spare).
2. **Read the loop direction d through the R-lens** (L15 and L19 separately): the token
   ranking of what the direction promotes and suppresses. This is the write-side read,
   and our most legible one.
3. **Read d through the S-lens**: which input tokens' footprints put the loop direction
   there. Expect weakness at our depth; a weak read is a recorded negative, not a
   silence.
4. **Read each of the 128 neurons' writes** through the R-lens: do the individual
   neurons agree with d, or does a subset carry the semantics? This is where names come
   from.
5. **Within-context fork**: run base (loops) and alpha=1.25 (closes) on the same
   prompts; log the 128 neurons' activations at MATCHED prefixes. Where the two
   trajectories diverge in these channels is where the loop is decided - and it is also
   the Stage-B reader's candidate input.
6. **Token drivers for d**: sweep common tokens through real documents, measure which
   ones drive the direction directly.

## Deliverable

Names. A table of 128 rows plus 2 direction rows: lens reads in words, top firings with
their contexts, drivers, and for the direction the answer to "what does removing it
stop the model from saying". If the reads come back weak, the deliverable is the
calibrated negative and the fallback design (behavioural/weight-side control) is
written down with the evidence.

## Cost

One Colab T4 session: lens fits ~30-60 min, reads and drivers ~20 min, the fork pass
reuses the sweep's paired prompts ~30 min. CPU cannot do it (the OOM history above;
also speed: 18 x 128 tokens took ~15 min on CPU vs ~1 min on a T4).

## Relationship to everything else

* Stage B (training the reader) is BLOCKED on step 5: the reader's input features must
  come from the generated trajectory, and this is the experiment that finds whether the
  128 channels carry the decision.
* The capability panel is unaffected.
* The trajectory-feature run proposed earlier is folded INTO step 5 (same rollouts now
  store their text and the activations), so it is not a separate experiment any more.


## Related work watch (2026-11-04)

**qwentensor** (Bobliuuu/qwentensor, 9 hours old, built with Claude on qwenTPU):
compiles a quantized (5-bit) PyTorch network into a prompt-written program that runs
INSIDE Qwen3.5 - DeltaNet rows as RAM, tokens as the trace, host only runs argmax;
demonstrated on 0.8B (our exact model), 2B, 4B; bit-exact vs a numpy golden on 540
sklearn digits; real-model runs on A100 for the unpaged nets (paged/weight-streaming is
CPU-simulator only).

Relevance to us - honest split:
* NOT a read tool. It does not advance the identity foothold; Stage A2 proceeds.
* It IS the proof-of-concept for our weight-side fallback: if Stage A2 finds no
  activation-level signature, the fallback is control that lives in the weights - and
  qwentensor demonstrates arbitrary integer circuits executing inside this model family.
  The glial taken literally: the regulator (alpha(t)) compiled to run in-model, the model
  carrying its own loop detector instead of an external hook.
* Mechanical reference: "DeltaNet rows as RAM" documents the exact machinery in our
  target layers, already bit-exactly manipulated on 0.8B by someone else.
* Limits to respect: 5-bit ints, ~45-column machines, a page of a small MLP costs ~142
  tokens; a reader consuming 128 float channels is a major lift. Endgame bookmark, not a
  Stage-B path.
* Ecosystem signal: the Qwen3.5 weight-space scene is moving fast (WSP paper Nov 2,
  this Nov 4). Timing of our write-up matters.
