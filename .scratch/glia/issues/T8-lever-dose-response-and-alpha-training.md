"""T8 — measure the runtime lever's OWN dose-response, then train alpha(t)

Status: pending (needs the Sunday quota refresh, 2026-11-07)

Why this exists. The checkpoint lever is proven logit-equivalent to the authors' edited
checkpoint (max |delta logits| = 0.164, argmax agrees), but at the full 16k cap it did
NOT close problem gsm8k_train_0 while the static edit closed it after 4,555 characters.
The cause is arithmetic, not algebra: two fp16 matmuls plus an add versus one matmul, and
the reasoning loop is chaotic over thousands of steps. So alpha = 1 is NOT "the fix" --
the lever's own behaviour must be measured before alpha(t) can be calibrated to it.

Task 8.1 — dose-response (GPU, ~30 min, 1 session)
  For alpha in {0, 0.5, 1.0, 1.5, 2.0}: generate on 4 paired problems at the full 16384
  cap with the checkpoint lever and record closed-thinking rate + generation length.
  Cost estimate: base-equivalent arms run to the cap (~13 min/problem on a T4); if the
  lever closes early, arms are ~2-4 min each. Use the session deadline guard (3300 s)
  and write results incrementally, so a timeout still yields the arms that finished.
  PASS CONDITION: at least one alpha in {0.5, 1.0, 1.5, 2.0} closes >= 3/4 problems.
  If none does, the runtime lever is not a viable fix path and the experiment must fall
  back to the weight-space form (patch down_proj.weight in place, which forecloses a
  per-token alpha) -- record that and stop.

Task 8.2 — train alpha(t) (GPU, ~45 min)
  Only after 8.1. A context-LOCAL reader (per-token residual at layers 4/8/12, NOT the
  document mean, which measured 99.8% identical across documents) emits one scalar per
  token; the lever applies it. Train against the healthy completions' NLL, with the
  capability guard strengthened beyond a single-sentence KL.
  Pre-registered gates: closed-thinking >= 70%, knockout (alpha forced to 0) closed
  rate <= 25%, selectivity: per-prompt mean alpha high on looping contexts and low on
  healthy ones.

Task 8.3 — the interference matrix (syco head)
  Only after 8.2 succeeds. A second behaviour head on the same field, with the
  cross-head interference matrix as the headline number.
"""
