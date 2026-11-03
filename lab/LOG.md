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
