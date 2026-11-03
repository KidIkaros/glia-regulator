# ADR-0001 — Fall back to verified neurons; attribution search does not transfer
Date: 2026-11-03 · Status: accepted

Context. The Neuron Zoo method ranks neurons by gradient attribution on push effects
(justinrhowell/neuron-steering; their target neuron ranked 42/193,536). We replicated
the method on Qwen3.5-0.8B and tested it against ground truth we own: 64 loop neurons
per layer in L15/L19, verified by an exact strength-6 weight edit that lifted closed-
thinking from 11% to 83%.

Decision. Attribution search's top-10 overlapped 2/10 (L15) and 0/10 (L19) with the
known set; a finite-difference re-check gave 1/10 and 0/10. We abandon identification by
attribution for the loop behavior and use the known neurons.

Rationale. The loop is an emergent long-generation behavior: first-order next-token
sensitivity does not expose its circuit. This matches neuron-lens finding 4 (lens and
steering agree because both average over contexts). Attributing an intervention to a
single neuron is therefore not the same as attributing a long-horizon behavior.

Consequence. The identification half of the experiment moves to a later phase (a
behavior with a shorter causal horizon, e.g. language register drift, where the
first-order effect is expected to be informative). The glial experiment proceeds with
verified neurons, which is the stronger control anyway.
