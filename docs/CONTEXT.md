# CONTEXT.md — the program glossary

Created 2026-11-04 during a grilling session, after the program's goal was stated
plainly for the first time. Every future session reads this BEFORE planning anything.
Deviations from this vocabulary are drift, and drift is the thing the user already
caught once.

## The mission

**Build the best lifelong AI companion for the user** — an AI that serves the user's
own purposes for the rest of their life. Not for resale. Not primarily for publication.
Public artifacts (GitHub/HF repos) exist because the work is real, not because papers
are the goal.

The three experiments are three organs of one body:

| experiment | organ | the question it answers |
|---|---|---|
| H-Net work (`hnet-kaggle`) | **the Substrate** | can the companion run on an efficient, owned architecture — cheaply, for decades? |
| neuron identity (Stage A2) | **the Identity** | can we locate, in weights, the units that encode a behaviour/traits — so a personality can be found, chosen, and kept? |
| glial regulator | **the Regulator** | given a chosen identity, can it be actively regulated and held stable over time? |

qwentensor (related work): evidence that arbitrary computation can live inside the
Substrate - relevant to efficiency and to the weight-side fallback.

## Vocabulary (use these terms exactly)

* **Companion** — the lifelong AI being built. The end product of the program.
* **Substrate** — the model that hosts the Companion. Today: Qwen3.5-0.8B (borrowed,
  practical). The intended endgame substrate is the user's own efficient architecture
  (the byte H-Net line), so the Companion's home is owned, not rented.
* **Identity** — the Companion's personality: traits, tone, values, continuity. NOT a
  paper concept; the thing the user will live with.
* **the neurons** — the weight loci that encode a target behaviour or trait. The
  128 loop neurons are the TESTBED for the find-and-regulate method: a behaviour with a
  causally verified edit, chosen because the method can be validated cleanly on it.
  Personality traits are the real target, and they will be messier: distributed,
  causal identification harder. That gap is known and accepted.
* **the Regulator (glial)** — the mechanism that reads context and applies a calibrated
  correction to hold a chosen behaviour/identity stable. Validated on loop suppression
  (alpha=1.25); its real job is Identity maintenance over a lifetime: drift is the
  enemy of a companion.
* **Efficiency** — the constraint that makes "for the rest of my life" literal: the
  Companion must run on hardware the user owns (consumer GPU), indefinitely. This is
  why the H-Net line exists at all.

## How the method is meant to chain (the honest causal order)

1. **Install**: a personality enters the weights through tuning/training - it is not
   usually sitting in a base model waiting to be found. (Open question for the user:
   see frontier Q2.)
2. **Locate**: the identity toolkit (lenses, drivers, activation profiling - Stage A2's
   method, validated on the loop testbed) finds WHERE the installed traits live.
3. **Regulate**: the Regulator holds the located identity stable against drift -
   context pressure, model updates, scaling, time.

Stage A2 (the 128 loop neurons) is step 2 *validated on the clean testbed*. It is not
the personality work itself; it is the proof that the instruments work.

## Hardware posture (ADR to be written)

Two tiers, declared deliberately: **T4-class for volume** (Kaggle 30 h/week, Colab
free tier) and **sm_80+ for the jobs nothing else can run** (the Modal H200 allowance).
The sm_75 wall has now been hit independently by both active threads (neuron-lens
Triton kernels; goombalab/hnet per the user's own reading index). Staying T4-scoped
forever is a choice the program has NOT made, and should not make by default.

## Standing rules (unchanged)

* GPU-quota-scale runs need explicit approval; Modal needs a vetted script and per-run
  approval (two-approval protocol: smoke, then full).
* Verified facts before plans that spend. Written plans; deviations recorded with
  reasons. Written plan files are the reference.
* /clean-code before any push.
