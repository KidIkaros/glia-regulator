# Reproducing this work

Two parts: the **weight-space reproduction** (the published checkpoint pair) and the
**glial regulator** (this repository's research code). Both were produced on Kaggle
T4s; the commands below are the ones that actually ran.

## 0. Prerequisites

```bash
# Kaggle CLI 2.2.5 with a working login
kaggle auth login                 # OAuth (browser); the access token expires ~6h
python3 scripts/kaggle_refresh.py # after ~6h: refresh + persist for the CLI
                                  # (the CLI's own auth path does not self-refresh)

# Hugging Face (for the edited checkpoint)
hf auth login
```

## 1. Unit gates (no GPU, seconds) — run these before every kernel launch

```bash
python3 tests/test_glial_gain.py       # gains preserve dtype; only target neurons change
python3 tests/test_payload_layout.py    # generated payload resolves at the paths the driver uses
```

The payload test is a *proved* regression test: revert the `ROOT` lookup in
`src/stageA.py` and it fails; restore it and it passes. Two kernel launches died from
exactly that bug before the test existed.

## 2. Diagnostics (≈5 min GPU)

```bash
python3 build_diag.py
kaggle kernels push -p kernel_diag
kaggle kernels logs mauricew/glia-diag-v1 -f
```

Answers: how much does the glial's read signal vary across documents (Q1/Q1b), and
what authority does a bounded per-neuron gain have (Q2).

## 3. Diff census (≈2 min GPU)

```bash
python3 build_census.py
kaggle kernels push -p kernel_census
```

Lists every tensor that differs between the base model and the authors' edited
checkpoint (answer: exactly 2 tensors, `layers.15/19.mlp.down_proj.weight`).

## 4. Lever arms (≈25 min GPU)

```bash
python3 build_census.py        # currently builds kernel_arms
kaggle kernels push -p kernel_arms
kaggle kernels logs mauricew/glia-lever-arms -f
```

Runs the corrected logit-equivalence check (α = 1 vs the edited checkpoint) and a
behavioural comparison of the base model against two candidate levers.

## 5. Stage A training (the main experiment, ≈45 min GPU)

```bash
python3 build_stageA.py
kaggle kernels push -p kernel_stageA
kaggle kernels logs mauricew/glia-stagea-v1 -f
```

The kernel writes `glial_state.pt`, `eval_results.jsonl`, `selectivity.json`,
`knockout.json` and `meta_stageA.json` to `/kaggle/working` (download with
`kaggle kernels output`).

**Pre-registered success band:** closed-thinking rate ≥ 70% **and** knockout closed
rate ≤ 25% (knockout = the same prompts with the regulator switched off; it must bring
the loop back, otherwise the regulator is a static edit in disguise).

**Guards built into the kernel, all of which earned their keep:**
- *saturation guard* — if the learned gain pins at its bound, skip the evaluation
  (it turned a would-be 4-hour degenerate generation run into a 2-minute exit);
- *session deadline* — the driver stops and writes partial results before the quota
  guard fires;
- *payload assertions* — the kernel refuses to start if a payload file is missing.

## 6. Weight-space reproduction (the ground truth this project stands on)

See the companion repository `KidIkaros/wsp-reproduction` for the full protocol:
paired presence-penalty probes, direction diagnosis, the bit-exact reconstruction of
the authors' edit, and the paired evaluation (2/18 → 15/18 closed thinking).

## Protocol notes and deviations

- **Sampling**: greedy (`do_sample=False`), 16k-token cap, `<|im_end|>` and EOS as
  stop tokens; completion is decided by EOS presence, never by length (HF batches
  uniform lengths, so per-row lengths mislead).
- **Prompt**: the same instruction the published work uses —
  `\nPlease reason step by step, and put your final answer within \boxed{}.`
- **dtype**: fp16 on T4 (the published work used bf16).
- **Closed-thinking metric**: the model's own `</think>` marker appears in the
  completion; its presence/absence is the label used everywhere in this project, both
  for the base model (16/18 looped) and after the fix (2/18 looped).
