"""Dose-response sweep with a capability panel (the next experiment, T9).

Question. The lever breaks the reasoning loop at alpha=1.5 - faster than the published
weight edit does - but alpha=1.5 removes 9x the loop direction instead of 6x. So the
binding constraint is no longer "does it work" but "what does it cost". This sweep maps
closed-thinking rate against capability damage, so alpha(t) can be trained on a setting
that is both effective and safe.

Arms: alpha in {1.0, 1.25, 1.5, 1.75, 2.0} x the 18 paired problems, plus a capability
panel per alpha.

Protocol (identical to the published one, and to the run that produced the headline
result): greedy, presence penalty 1.5 on EVERY arm, fp32, 8192-token cap, and the
closed-thinking label is the model's own </think> marker.

Capability panel per alpha:
  * closed_rate  - fraction of the 18 problems whose thinking closes;
  * ordinary_kl  - mean KL against the base model on the next-token distribution of 8
                    ordinary prompts (does the intervention distort normal text?);
  * ordinary_agree - fraction of those positions where the argmax is unchanged;
  * ppl_delta   - perplexity change on a fixed neutral English passage (embedded, so the
                    panel is deterministic and needs no download).

Problem set. BALANCED=1 selects half loopers and half healthy controls (problems the
base model already handles), so we see both that the intervention breaks loops and that
it does not damage ordinary work. With 8 problems the closed-rate resolution is 1/8.

Pre-registered gates (fixed before the run):
  * the best alpha must close >= 70% of the problems;
  * and at that alpha, ordinary_kl <= 0.05 and ppl_delta <= +20%;
  * if no alpha clears both, the runtime lever is not a viable regulator at this
    authority and the design pivots to the weight-space form (see ADR-0002).

Environment: ALPHAS, N_PROBLEMS, ROLLOUT, PRESENCE_PENALTY, MAX_ARMS_PER_RUN,
RESULTS_PATH. One arm per invocation keeps each cell inside the CLI's reply deadline,
and the results file makes the next invocation resume.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from lever import DeltaLever, build_lever_from_checkpoints

block_triton_kernels()

MODEL_ID = "Qwen/Qwen3.5-0.8B"
EDITED_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "1.0 1.25 1.5 1.75 2.0").split()]
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "18"))
ROLLOUT = int(os.environ.get("ROLLOUT", "8192"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "1.5"))
MAX_PROBLEMS_PER_RUN = int(os.environ.get("MAX_PROBLEMS_PER_RUN", "2"))
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/content/glia_out/sweep.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
started = time.time()

ORDINARY_PROMPTS = [
    "Write a short paragraph explaining why bread dough rises.",
    "Give three tips for someone learning to swim as an adult.",
    "Summarise what a solar eclipse is in two sentences.",
    "What is the difference between weather and climate?",
    "Suggest a simple weeknight dinner using lentils.",
    "Explain what a hash function is to a curious beginner.",
    "Describe one benefit of walking to work instead of driving.",
    "How would you teach a child to count change?",
]
NEUTRAL_PASSAGE = (
    "Bread dough rises because yeast consumes sugars in the flour and releases carbon "
    "dioxide, which is trapped by the gluten network that has been stretched and "
    "folded. The gluten gives the dough its elasticity, the gas inflates the small "
    "cells of the crumb, and the oven sets the protein structure so the loaf keeps "
    "the shape it had when it left the pan. Resting the dough before baking lets the "
    "network relax, which is why a gentler handling produces a more even crumb."
)


def as_bool(value):
    """s1_pairs.json stores base_looped as the STRING "False"/"True", which is truthy
    for both - so it must be parsed, not truth-tested."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def log(*parts):
    print(f"[sweep {round(time.time()-started)}s]", *parts, flush=True)


def find_pairs():
    for candidate in (os.path.join(HERE, "s1_pairs.json"),
                      os.path.join(os.path.dirname(HERE), "s1_pairs.json")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("s1_pairs.json not found next to the driver or one level up")


def read_rows():
    """One row per (alpha, problem) plus one panel row per alpha."""
    rows = {}
    if not os.path.exists(RESULTS_PATH):
        return rows
    for line in open(RESULTS_PATH):
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
                rows[row["key"]] = row
            except json.JSONDecodeError:
                continue
    return rows


def record(key, payload):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as handle:
        handle.write(json.dumps(dict(payload, key=key)) + "\n")
    log(f"recorded {key}")


log(f"device={DEVICE} dtype={DTYPE} rollout={ROLLOUT} penalty={PRESENCE_PENALTY} "
    f"alphas={ALPHAS} -> {RESULTS_PATH}")
rows = read_rows()
if rows:
    log(f"resuming: {len(rows)} rows already recorded")

base, tok = load_model(MODEL_ID, device=DEVICE, dtype=DTYPE)
edited, _ = load_model(EDITED_ID, device=DEVICE, dtype=DTYPE)
spec, diagnostics = build_lever_from_checkpoints(base, edited)
switch = {"a": 0.0}
levers = [DeltaLever(base.model.layers[layer].mlp.down_proj, spec[layer][0],
                     spec[layer][1], lambda: switch["a"]) for layer in spec]
stop_ids = eos_token_ids(tok)
all_pairs = load_pairs(find_pairs())
if os.environ.get("BALANCED", "1") == "1":
    # a dose-response needs healthy CONTROLS: part of the set must be problems the base
    # model already handles, so we can see whether the intervention damages them
    loopers = [q for q in all_pairs if as_bool(q.get("base_looped", True))]
    healthy = [q for q in all_pairs if not as_bool(q.get("base_looped", True))]
    half = max(1, N_PROBLEMS // 2)
    pairs = loopers[:half] + healthy[:N_PROBLEMS - half]
    if len(pairs) < N_PROBLEMS:
        chosen = set(id(q) for q in pairs)
        pairs += [q for q in all_pairs if id(q) not in chosen][:N_PROBLEMS - len(pairs)]
else:
    pairs = all_pairs[:N_PROBLEMS]
log(f"problem set: {len(pairs)} = {sum(q.get('base_looped', True) for q in pairs)} loopers + "
    f"{sum(not q.get('base_looped', True) for q in pairs)} healthy controls")

# ---- base references for the capability panel (alpha = 0) ----
ordinary_ids = [prompt_ids(tok, p).to(DEVICE) for p in ORDINARY_PROMPTS]
with torch.no_grad():
    ordinary_base = [torch.softmax(base(ids).logits[0, -1].float(), -1) for ids in ordinary_ids]
with torch.no_grad():
    neutral_ids = tok(NEUTRAL_PASSAGE, return_tensors="pt").input_ids.to(DEVICE)
    neutral_base_nll = float(base(neutral_ids, labels=neutral_ids).loss)

def capability_panel():
    with torch.no_grad():
        kls, agrees = [], []
        for ids, reference in zip(ordinary_ids, ordinary_base):
            probs = torch.softmax(base(ids).logits[0, -1].float(), -1)
            kls.append(float((reference * (reference.clamp_min(1e-12).log()
                                            - probs.clamp_min(1e-12).log())).sum()))
            agrees.append(float(probs.argmax() == reference.argmax()))
        neutral_nll = float(base(neutral_ids, labels=neutral_ids).loss)
    return dict(ordinary_kl=float(np.mean(kls)), ordinary_agree=float(np.mean(agrees)),
                ppl_base=float(np.exp(neutral_base_nll)),
                ppl_at_alpha=float(np.exp(neutral_nll)),
                ppl_delta_pct=float((np.exp(neutral_nll) / np.exp(neutral_base_nll) - 1) * 100))


done = 0
for alpha in ALPHAS:
    prefix = f"alpha_{alpha:g}".replace(".", "_")
    switch["a"] = alpha
    panel_key = prefix + "_panel"
    if panel_key not in rows:
        panel = capability_panel()
        rows[panel_key] = dict(key=panel_key, alpha=alpha, kind="panel", **panel)
        record(panel_key, rows[panel_key])
        log(f"panel alpha={alpha}: ordinary_kl {panel['ordinary_kl']:.4f} | "
            f"agree {panel['ordinary_agree']:.3f} | ppl {panel['ppl_base']:.1f} -> "
            f"{panel['ppl_at_alpha']:.1f} ({panel['ppl_delta_pct']:+.1f}%)")
    for problem in pairs:
        key = prefix + "_" + str(problem["id"])
        if key in rows:
            continue
        if done >= MAX_PROBLEMS_PER_RUN:
            log("chunk budget reached - run the script again to continue")
            break
        encoded = prompt_ids(tok, problem["problem"]).to(DEVICE)
        tokens, info = greedy_rollout(base, encoded, ROLLOUT, stop_ids, tokenizer=tok,
                                       presence_penalty=PRESENCE_PENALTY)
        row = dict(key=key, alpha=alpha, kind="problem", id=problem["id"],
                   base_looped=as_bool(problem.get("base_looped", False)),
                   closed=bool(info["closed"]), chars=info["chars"], n=info["n"],
                   seconds=info["seconds"])
        record(key, row)
        done += 1

switch["a"] = 0.0
summary = {}
for alpha in ALPHAS:
    prefix = f"alpha_{alpha:g}".replace(".", "_")
    problem_rows = [r for k, r in rows.items()
                    if k.startswith(prefix + "_") and r.get("kind") == "problem"]
    panel = rows.get(prefix + "_panel", {})
    if problem_rows:
        summary[str(alpha)] = dict(
            closed=sum(r["closed"] for r in problem_rows), n=len(problem_rows),
            closed_rate=sum(r["closed"] for r in problem_rows) / len(problem_rows),
            mean_chars=float(np.mean([r["chars"] for r in problem_rows])),
            ordinary_kl=panel.get("ordinary_kl"), ordinary_agree=panel.get("ordinary_agree"),
            ppl_base=panel.get("ppl_base"), ppl_at_alpha=panel.get("ppl_at_alpha"),
            ppl_delta_pct=panel.get("ppl_delta_pct"))
print("\n=== SWEEP SUMMARY ===")
print(json.dumps(summary, indent=1))
print("results file:", RESULTS_PATH)
