"""Trajectory-equivalence horizon of the runtime lever (CPU, no GPU quota).

What we know. Teacher-forced on a shared token path, the runtime lever at alpha=1
reproduces the authors' edited checkpoint's predictions to KL 2.3e-5 (620x closer than
the base model), with argmax agreement 1.000 over 256 tokens. And yet at the full 16k
cap the runtime lever did NOT close a loop that the weight edit closes. So local
equivalence is not the missing piece; ACCUMULATION is.

This measures the accumulation directly: run FREE-RUNNING GREEDY rollouts and find the
first token where they differ.

  * reference: the edited checkpoint's own greedy rollout (the fix's behaviour);
  * arms: base + lever(alpha) greedy rollouts from the same prompt.

  first_divergence_index(arm) = the number of tokens the arm and the fix agree on
  before producing different tokens. A large number means the lever is nearly
  trajectory-equivalent and the 16k behavioural difference is a knife-edge effect of a
  chaotic system; a small number means the runtime lever can never reproduce a
  weight-space fix over a long horizon, and the experiment must pivot to the
  weight-space form (patching down_proj.weight in place), which forecloses a
  per-token alpha(t).

This is the cheap pre-screen for the Sunday GPU dose-response: it can fire the kill
rule without spending GPU quota.

Environment: ALPHAS (default "1.0 1.5 2.0"), ROLLOUT (default 1024), DEADLINE_S (5400).
"""
import json
import os
import sys
import threading
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from lever import DeltaLever, build_lever_from_checkpoints

block_triton_kernels()

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
EDITED_ID = os.environ.get("EDITED_ID", "RikiyaT/Qwen3.5-0.8B-weight-space-programming")
ROLLOUT = int(os.environ.get("ROLLOUT", "1024"))
DEADLINE_S = int(os.environ.get("DEADLINE_S", "5400"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "1.0 1.5 2.0").split()]
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[traj {round(time.time()-t0)}s]", *a, flush=True)


def deadline_guard():
    time.sleep(max(0, DEADLINE_S - (time.time() - t0)))
    log("DEADLINE reached - writing partial results and exiting")
    try:
        json.dump(dict(status="deadline", partial=True, results=globals().get("RESULTS", {})),
                  open(os.path.join(OUT, "cpu_trajectory.json"), "w"), indent=1)
    except Exception:
        pass
    os._exit(0)


threading.Thread(target=deadline_guard, daemon=True).start()

torch.set_num_threads(int(os.environ.get("N_THREADS", "4")))
log("loading base")
base, tok = load_model(MODEL_ID, device="cpu")
log("loading edited")
edited, _ = load_model(EDITED_ID, device="cpu")
pairs = load_pairs(os.path.join(ROOT, "s1_pairs.json"))
switch = {"a": 0.0}
spec, lever_diagnostics = build_lever_from_checkpoints(base, edited)
levers = [DeltaLever(base.model.layers[L].mlp.down_proj, spec[L][0], spec[L][1],
                     lambda st=switch: st["a"]) for L in (15, 19)]

problem = pairs[0]
encoded = prompt_ids(tok, problem["problem"])
stop_ids = eos_token_ids(tok)
log(f"prompt {encoded.shape[1]} tokens; rollout {ROLLOUT}")


def rollout(model):
    return greedy_rollout(model, encoded, ROLLOUT, stop_ids, tokenizer=tok)


# reference: the fix's own behaviour
switch["a"] = 0.0
ref_tokens, ref_info = rollout(edited)
log(f"reference (edited checkpoint): {ref_info}")

RESULTS = {}


def compare(name, toks, info):
    n = min(len(toks), len(ref_tokens))
    diff = None
    for i in range(n):
        if int(toks[i]) != int(ref_tokens[i]):
            diff = i
            break
    row = dict(info=info, matched_prefix=None if diff is None else int(diff),
               identical_to_fix=bool(diff is None and len(toks) == len(ref_tokens)))
    RESULTS[name] = row
    log(f"{name}: matched {row['matched_prefix']} tokens before differing "
        f"(ref len {len(ref_tokens)}, arm len {len(toks)})")
    return row


compare("base_vs_fix", base_tokens, base_info)


for alpha in ALPHAS:
    switch["a"] = alpha
    toks, info = rollout(base)
    compare(f"lever_alpha{alpha:g}", toks, info)

# the base control runs LAST: it is the least informative arm, and on a 16k-token
# rollout it is the one that should be sacrificed if the deadline arrives first
if os.environ.get("BASE_CONTROL", "1") == "1":
    base_tokens, base_info = rollout(base)
    log(f"control (base model): {base_info}")
    compare("base_vs_fix", base_tokens, base_info)

switch["a"] = 0.0
json.dump(dict(status="complete", ref=ref_info, lever=lever_diagnostics, results=RESULTS,
               rollout_requested=ROLLOUT, prompt_tokens=int(encoded.shape[1])),
          open(os.path.join(OUT, "cpu_trajectory.json"), "w"), indent=1)
log("SUMMARY", json.dumps({k: v.get("matched_prefix") for k, v in RESULTS.items()}))
