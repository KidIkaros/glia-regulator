"""Horizon-equivalence of the runtime lever - a CPU experiment.

The question. ADR-0002's amendment says the runtime lever is logit-equivalent to the
weight edit at ONE step (max |delta logits| = 0.164) yet does not reproduce its
behaviour over a 16k-token generation. Nobody has measured HOW FAST the two separate.
That quantity decides whether the runtime-lever path is viable at all:

  * if base+lever(alpha=1) tracks the edited checkpoint's predictions for hundreds of
    tokens inside a loop, the lever is nearly horizon-equivalent and a slightly larger
    alpha may buy the fix's behaviour;
  * if it separates within tens of tokens, the runtime lever can never reproduce a
    weight-space fix over a long horizon, and the glial must act on the WEIGHTS
    (alpha(t) -> a piecewise weight edit) rather than on activations.

Why CPU is the right tool. This needs teacher-forcing, not long generations: one
rollout to get a real looping token sequence, then one cached prefill per arm to read
the model's predictions at many positions. That is a handful of forward passes - a few
minutes on CPU kernels, which do not draw GPU quota (we have 0.03h of GPU left until
the Sunday refresh).

The design is a controlled comparison on ONE shared token sequence:
  1. rollout the BASE model on a paired problem -> a real looping trajectory S;
  2. teacher-force the AUTHORS' EDITED checkpoint on S -> its predictions (the target);
  3. teacher-force base + lever(alpha) on S for each alpha -> its predictions;
  4. per checkpoint position, report KL(target || arm) and argmax agreement, with
     base alone (alpha=0) as the control.

Same tokens, different models -> the comparison is paired and within-run.

Environment: ALPHAS (default "0.5 1.0 1.5 2.0"), ROLLOUT (default 256),
CHECKPOINTS (default 8), DEADLINE_S (default 3300).
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
ROLLOUT = int(os.environ.get("ROLLOUT", "256"))
CHECKPOINTS = int(os.environ.get("CHECKPOINTS", "8"))
DEADLINE_S = int(os.environ.get("DEADLINE_S", "3300"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.5 1.0 1.5 2.0").split()]
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[cpu {round(time.time()-t0)}s]", *a, flush=True)


def deadline_guard():
    time.sleep(max(0, DEADLINE_S - (time.time() - t0)))
    log("DEADLINE reached - writing partial results and exiting")
    try:
        json.dump(dict(status="deadline", partial=True, results=globals().get("RESULTS", {})),
                  open(os.path.join(OUT, "cpu_horizon.json"), "w"), indent=1)
    except Exception:
        pass
    os._exit(0)


threading.Thread(target=deadline_guard, daemon=True).start()

torch.set_num_threads(int(os.environ.get("N_THREADS", "4")))
log("loading base")
base, tok = load_model(MODEL_ID, device="cpu")
log("loading edited checkpoint")
edited, _ = load_model(EDITED_ID, device="cpu")

pairs = load_pairs(os.path.join(ROOT, "s1_pairs.json"))
switch = {"a": 0.0}

# ---------------- lever derived in-kernel from the two checkpoints ----------------
spec, lever_diagnostics = build_lever_from_checkpoints(base, edited)
levers = [DeltaLever(base.model.layers[L].mlp.down_proj, spec[L][0], spec[L][1],
                     lambda st=switch: st["a"]) for L in (15, 19)]
log("lever installed on layers", list(spec))


def prompt_ids(problem):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tok(txt, return_tensors="pt").input_ids


# ---------------- 1. a real looping rollout from the BASE model ----------------
problem = pairs[0]
switch["a"] = 0.0
encoded = prompt_ids(tok, problem["problem"])
log(f"prompt: {encoded.shape[1]} tokens; rolling out {ROLLOUT}")
new_tokens, rollout_info = greedy_rollout(base, encoded, ROLLOUT, eos_token_ids(tok))
rate = rollout_info["n"] / max(1e-6, rollout_info["seconds"])
log(f"rollout done: {rollout_info['n']} tokens at {rate:.2f} tok/s | "
    f"still-thinking={not rollout_info['closed']}")

seq = torch.cat([encoded[0], new_tokens])        # prompt + rollout (the shared token path)
n_prompt = encoded.shape[1]
positions = [n_prompt + int(p) for p in np.linspace(1, gen.shape[1] - n_prompt,
                                                    min(CHECKPOINTS, gen.shape[1] - n_prompt),
                                                    endpoint=False)]
positions = [p for p in positions if 0 < p < gen.shape[1]]
log("checkpoint positions:", positions)


def predictions_at(model, tokens, positions, use_lever):
    """Teacher-force `tokens` once (cached) and return distributions at `positions`."""
    # tokens is a 1-D [T] tensor (the shared rollout path); the model expects
    # [batch, seq, hidden] -- a 1-D input makes the Qwen3.5 forward see a 2-D hidden
    # state and fail with "not enough values to unpack (expected 3, got 2)".
    with torch.no_grad():
        out = model(tokens.unsqueeze(0))
    logits = out.logits[0].float()              # [T, V]
    return {p: torch.softmax(logits[p - 1], -1) for p in positions}


# ---------------- 2. the target: the edited checkpoint on the SAME tokens ----------------
tgt = predictions_at(edited, seq, positions, False)
log("target distributions (edited checkpoint) computed")

# ---------------- 3. the arms ----------------
RESULTS = {}
for alpha in [0.0] + ALPHAS:
    switch["a"] = alpha
    t_arm = time.time()
    arm = predictions_at(base, seq, positions, alpha > 0)
    rows = []
    for p in positions:
        a = arm[p]
        b = tgt[p]
        kl = float((b * (b.clamp_min(1e-12).log() - a.clamp_min(1e-12).log())).sum())
        rows.append(dict(pos=int(p), offset=int(p - n_prompt),
                         kl_target_vs_arm=kl,
                         agree_argmax=bool(a.argmax() == b.argmax()),
                         arm_entropy=float(-(a * a.clamp_min(1e-12).log()).sum()),
                         target_entropy=float(-(b * b.clamp_min(1e-12).log()).sum())))
    RESULTS[str(alpha)] = dict(rows=rows,
                               mean_kl=float(np.mean([r["kl_target_vs_arm"] for r in rows])),
                               mean_agree=float(np.mean([r["agree_argmax"] for r in rows])),
                               first_pos_kl=rows[0]["kl_target_vs_arm"],
                               last_pos_kl=rows[-1]["kl_target_vs_arm"],
                               arm_seconds=time.time() - t_arm)
    log(f"alpha={alpha}: mean KL(target||arm) {RESULTS[str(alpha)]['mean_kl']:.5f} | "
        f"argmax agree {RESULTS[str(alpha)]['mean_agree']:.3f} | "
        f"first pos {RESULTS[str(alpha)]['first_pos_kl']:.5f} -> "
        f"last pos {RESULTS[str(alpha)]['last_pos_kl']:.5f} "
        f"({RESULTS[str(alpha)]['arm_seconds']:.0f}s)")

out = dict(status="complete", rollout_tokens=int(new_tokens.shape[0]),
           tokens_per_second=rate, still_thinking=not rollout_info["closed"],
           lever=lever_diagnostics,
           positions=positions, prompt_tokens=int(n_prompt), results=RESULTS,
           note="KL(target||arm) at shared positions inside the BASE model's looping "
                "trajectory; alpha=0 is the control (base model alone)")
json.dump(out, open(os.path.join(OUT, "cpu_horizon.json"), "w"), indent=1)
log("SUMMARY", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                            for k, v in RESULTS.items()}))
