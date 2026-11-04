"""Verification of the CORRECTED "ours" lever, plus the checkpoint lever's equivalence.

Two bugs were found and fixed before this run:
  1. SIGN: `proj_L*.npy` is stored NEGATED (verified locally: corr(live W.d_hat, proj) =
     -1.000000 exactly), so the edit's per-neuron alpha is +6*proj_j, and the delta must
     be D[:,j] = +6*proj_j*d_hat. Building it as -6*proj*d_hat applied the ANTI-fix -
     which is visible in the behavior (the "ours" arm looped LONGER than base).
  2. The earlier logit-equivalence check compared the OURS lever against the AUTHORS'
     edited checkpoint. Those are two different edits (different directions), so the gap
     measured the distance between edits, not a bug. Equivalence must use the
     CHECKPOINT delta.

This kernel:
  A. verifies the corrected delta against the LIVE weights in-kernel
     (D must equal -6*(W[:,j].d_hat)*d_hat for every selected neuron);
  B. verifies that the checkpoint lever at alpha=1 reproduces the authors' edited
     checkpoint's logits (max |delta| must be ~0, argmax must agree);
  C. runs the corrected OURS lever on the paired problems and reports closed-thinking.

The base arm is not re-run: it was measured at 0/2 closed in the previous session
(both problems looped to the 8192-token cap).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lever import DeltaLever, extract_edit

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
EDITED_ID = os.environ.get("EDITED_ID", "RikiyaT/Qwen3.5-0.8B-weight-space-programming")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "2"))
MAX_NEW = int(os.environ.get("MAX_NEW", "8192"))
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[v2 {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
edited = AutoModelForCausalLM.from_pretrained(EDITED_ID, dtype=torch.float16).cuda().eval()
pairs = json.load(open(os.path.join(ROOT, "s1_pairs.json")))["pairs"]
switch = {"a": 0.0}


def prompt_ids(problem):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tok(txt, return_tensors="pt").input_ids.cuda()


# ---------- A. corrected ours-delta, verified against the live weights ----------
ours_spec = {}
for L in (15, 19):
    idx = np.load(os.path.join(ROOT, f"neurons_L{L}.npy"))
    d = np.load(os.path.join(ROOT, f"loopdir_L{L}.npy")).astype(np.float32)
    d = d / (np.linalg.norm(d) + 1e-9)
    proj = np.load(os.path.join(ROOT, f"proj_L{L}.npy")).astype(np.float32)
    W = model.model.layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()
    c_live = W[:, [int(i) for i in idx]].T @ d          # live W.d_hat
    D_stored = (6.0 * proj)[None, :] * d[:, None]       # corrected: +6*proj*d_hat
    D_live = (-6.0 * c_live)[None, :] * d[:, None]      # the edit the paper/our stage-2 used
    err = float(np.abs(D_stored - D_live).max())
    corr = float(np.corrcoef(c_live.ravel(), proj.ravel())[0, 1])
    log(f"A. L{L}: max|D_stored - D_live| = {err:.6f} | corr(c_live, proj) = {corr:+.6f} "
        f"| |D| mean {np.abs(D_stored).mean():.4f}")
    ours_spec[L] = ([int(i) for i in idx], D_stored)

# ---------- B. checkpoint lever at alpha=1 == the authors' edited checkpoint ----------
ckpt_spec = {}
for L in (15, 19):
    idx, D, _ = extract_edit(model.model.layers[L].mlp.down_proj,
                             edited.model.layers[L].mlp.down_proj, k=64)
    ckpt_spec[L] = ([int(i) for i in idx], D)
    log(f"B. L{L}: {len(idx)} changed columns, |delta| mean {np.linalg.norm(D, axis=0).mean():.4f}")

ckpt_levers = [DeltaLever(model.model.layers[L].mlp.down_proj, ckpt_spec[L][0],
                          ckpt_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]
maxdiff, agree = 0.0, True
for p in pairs[:2]:
    enc = prompt_ids(p["problem"])
    switch["a"] = 1.0
    with torch.no_grad():
        lg_a = model(enc).logits[:, -32:].float()
    switch["a"] = 0.0
    with torch.no_grad():
        lg_b = edited(enc).logits[:, -32:].float()
    d = float((lg_a - lg_b).abs().max())
    maxdiff = max(maxdiff, d)
    agree = agree and bool((lg_a.argmax(-1) == lg_b.argmax(-1)).all())
    log(f"B. id={p['id']}: max|delta logits| = {d:.5f} argmax agree {agree}")
for lv in ckpt_levers:
    lv.remove()
log(f"B. CHECKPOINT LEVER EQUIVALENCE: max|delta| = {maxdiff:.5f} argmax agree {agree}")

# ---------- C. corrected ours lever, behavioural ----------
ours_levers = [DeltaLever(model.model.layers[L].mlp.down_proj, ours_spec[L][0],
                          ours_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]
switch["a"] = 1.0
closed, chars = 0, []
for p in pairs[:N_PROBLEMS]:
    enc = prompt_ids(p["problem"])
    with torch.no_grad():
        out = model.generate(enc, max_new_tokens=MAX_NEW, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                           tok.eos_token_id],
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, enc.shape[1]:], skip_special_tokens=True)
    ok = "</think>" in text
    closed += int(ok)
    chars.append(len(text))
    log(f"C. ours(corrected) id={p['id']} closed={ok} chars={len(text)}")
log(f"C. OURS LEVER (corrected): closed {closed}/{N_PROBLEMS}")

out = dict(status="complete",
           checkpoint_lever_equivalence=dict(max_abs_logit_diff=maxdiff, argmax_agree=agree),
           ours_lever_behaviour=dict(closed=closed, n=N_PROBLEMS,
                                     mean_chars=float(np.mean(chars))),
           base_reference=dict(closed=0, n=2, note="previous session, same 8192 cap"))
json.dump(out, open(os.path.join(OUT, "lever_v2.json"), "w"), indent=1)
log("SUMMARY", json.dumps(out))
