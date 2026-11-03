"""Behavioural comparison of the candidate levers (3 problems, 8192-token cap).

Three arms on the same paired problems:
  base      alpha = 0                                  -> the loop
  ours      our VERIFIED static edit as the lever      -> known to close 15/18
            (D = -6 * (W[:,j].d_hat) * d_hat over our diagnosed 64+64 neurons)
  ckpt      the authors' checkpoint delta as the lever  -> D = W_edited - W_base
            over the 62+63 columns that differ

alpha = 1 in each case means "apply the full fix". The point of the experiment is not
the base arm (we know it loops) but whether a lever reproduces the fix's behaviour,
because the glial's learned quantity will be alpha(t) on top of one of these levers.

Cost: base arm runs to the cap on every problem (~7 min each on a T4), the fix arms
should stop early. Budget ~35 min.
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
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "3"))
MAX_NEW = int(os.environ.get("MAX_NEW", "8192"))
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[arms {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
edited = AutoModelForCausalLM.from_pretrained(EDITED_ID, dtype=torch.float16).cuda().eval()
pairs = json.load(open(os.path.join(ROOT, "s1_pairs.json")))["pairs"]

# ---------- lever A: ours (verified static edit) ----------
ours_spec = {}
for L in (15, 19):
    idx = np.load(os.path.join(ROOT, f"neurons_L{L}.npy"))
    d = np.load(os.path.join(ROOT, f"loopdir_L{L}.npy"))
    c = np.load(os.path.join(ROOT, f"proj_L{L}.npy"))
    d = d / (np.linalg.norm(d) + 1e-9)
    D = (-6.0 * c)[None, :] * d[:, None]        # [1024, n]  == W[:,j] -= 6 c_j d_hat
    ours_spec[L] = ([int(i) for i in idx], D)
    log(f"ours L{L}: {len(idx)} neurons, |D| mean {np.linalg.norm(D, axis=0).mean():.4f}")

# ---------- lever B: the authors' checkpoint delta ----------
ckpt_spec = {}
for L in (15, 19):
    idx, D, norms = extract_edit(model.model.layers[L].mlp.down_proj,
                                  edited.model.layers[L].mlp.down_proj, k=64)
    ckpt_spec[L] = ([int(i) for i in idx], D)
    log(f"ckpt L{L}: {len(idx)} columns differ (|delta|>0.05)")

switch = {"a": 0.0}

def make_levers(which):
    """Fresh lever pair for one arm (the caller removes the previous pair first)."""
    spec = ours_spec if which == "ours" else ckpt_spec
    return [DeltaLever(model.model.layers[L].mlp.down_proj, spec[L][0], spec[L][1],
                       lambda st=switch: st["a"]) for L in (15, 19)]


levers = make_levers("ours")


def prompt_ids(problem):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tok(txt, return_tensors="pt").input_ids.cuda()


def gen(problem, max_new=MAX_NEW):
    enc = prompt_ids(problem)
    with torch.no_grad():
        out = model.generate(enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                           tok.eos_token_id],
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.shape[1]:], skip_special_tokens=True)


def run_arm(label, which, alpha):
    global levers
    for lv in levers:
        lv.remove()
    levers = make_levers(which)
    switch["a"] = alpha
    closed, chars = 0, []
    for p in pairs[:N_PROBLEMS]:
        text = gen(p["problem"])
        ok = "</think>" in text
        closed += int(ok)
        chars.append(len(text))
        log(f"  {label} id={p['id']} closed={ok} chars={len(text)}")
    return dict(lever=which, alpha=alpha, closed=closed, n=N_PROBLEMS,
                closed_rate=closed / N_PROBLEMS, mean_chars=float(np.mean(chars)))


# ---------- corrected logit equivalence: ckpt lever at alpha=1 vs the edited checkpoint ----------
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
    log(f"A. id={p['id']}: max|delta logits| = {d:.5f} argmax agree {agree}")
log(f"A. LOGIT EQUIVALENCE (alpha=1): max|delta| = {maxdiff:.5f} argmax agree {agree}")

arms = {}
arms["base"] = run_arm("base", "ours", 0.0)
arms["ours_alpha1"] = run_arm("ours_a1", "ours", 1.0)
arms["ckpt_alpha1"] = run_arm("ckpt_a1", "ckpt", 1.0)
log("SUMMARY", json.dumps(arms))
json.dump(dict(status="complete", logit_max_abs_diff=maxdiff, argmax_agree=agree, arms=arms,
               reference=dict(base_rate=0.11, static_rate=0.83, n_full=18)),
          open(os.path.join(OUT, "lever_arms.json"), "w"), indent=1)
