"""Lever equivalence check: does alpha=1 of the DeltaLever reproduce the authors'
edited checkpoint, and does it reproduce its behaviour?

A. LOGIT EQUIVALENCE: max |delta logits| between (base + lever(alpha=1)) and the
   authors' edited checkpoint, on 2 prompts. Expect ~0 (bf16 rounding only).
B. AUTHORITY CURVE: KL(base||lever(alpha)) for alpha in {0, 0.25, 0.5, 0.75, 1}.
   This is the amount of authority the glial has available at each strength.
C. BEHAVIOUR: closed-thinking rate over N problems at alpha=0 and alpha=1.
   Expect base ~0-1 closed of 6, edited-equivalent ~5-6 of 6.

Cost: ~20 min on one T4 (2 model loads + 12 generations at up to 16384 tokens).
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
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "6"))
MAX_NEW = int(os.environ.get("MAX_NEW", "16384"))
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[lever {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
edited = AutoModelForCausalLM.from_pretrained(EDITED_ID, dtype=torch.float16).cuda().eval()
log("both checkpoints loaded")

pairs = json.load(open(os.path.join(ROOT, "s1_pairs.json")))["pairs"]

# ---- extract the authors' edit (columns with a non-negligible delta) ----
switch = {"a": 0.0}
levers = []
edit_info = {}
for L in (15, 19):
    idx, delta, norms = extract_edit(model.model.layers[L].mlp.down_proj,
                                     edited.model.layers[L].mlp.down_proj, k=64)
    changed = int((norms > 0.05).sum())
    edit_info[L] = dict(n_changed_columns=changed, delta_norm_mean=float(np.linalg.norm(delta, axis=0).mean()),
                        idx_first10=[int(x) for x in idx[:10]])
    log(f"L{L}: columns with |delta|>0.05: {changed}/3584 (selected {len(idx)}); |delta| mean "
        f"{np.linalg.norm(delta, axis=0).mean():.4f}")
    ours = np.load(os.path.join(ROOT, f"neurons_L{L}.npy"))
    ov = len(set(idx.tolist()) & set(int(x) for x in ours))
    log(f"L{L}: overlap with OUR diagnosed 64 neurons: {ov}/64")
    edit_info[L]["overlap_with_our_diagnosis"] = ov
    lv = DeltaLever(model.model.layers[L].mlp.down_proj, [int(i) for i in idx], delta,
                    lambda st=switch: st["a"])
    levers.append(lv)


def prompt_ids(problem):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tok(txt, return_tensors="pt").input_ids.cuda()


# ---------- A. logit equivalence ----------
maxdiff, agree_all = 0.0, True
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
    same = bool((lg_a.argmax(-1) == lg_b.argmax(-1)).all())
    agree_all = agree_all and same
    log(f"A. id={p['id']}: max|delta logits| = {d:.5f} | argmax agree: {same}")
log(f"A. LOGIT EQUIVALENCE: max|delta| = {maxdiff:.5f} over 2 prompts, argmax agree {agree_all}")

# ---------- B. authority curve ----------
base_p = None
curve = []
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    switch["a"] = alpha
    enc = prompt_ids(pairs[0]["problem"])
    with torch.no_grad():
        p = torch.softmax(model(enc).logits[0, -1].float(), -1)
    if base_p is None:
        base_p = p
    kl = float((base_p * (base_p.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())
    curve.append(dict(alpha=alpha, kl_from_base=kl))
    log(f"B. alpha={alpha:.2f}: KL(base||lever) = {kl:.5f}")

# ---------- C. behaviour ----------
def gen(problem, max_new=MAX_NEW):
    enc = prompt_ids(problem)
    with torch.no_grad():
        out = model.generate(enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                           tok.eos_token_id],
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.shape[1]:], skip_special_tokens=True)


behaviour = {}
for label, alpha in (("base_alpha0", 0.0), ("edited_alpha1", 1.0)):
    switch["a"] = alpha
    closed, chars = 0, []
    for p in pairs[:N_PROBLEMS]:
        text = gen(p["problem"])
        ok = "</think>" in text
        closed += int(ok)
        chars.append(len(text))
        log(f"C. {label} id={p['id']} closed={ok} chars={len(text)}")
    behaviour[label] = dict(alpha=alpha, closed=closed, n=N_PROBLEMS,
                            closed_rate=closed / N_PROBLEMS,
                            mean_chars=float(np.mean(chars)))
    log(f"C. {label}: closed {closed}/{N_PROBLEMS}")

out = dict(status="complete", logit_max_abs_diff=maxdiff, argmax_agree=agree_all,
           edit_info=edit_info, authority_curve=curve, behaviour=behaviour,
           static_edit_reference=dict(base_closed=2, static_closed=15, n=18,
                                      base_rate=0.11, static_rate=0.83))
json.dump(out, open(os.path.join(OUT, "lever_equivalence.json"), "w"), indent=1)
log("SUMMARY", json.dumps(out))
