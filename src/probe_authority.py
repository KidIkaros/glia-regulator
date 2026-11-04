"""Where does the lever have authority? Probe it INSIDE a looping generation.

Earlier measurements of the lever's authority were taken at the prompt's last position,
where the model has not started looping yet: KL was ~1e-4 for the checkpoint lever and
~3e-3 for an activation gate. That position cannot show a fix that only matters once
the model is looping.

This probe generates a real looping continuation with the BASE model, then measures the
next-token distribution divergence at the end of that continuation between:
  base (alpha=0)  vs  checkpoint lever (alpha=1)
and, as a bonus, whether our OWN 64+64 neuron set (62/64 and 63/64 of the authors'
edited columns) is sufficient on its own -- the lever rebuilt in-kernel from the
checkpoint's direction over our neuron set.

Cheap: one 512-token generation plus a few forward passes.
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
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "512"))
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[probe {round(time.time()-t0)}s]", *a, flush=True)


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


# --- levers ---
ckpt_spec, our_spec = {}, {}
for L in (15, 19):
    idx, D, _ = extract_edit(model.model.layers[L].mlp.down_proj,
                             edited.model.layers[L].mlp.down_proj, k=64)
    ckpt_spec[L] = ([int(i) for i in idx], D)
    # rebuild the same fix over OUR neuron set: direction from the checkpoint delta,
    # magnitude from the LIVE weights (the saved npys describe an older revision)
    d = D[:, 0] / (np.linalg.norm(D[:, 0]) + 1e-9)
    neurons = [int(x) for x in np.load(os.path.join(ROOT, f"neurons_L{L}.npy"))]
    W = model.model.layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()
    c = W[:, neurons].T @ d
    our_spec[L] = (neurons, (-6.0 * c)[None, :] * d[:, None])
    log(f"L{L}: ckpt columns {len(idx)} | our neurons {len(neurons)} | "
        f"overlap {len(set(idx) & set(neurons))} | |d| {np.linalg.norm(d):.4f}")

ckpt_levers = [DeltaLever(model.model.layers[L].mlp.down_proj, ckpt_spec[L][0],
                          ckpt_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]
our_levers = [DeltaLever(model.model.layers[L].mlp.down_proj, our_spec[L][0],
                         our_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]


def dist(enc):
    with torch.no_grad():
        lg = model(enc).logits[0, -1].float()
    return torch.softmax(lg, -1)


results = {}
for pi, p in enumerate(pairs[:2]):
    enc = prompt_ids(p["problem"])
    # 1. a real looping continuation from the BASE model
    switch["a"] = 0.0
    for lv in ckpt_levers + our_levers:
        lv.remove()
    ckpt_levers[:] = [DeltaLever(model.model.layers[L].mlp.down_proj, ckpt_spec[L][0],
                                 ckpt_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]
    our_levers[:] = [DeltaLever(model.model.layers[L].mlp.down_proj, our_spec[L][0],
                                our_spec[L][1], lambda st=switch: st["a"]) for L in (15, 19)]
    with torch.no_grad():
        gen = model.generate(enc, max_new_tokens=GEN_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                           tok.eos_token_id],
                             pad_token_id=tok.eos_token_id)
    cont = gen[0, enc.shape[1]:]
    text = tok.decode(cont, skip_special_tokens=True)
    looping = "</think>" not in text
    log(f"problem {p['id']}: generated {len(cont)} tokens, still-thinking={looping}")

    # 2. KL at the prompt position vs at the end of the looping continuation
    p_prompt_base = dist(enc)
    p_loop_base = dist(gen)
    out = {"tokens": int(len(cont)), "still_thinking": bool(looping)}

    for label, levers in (("ckpt", ckpt_levers), ("ours", our_levers)):
        for lv in levers:
            lv.remove()
        switch["a"] = 1.0
        if label == "ckpt":
            ckpt_levers[:] = [DeltaLever(model.model.layers[L].mlp.down_proj,
                                         ckpt_spec[L][0], ckpt_spec[L][1],
                                         lambda st=switch: st["a"]) for L in (15, 19)]
        else:
            our_levers[:] = [DeltaLever(model.model.layers[L].mlp.down_proj,
                                        our_spec[L][0], our_spec[L][1],
                                        lambda st=switch: st["a"]) for L in (15, 19)]
        p_prompt_lev = dist(enc)
        p_loop_lev = dist(gen)
        switch["a"] = 0.0

        def kl(a, b):
            return float((a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum())

        out[label] = dict(kl_at_prompt=kl(p_prompt_base, p_prompt_lev),
                          kl_at_loop_position=kl(p_loop_base, p_loop_lev),
                          top1_changed_prompt=bool(p_prompt_base.argmax() != p_prompt_lev.argmax()),
                          top1_changed_loop=bool(p_loop_base.argmax() != p_loop_lev.argmax()))
        log(f"  {label}: KL@prompt {out[label]['kl_at_prompt']:.5f} | "
            f"KL@loop-position {out[label]['kl_at_loop_position']:.5f} | "
            f"top1 changed: prompt {out[label]['top1_changed_prompt']} "
            f"loop {out[label]['top1_changed_loop']}")
    results[p["id"]] = out

json.dump(dict(status="complete", results=results), open(os.path.join(OUT, "probe_authority.json"), "w"), indent=1)
log("SUMMARY", json.dumps(results))
