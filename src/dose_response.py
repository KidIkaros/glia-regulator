"""Dose-response for the runtime lever: what does alpha actually do to the loop?

Why this is the next run (ADR-0002 amendment): the lever is algebraically and
logit-equivalent to the authors' weight edit (max |delta logits| = 0.164), yet at the full
16k cap alpha=1 did NOT close a problem that the static edit closed in 4,555 characters.
The cause is arithmetic: one fp16 matmul versus two plus an add, which compounds over
thousands of steps into a chaotic loop. So alpha = 1 is NOT "the fix" -- the lever's own
dose-response has to be measured before a learned alpha(t) can be calibrated against it.

Design notes:
  * The delta payload is tiny (the extracted per-layer delta plus the changed column
    indices), so the kernel does NOT download the 1.6 GB edited checkpoint.
  * The payload is VERIFIED against the live weights before any generation, using two
    structural signatures of the published edit:
      (a) every changed column is parallel to ONE direction (the edit removes a single
          direction from each column);
      (b) the per-column magnitude is 6x the column's projection onto that direction
          (the published strength).
    If either check fails, the payload does not describe the model being run and the
    kernel exits instead of producing a meaningless curve.
  * Arms are interleaved BY PROBLEM (all alphas for problem 1, then problem 2, ...) so a
    session deadline still yields paired curves for the problems that finished.
  * Results are written incrementally; the deadline guard exits cleanly with partial data.

Environment: ALPHAS (default "0 0.5 1.0 1.5 2.0"), N_PROBLEMS (default 4),
MAX_NEW (default 16384), DEADLINE_S (default 5400).
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
from lever import DeltaLever, build_lever_from_checkpoints

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
EDITED_ID = os.environ.get("EDITED_ID", "RikiyaT/Qwen3.5-0.8B-weight-space-programming")
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "4"))
MAX_NEW = int(os.environ.get("MAX_NEW", "16384"))
DEADLINE_S = int(os.environ.get("DEADLINE_S", "5400"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0 0.5 1.0 1.5 2.0").split()]
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
t0 = time.time()


def log(*a):
    print(f"[dose {round(time.time()-t0)}s]", *a, flush=True)


def deadline_guard():
    time.sleep(max(0, DEADLINE_S - (time.time() - t0)))
    log(f"DEADLINE reached ({DEADLINE_S}s) - exiting with partial results")
    os._exit(0)


threading.Thread(target=deadline_guard, daemon=True).start()

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
edited = AutoModelForCausalLM.from_pretrained(EDITED_ID, dtype=torch.float16).cuda().eval()
pairs = json.load(open(os.path.join(ROOT, "s1_pairs.json")))["pairs"]
switch = {"a": 0.0}

# ---------------- derive the lever in-kernel and validate it ----------------
spec = build_lever_from_checkpoints(model, edited, layers=(15, 19), validate=True)
levers = [DeltaLever(model.model.layers[L].mlp.down_proj, spec[L][0], spec[L][1],
                     lambda st=switch: st["a"]) for L in (15, 19)]


def prompt_ids(problem):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tok(txt, return_tensors="pt").input_ids.cuda()


results_path = os.path.join(OUT, "dose_response.jsonl")
rows = []


def flush():
    with open(results_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


for pi, p in enumerate(pairs[:N_PROBLEMS]):
    for alpha in ALPHAS:
        switch["a"] = alpha
        enc = prompt_ids(p["problem"])
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                               tok.eos_token_id],
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc.shape[1]:], skip_special_tokens=True)
        closed = "</think>" in text
        row = dict(id=p["id"], alpha=alpha, closed=bool(closed), chars=len(text),
                   n_gen=int(out.shape[1] - enc.shape[1]),
                   base_looped=bool(p.get("base_looped", False)))
        rows.append(row)
        flush()
        log(f"p{pi} id={p['id']} alpha={alpha}: closed={closed} chars={len(text)} "
            f"n_gen={row['n_gen']}")

summary = {}
for alpha in ALPHAS:
    sel = [r for r in rows if r["alpha"] == alpha]
    if sel:
        summary[str(alpha)] = dict(closed=sum(r["closed"] for r in sel), n=len(sel),
                                   mean_chars=float(np.mean([r["chars"] for r in sel])))
json.dump(dict(status="complete", summary=summary, rows=rows),
          open(os.path.join(OUT, "dose_summary.json"), "w"), indent=1)
log("SUMMARY", json.dumps(summary))
