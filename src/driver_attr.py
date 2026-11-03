"""Session A: identification kernel. GATE 1 (attribution vs known ground truth) ->
language-drift attribution -> push screen (top candidates, dose 0.8) -> recipes.

Self-gating: GATE 1 failure aborts before the screen. Results incrementally.
"""
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribute import attribute_layer, finite_diff_rank
from push_screen import (DRIFT_PROMPTS, INDO_MARKERS, ORDINARY_PROMPTS, generate_with_push,
                         is_drift, is_garbled, ppl_delta, residual_norms)

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
DEADLINE = float(os.environ.get("DEADLINE_S", "4800"))
N_SCREEN = int(os.environ.get("N_SCREEN", "24"))
DRIFT_N = int(os.environ.get("DRIFT_N", "10"))
ORD_N = int(os.environ.get("ORD_N", "4"))
t0 = time.time()


def guard():
    time.sleep(DEADLINE)
    json.dump({"status": "deadline_fired"}, open(os.path.join(OUT, "meta_attr.json"), "w"))
    os._exit(0)


threading.Thread(target=guard, daemon=True).start()


def log(*a):
    print(f"[attr {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).cuda()
model.eval()
# attribution phase runs in FP32: the arange-weighted objective overflows fp16 backward
# (grad norm ~6297 at fp32; fp16 caps at 65504 and produced NaN). Generation later
# switches back to fp16 (reverting r_L differences are negligible; noted as deviation).
torch.backends.cuda.enable_flash_sdp(False)   # T4: no flash
log("model loaded")
log("warmup forward")
model(torch.randint(100, 1000, (1, 64)).cuda())
log("warmup done")

# calibration norms
r_L = {L: residual_norms(model, tok, L) for L in (15, 19, 21, 23)}
model_half_rL = dict(r_L)   # norms computed in fp32 (attribution) and reused for fp16 screen
log("r_L:", r_L)

# GATE 1: loop attribution vs known ground truth
import numpy as np
known = {15: np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "ground_truth", "neurons_L15.npy")).tolist(),
         19: np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "ground_truth", "neurons_L19.npy")).tolist()}
s1 = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "s1_pairs.json")))
pairs = s1["pairs"][:6]
variant = "autograd"
overlap_detail = {}
for L in (15, 19):
    neurons = torch.arange(3584, device="cuda")
    r_L_at = r_L.get(L, 24.0)
    scores_total = torch.zeros(3584)
    for p in pairs[:3]:
        trait = p["problem"] + "\n\nThe answer is:\n" + (p.get("healthy_completion") or "The answer is correct.")
        plain = p["problem"] + "\n\nI'm not sure how to approach this problem."
        ids_t = tok(trait, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        ids_p = tok(plain, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        s = attribute_layer(model, L, ids_t, ids_p, r_L_at, neurons)
        scores_total += s.cpu()
    top10 = scores_total.topk(10).indices.tolist()
    ov = len(set(top10) & set(known[L]))
    overlap_detail[L] = dict(top10=top10, overlap=ov)
    log(f"GATE 1 layer {L}: top-10 overlap with known = {ov}/10")
gate_pass = sum(d["overlap"] for d in overlap_detail.values()) >= 3
log("GATE 1:", "PASS" if gate_pass else "FAIL", "| variant:", variant)
if not gate_pass:
    # recorded fallback: finite difference on top-200 of the (possibly degenerate) ranking
    variant = "finite_diff"
    log("GATE 1 failed on autograd variant - trying finite difference on 200 candidates")
    overlap_detail = {}
    for L in (15, 19):
        neurons = torch.arange(3584, device="cuda")
        r_L_at = r_L.get(L, 24.0)
        # rank by first pass, then re-rank top-200 by finite difference
        ids_t = tok(pairs[0]["problem"] + "\n\nThe answer is:\n42", return_tensors="pt",
                    truncation=True, max_length=512).input_ids.cuda()
        ids_p = tok(pairs[0]["problem"] + "\n\nI'm not sure how to approach this.",
                    return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        pre = attribute_layer(model, L, ids_t, ids_p, r_L_at, neurons)
        cand = pre.topk(200).indices
        fd = finite_diff_rank(model, L, ids_t, ids_p, r_L_at, cand)
        ranked = sorted(fd, key=fd.get, reverse=True)
        top10 = ranked[:10]
        ov = len(set(top10) & set(known[L]))
        overlap_detail[L] = dict(top10=top10, overlap=ov)
        log(f"finite-diff GATE 1 layer {L}: overlap {ov}/10")
    gate_pass = sum(d["overlap"] for d in overlap_detail.values()) >= 3
    log("GATE 1 (finite_diff):", "PASS" if gate_pass else "FAIL")
json.dump(dict(gate1_pass=bool(gate_pass), variant=variant, overlap=overlap_detail,
               r_L=r_L), open(os.path.join(OUT, "gate1.json"), "w"), indent=1)
if not gate_pass:
    json.dump({"status": "bad_attribution"}, open(os.path.join(OUT, "meta_attr.json"), "w"))
    raise SystemExit("GATE 1 failed - identification pipeline not validated, aborting")

# language-drift attribution (new behavior; no ground truth)
trait_prompt = ("Reply to the following in Indonesian.\n\nWhat is the capital of France?"
                "\n\nReply (in Indonesian):")
trait_ans = (" Ibu kota Prancis adalah Paris. Saya akan menjawab pertanyaan ini dalam "
             "bahasa Indonesia.")
plain_ans = " The capital of France is Paris."
drift_scores = {}
for L in (19, 21, 23):
    ids_t = tok(trait_prompt + trait_ans, return_tensors="pt", truncation=True,
                max_length=512).input_ids.cuda()
    ids_p = tok(trait_prompt + plain_ans, return_tensors="pt", truncation=True,
                max_length=512).input_ids.cuda()
    r_L_at = r_L.get(L, 24.0)
    s = attribute_layer(model, L, ids_t, ids_p, r_L_at, torch.arange(3584, device="cuda"))
    drift_scores[L] = s.cpu()
    log(f"drift attribution layer {L} done")

# candidates: global top-N_SCREEN across layers
cands = []
for L, s in drift_scores.items():
    for i in s.topk(N_SCREEN).indices.tolist():
        cands.append(dict(layer=L, neuron=i, score=float(s[i])))
cands.sort(key=lambda c: -c["score"])
cands = cands[:N_SCREEN]
log("candidates:", [(c["layer"], c["neuron"]) for c in cands])

# push screen (fp16 for generation speed)
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda()
model.eval()
res_path = os.path.join(OUT, "screen_results.jsonl")
base_drift = []
for p in DRIFT_PROMPTS[:DRIFT_N]:
    out = generate_with_push(model, 19, tok, [p], neuron=0, dose=0.0, r_L=r_L.get(19, 24.0))[0]
    base_drift.append(is_drift(out))
base_rate = sum(base_drift) / max(len(base_drift), 1)
log(f"unsteered drift rate: {base_rate:.2f}")
random_controls = []
rng = torch.Generator().manual_seed(0)
for _ in range(3):
    L = [19, 21, 23][int(torch.randint(0, 3, (1,), generator=rng))]
    n = int(torch.randint(0, 3584, (1,), generator=rng))
    outs = generate_with_push(model, L, tok, DRIFT_PROMPTS[:DRIFT_N], neuron=n,
                              dose=0.8, r_L=r_L.get(L, 24.0))
    random_controls.append(sum(is_drift(o) for o in outs) / len(outs))
log("random control drift rates:", [round(x, 2) for x in random_controls])

recipes = []
for c in cands:
    L, n = c["layer"], c["neuron"]
    outs_d = generate_with_push(model, L, tok, DRIFT_PROMPTS[:DRIFT_N], neuron=n,
                                dose=0.8, r_L=r_L.get(L, 24.0))
    outs_o = generate_with_push(model, L, tok, ORDINARY_PROMPTS[:ORD_N], neuron=n,
                                dose=0.8, r_L=r_L.get(L, 24.0))
    drift_rate = sum(is_drift(o) for o in outs_d) / len(outs_d)
    garble = sum(is_garbled(o) for o in outs_d + outs_o) / (len(outs_d) + len(outs_o))
    a = round(drift_rate * DRIFT_N)
    b = DRIFT_N - a
    cc = round(base_rate * DRIFT_N)
    d = DRIFT_N - cc
    from push_screen import fisher_exact_2x2
    fp = fisher_exact_2x2(a, b, cc, d)
    rec = dict(layer=L, neuron=n, dose=0.8, drift_rate=drift_rate,
               base_rate=base_rate, garble_rate=garble, fisher_p=fp,
               qualifies=bool(fp < 0.001 / 40 and drift_rate > max(random_controls)))
    recipes.append(rec)
    with open(res_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"cand L{L} N{n}: drift {drift_rate:.2f} garble {garble:.2f} p={fp:.2e} "
        f"{'QUALIFIES' if rec['qualifies'] else ''}")

qualified = [r for r in recipes if r["qualifies"]]
json.dump(dict(model="Qwen/Qwen3.5-0.8B", behavior="language-drift",
               n_candidates=len(cands), n_qualified=len(qualified),
               base_rate=base_rate, random_controls=random_controls,
               recipes=recipes),
          open(os.path.join(OUT, "recipes", "language_drift.json"), "w"), indent=1)
summary = dict(status="complete", variant=variant, gate1=bool(gate_pass),
               n_screen=len(cands), n_qualified=len(qualified),
               elapsed_s=time.time() - t0)
json.dump(summary, open(os.path.join(OUT, "meta_attr.json"), "w"), indent=1)
log("SUMMARY", json.dumps(summary))
