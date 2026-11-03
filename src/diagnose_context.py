"""Cheap diagnostic (one kernel, ~5 min GPU) answering two design questions for the
glial redesign:

Q1 CONTEXT-BLINDNESS: the glial reads MEAN-POOLED residuals over a whole document.
    If those pooled vectors are nearly identical across documents, the reader cannot
    be context-dependent -> the head saturates at a constant gate (observed). We
    measure pairwise cosine similarity across documents, document-level vs
    last-chunk-level pooling.

Q2 AUTHORITY: does a bounded per-neuron gain (|g| <= 0.5) move the model's next-token
    distribution at all? Sweep constant gains on the 128 verified loop neurons and
    measure KL(p_g || p_0) and the probability mass on a repetitive continuation.
    If |g| <= 0.5 barely moves the distribution, a bounded gate cannot emulate the
    static edit (which removes the loop DIRECTION at strength 6) and the redesign needs
    a different mechanism, not a bigger sigma.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glial import _resolve

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()


def log(*a):
    print(f"[diag {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda()
model.eval()
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

pairs = json.load(open(os.path.join(ROOT, "s1_pairs.json")))["pairs"]
gt15 = np.load(os.path.join(ROOT, "gt15.npy"))
gt19 = np.load(os.path.join(ROOT, "gt19.npy"))
neurons = {15: [int(x) for x in gt15], 19: [int(x) for x in gt19]}
log("neurons:", {k: len(v) for k, v in neurons.items()})

# ---------- Q1: pooled-residual context ----------
captured = {}
handles = []
for L in (4, 8, 12):
    _, layer = _resolve(model, f".layers.{L}")

    def mk(L):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[L] = out.detach().float()
        return hook
    handles.append(layer.register_forward_hook(mk(L)))

doc_vecs, chunk_vecs, doc_layers, chunk_layers = [], [], [], []
for p in pairs[:12]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": p["problem"] + MATH_INSTR}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    enc = tok(ids, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
    with torch.no_grad():
        model(enc)
    per_layer = {L: captured[L][0].mean(0) for L in (4, 8, 12)}
    per_layer_chunk = {L: captured[L][0, -64:].mean(0) for L in (4, 8, 12)}
    doc_layers.append(per_layer)
    chunk_layers.append(per_layer_chunk)
    doc_vecs.append(torch.cat([per_layer[L] for L in (4, 8, 12)]))
    chunk_vecs.append(torch.cat([per_layer_chunk[L] for L in (4, 8, 12)]))
for h in handles:
    h.remove()


def mean_pairwise_cos(vs):
    V = torch.stack(vs)
    V = V / V.norm(dim=1, keepdim=True)
    C = V @ V.T
    n = C.shape[0]
    off = (C.sum() - C.diag().sum()) / (n * (n - 1))
    return float(off)


doc_cos = mean_pairwise_cos(doc_vecs)
chunk_cos = mean_pairwise_cos(chunk_vecs)
log(f"Q1 raw pooled cosine: document-level {doc_cos:.4f} | last-64-token {chunk_cos:.4f}")

# Q1b: the READER's output variability across documents (what training depends on)
from glial import GlialConfig, GlialField
cfg = GlialConfig(behaviors=("loop",), read_layers=(4, 8, 12))
field = GlialField(cfg, d_model=1024, inter=3584, n_neurons={15: 64, 19: 64},
                   neuron_idx={15: torch.arange(64), 19: torch.arange(64)})
field = field.to("cuda")
zs = []
with torch.no_grad():
    for per_layer in doc_layers:
        field._state = None
        field._buf = {L: per_layer[L].unsqueeze(0).to("cuda") for L in (4, 8, 12)}
        field.step(1)
        zs.append(field._state[0].clone())
z_cos = mean_pairwise_cos(zs)
log(f"Q1b reader-output (z) pairwise cosine across documents: {z_cos:.4f}")

# ---------- Q2: authority of a bounded gain ----------
_, down15 = _resolve(model, ".layers.15.mlp.down_proj")
_, down19 = _resolve(model, ".layers.19.mlp.down_proj")


def apply_gain(gain):
    """Scale the 128 verified neurons' down_proj INPUT by (1+g) via pre-hooks."""
    hs = []
    for L, down in ((15, down15), (19, down19)):
        idx = neurons[L]

        def mk(idx):
            def hook(module, args, gain=gain):
                act = args[0]
                scale = torch.ones(act.shape[-1], device=act.device, dtype=act.dtype)
                scale[idx] = scale[idx] + gain
                act = act.clone()
                act[:, :, idx] = act[:, :, idx] * scale[idx]
                return (act,) + tuple(args[1:])
            return hook
        hs.append(down.register_forward_pre_hook(mk(idx)))
    return hs


probe = pairs[0]
ids = tok.apply_chat_template(
    [{"role": "user", "content": probe["problem"] + MATH_INSTR}], tokenize=False,
    add_generation_prompt=True, enable_thinking=True)
enc = tok(ids, return_tensors="pt").input_ids.cuda()
loop_probe = tok(" Wait, wait, let me check that again. Wait, wait,",
                return_tensors="pt", add_special_tokens=False).input_ids.cuda()


def dist_at_probe():
    with torch.no_grad():
        lg = model(enc, logits_to_keep=1).logits[0, -1].float()
    p = torch.softmax(lg, -1)
    loop_mass = float(p[loop_probe[0, 0]].item())
    return p, loop_mass


base_p, base_loop = dist_at_probe()
results = []
for g in (0.0, 0.25, 0.5, -0.5, -1.0):
    hs = apply_gain(g)
    p, lm = dist_at_probe()
    for h in hs:
        h.remove()
    kl = float((base_p * (base_p.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())
    results.append(dict(gain=g, kl_from_base=kl, loop_token_mass=lm,
                         loop_mass_delta=lm - base_loop))
    log(f"Q2 gain {g:+.2f}: KL {kl:.5f} loop-token mass {lm:.5f} (delta {lm-base_loop:+.5f})")

out = dict(status="complete", q1_doc_cosine=doc_cos, q1_chunk_cosine=chunk_cos,
           q1b_reader_z_cosine=z_cos,
           q2_authority=results, base_loop_token_mass=base_loop)
json.dump(out, open(os.path.join(OUT, "diag_context_authority.json"), "w"), indent=1)
log("SUMMARY", json.dumps(out))
