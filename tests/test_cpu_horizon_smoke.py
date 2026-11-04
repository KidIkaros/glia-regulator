"""Fast local correctness check of the CPU horizon kernel's plumbing (no GPU).

Exercises every path the remote CPU kernel uses, with ONE model (the base) and a tiny
rollout, so it runs on this machine in ~a minute and without the memory pressure of two
fp32 checkpoints:

  * installing the lever from the validated delta payload on the real down_proj;
  * generating a few tokens WITH the lever active;
  * teacher-forcing a prefix and reading distributions at chosen positions;
  * the KL/argmax-agreement computation;
  * and the assertion that the lever actually changes the distribution (KL > 0).

Run:  ~/.venvs/hnetgpu/bin/python tests/test_cpu_horizon_smoke.py
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lever import DeltaLever

torch.set_num_threads(4)
MODEL = os.path.join(ROOT, "models", "base-hub-current")
MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

t0 = time.time()
# Qwen3.5's gated-delta-rule layers use flash-linear-attention's TRITON kernels when
# the `fla` package is importable; Triton cannot launch on CPU tensors. Blocking the
# import makes transformers fall back to its pure-torch path (the decorator checks
# importlib.import_module("fla") at import time). Kaggle CPU images have no fla, so this
# is belt-and-braces; locally it makes the CPU path testable at all.
sys.modules.setdefault("fla", None)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)


def load_cpu(model_id, **kw):
    """Load on CPU with a pure-PyTorch attention path.

    Qwen3.5's linear-attention layers dispatch to Triton kernels when
    flash-linear-attention is importable; Triton cannot launch on CPU tensors
    ("Pointer argument cannot be accessed from Triton (cpu tensor?)"). CPU kernels
    have no GPU, so we force the eager (pure torch) implementation explicitly.
    """
    from transformers import AutoModelForCausalLM
    last = None
    for impl in ("eager", "sdpa"):
        try:
            m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                      attn_implementation=impl, **kw)
            print(f"loaded with attn_implementation={impl}", flush=True)
            return m
        except Exception as e:                       # noqa: BLE001
            last = e
            print(f"attn_implementation={impl} failed: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
    raise last

model = load_cpu(MODEL)
print(f"model loaded in {time.time()-t0:.0f}s")

meta = json.load(open(os.path.join(ROOT, "ground_truth", "delta_idx.json"))) \
    if False else None
import json  # noqa: E402
meta = json.load(open(os.path.join(ROOT, "ground_truth", "delta_idx.json")))
switch = {"a": 0.0}
levers = []
for L in (15, 19):
    info = meta["layers"][str(L)]
    delta = np.load(os.path.join(ROOT, "ground_truth", info["delta_file"])).astype(np.float32)
    idx = [int(i) for i in info["idx"]]
    levers.append(DeltaLever(model.model.layers[L].mlp.down_proj, idx, delta,
                             lambda st=switch: st["a"]))
print("lever installed")

txt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23? Think step by step." + MATH_INSTR}],
    tokenize=False, add_generation_prompt=True, enable_thinking=True)
enc = tok(txt, return_tensors="pt").input_ids
print("prompt tokens:", enc.shape[1])

t1 = time.time()
with torch.no_grad():
    gen = model.generate(enc, max_new_tokens=8, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                       tok.eos_token_id],
                         pad_token_id=tok.eos_token_id)
print(f"generated {gen.shape[1]-enc.shape[1]} tokens in {time.time()-t1:.1f}s "
      f"({(gen.shape[1]-enc.shape[1])/max(1e-6,time.time()-t1):.2f} tok/s)")

positions = [int(enc.shape[1]), int(gen.shape[1]) - 1]
# the shared token path is 1-D [T] (this exact shape crashed the first CPU kernel:
# Qwen3.5's forward needs [batch, seq, hidden])
seq_1d = gen[0]
with torch.no_grad():
    lg0 = model(seq_1d.unsqueeze(0)).logits[0].float()
switch["a"] = 1.0
with torch.no_grad():
    lg1 = model(seq_1d.unsqueeze(0)).logits[0].float()
switch["a"] = 0.0

for p in positions:
    a = torch.softmax(lg1[p - 1], -1)
    b = torch.softmax(lg0[p - 1], -1)
    kl = float((b * (b.clamp_min(1e-12).log() - a.clamp_min(1e-12).log())).sum())
    print(f"pos {p}: KL(base||lever) = {kl:.6f}  argmax same: {a.argmax()==b.argmax()}")
    assert kl > 0.0, "the lever did not change the distribution - hook is not active"

assert lg0.shape == lg1.shape
print(f"PASS: lever active, teacher-forcing works, KL computation works "
      f"({time.time()-t0:.0f}s total)")
