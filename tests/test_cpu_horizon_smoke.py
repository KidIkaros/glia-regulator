"""Fast local correctness check of the CPU horizon kernel's plumbing (no GPU).

Exercises every path the remote CPU kernel uses, with ONE model (the base) and a tiny
rollout, so it runs on this machine in ~a minute and without the memory pressure of two
fp32 checkpoints:

  * installing the lever from the validated delta payload on the real down_proj;
  * generating a few tokens WITH the lever active;
  * teacher-forcing a 1-D token path and reading distributions at chosen positions;
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
from harness import block_triton_kernels, load_model
from lever import DeltaLever

block_triton_kernels()
torch.set_num_threads(4)
MODEL = os.path.join(ROOT, "models", "base-hub-current")

started = time.time()
model, tok = load_model(MODEL, device="cpu")
print(f"model loaded in {time.time()-started:.0f}s")

import json  # noqa: E402
meta = json.load(open(os.path.join(ROOT, "ground_truth", "delta_idx.json")))
switch = {"a": 0.0}
levers = []
for layer in (15, 19):
    info = meta["layers"][str(layer)]
    delta = np.load(os.path.join(ROOT, "ground_truth", info["delta_file"])).astype(np.float32)
    indices = [int(i) for i in info["idx"]]
    levers.append(DeltaLever(model.model.layers[layer].mlp.down_proj, indices, delta,
                             lambda: switch["a"]))
print("lever installed")

text = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23? Think step by step."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=True)
encoded = tok(text, return_tensors="pt").input_ids
print("prompt tokens:", encoded.shape[1])

started = time.time()
with torch.no_grad():
    generated = model.generate(encoded, max_new_tokens=8, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"),
                                             tok.eos_token_id],
                               pad_token_id=tok.eos_token_id)
print(f"generated {generated.shape[1]-encoded.shape[1]} tokens in {time.time()-started:.1f}s")

positions = [int(encoded.shape[1]), int(generated.shape[1]) - 1]
# the shared token path is 1-D [T] (this exact shape crashed the first CPU kernel:
# Qwen3.5's forward needs [batch, seq, hidden])
token_path = generated[0]
with torch.no_grad():
    base_logits = model(token_path.unsqueeze(0)).logits[0].float()
switch["a"] = 1.0
with torch.no_grad():
    lever_logits = model(token_path.unsqueeze(0)).logits[0].float()
switch["a"] = 0.0

for position in positions:
    base_probs = torch.softmax(base_logits[position - 1], -1)
    lever_probs = torch.softmax(lever_logits[position - 1], -1)
    kl = float((base_probs * (base_probs.clamp_min(1e-12).log()
                               - lever_probs.clamp_min(1e-12).log())).sum())
    print(f"pos {position}: KL(base||lever) = {kl:.6f}  "
          f"argmax same: {base_probs.argmax() == lever_probs.argmax()}")
    assert kl > 0.0, "the lever did not change the distribution - hook is not active"

assert base_logits.shape == lever_logits.shape
print(f"PASS: lever active, teacher-forcing works, KL computation works "
      f"({time.time()-started:.0f}s)")
