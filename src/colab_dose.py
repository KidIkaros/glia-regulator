"""Resumable driver for the Colab run: does the runtime lever reproduce the fix's
behaviour over a full 16k-token generation, in fp32?

Why this run exists. The lever is logit-equivalent to the authors' weight edit at one
step (max |delta logits| = 0.164) and matches the fix's greedy trajectory for 875 of
1024 tokens, yet on a T4 in fp16 it did NOT close a loop that the weight edit closes
after 4,555 characters. The suspected cause is arithmetic: the edited checkpoint
computes one fp16 matmul, the lever two plus an add, so the paths differ ~1e-3 per step
and the difference compounds over a chaotic 16k-token generation. This run removes that
confound by working in fp32.

Arms, in priority order (a dropped session must not lose the important ones):
  1. fix_reference : the edited checkpoint's own greedy rollout (the target behaviour)
  2. lever_alpha1  : base + lever(alpha=1)
  3. lever_alpha1_5: base + lever(alpha=1.5) - does over-correction break the loop?

Each arm appends one JSON line to RESULTS_PATH as soon as it finishes, and an arm
already present in that file is SKIPPED, so re-running after a Colab disconnect
resumes instead of restarting. Nothing is written back into the model: the base
weights stay frozen and the only intervention is the lever's runtime correction.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from lever import DeltaLever, build_lever_from_checkpoints

block_triton_kernels()

MODEL_ID = "Qwen/Qwen3.5-0.8B"
EDITED_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
ROLLOUT = int(os.environ.get("ROLLOUT", "16384"))
# the reference protocol (published work + our stage-2 replication) used 1.5 on every arm
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "1.5"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "1.0 1.5").split()]
DTYPE = torch.float32                      # the point of this run: no fp16 rounding
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/content/drive/glia_colab/results.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
started = time.time()


def log(*parts):
    print(f"[colab {round(time.time()-started)}s]", *parts, flush=True)


def completed_arms():
    """Arms already recorded, so a relaunch resumes instead of repeating work."""
    if not os.path.exists(RESULTS_PATH):
        return {}
    done = {}
    for line in open(RESULTS_PATH):
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
                done[row["arm"]] = row
            except json.JSONDecodeError:
                continue
    return done


def record(arm, payload):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as handle:
        handle.write(json.dumps(dict(arm=arm, **payload)) + "\n")
    log(f"recorded arm {arm}")


log(f"device={DEVICE} dtype={DTYPE} rollout={ROLLOUT} results -> {RESULTS_PATH}")
done = completed_arms()
if done:
    log(f"resuming: already have {sorted(done)}")

base, tok = load_model(MODEL_ID, device=DEVICE, dtype=DTYPE)
edited, _ = load_model(EDITED_ID, device=DEVICE, dtype=DTYPE)
spec, diagnostics = build_lever_from_checkpoints(base, edited)
switch = {"a": 0.0}
levers = [DeltaLever(base.model.layers[layer].mlp.down_proj, spec[layer][0],
                     spec[layer][1], lambda: switch["a"]) for layer in spec]

def find_pairs():
    """The pairs file may sit next to this script or one level up (payload layouts differ)."""
    for candidate in (os.path.join(HERE, "s1_pairs.json"),
                      os.path.join(os.path.dirname(HERE), "s1_pairs.json")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("s1_pairs.json not found next to the driver or one level up")


pairs = load_pairs(find_pairs())
problem = pairs[0]
encoded = prompt_ids(tok, problem["problem"]).to(DEVICE)
stop_ids = eos_token_ids(tok)
log(f"problem {problem['id']}; prompt {encoded.shape[1]} tokens")

reference_tokens = None
if "fix_reference" in done:
    reference_tokens = torch.tensor(done["fix_reference"]["token_ids"], device=DEVICE)
    log("using the recorded reference rollout")
else:
    log("arm fix_reference: the edited checkpoint's own rollout")
    tokens, info = greedy_rollout(edited, encoded, ROLLOUT, stop_ids, tokenizer=tok,
                                  presence_penalty=PRESENCE_PENALTY)
    reference_tokens = tokens
    record("fix_reference", dict(info=info, token_ids=tokens.tolist(),
                                  diagnostics=diagnostics))

for alpha in ALPHAS:
    arm = f"lever_alpha{alpha:g}".replace(".", "_")
    if arm in done:
        log(f"arm {arm} already recorded - skipping")
        continue
    switch["a"] = alpha
    log(f"arm {arm}: base + lever(alpha={alpha})")
    tokens, info = greedy_rollout(base, encoded, ROLLOUT, stop_ids, tokenizer=tok,
                                  presence_penalty=PRESENCE_PENALTY)
    matched = None
    limit = min(len(tokens), len(reference_tokens))
    for index in range(limit):
        if int(tokens[index]) != int(reference_tokens[index]):
            matched = index
            break
    record(arm, dict(info=info, matched_prefix=matched,
                     identical_to_fix=matched is None and len(tokens) == len(reference_tokens),
                     token_ids=tokens.tolist()))
    log(f"arm {arm}: matched {matched} tokens with the fix "
        f"(arm {len(tokens)} tokens, fix {len(reference_tokens)} tokens) | closed={info['closed']}")

summary = {}
for arm, row in completed_arms().items():
    summary[arm] = dict(n=row.get("info", {}).get("n"),
                        closed=row.get("info", {}).get("closed"),
                        chars=row.get("info", {}).get("chars"),
                        matched_prefix=row.get("matched_prefix"))
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=1))
