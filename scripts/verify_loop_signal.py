"""Does the loop direction separate looping contexts from healthy ones, cheaply?

This is the gate on Stage B. Training a context-reading regulator needs labels: which
contexts are looping and which are not. We have 18 labelled problems and rollout-based
labelling costs ~410 s per looping problem, so 1000 labels would be ~114 GPU-hours.

The hypothesis: the loop has a MECHANISM - the published fix removes one direction d
from the MLP output of layers 15 and 19 - so the component along d should be a signature
of the loop state, available after a SHORT generation instead of a full one.

This script measures the projection <down_proj_out[t], d_hat> at both layers over a
short greedy generation (and over the prompt alone), then reports how well the score
separates the known loopers from the known healthy controls.

    ~/.venvs/glia/bin/python scripts/verify_loop_signal.py --tokens 128
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from lever import build_lever_from_checkpoints
from loop_signal import LoopSignalProbe, loop_direction, separation, summarise

BASE = os.path.join(ROOT, "models", "base-hub-current")
EDITED = os.path.join(ROOT, "models", "edited-authors")
PAIRS = os.path.join(ROOT, "s1_pairs.json")


def load_base_and_directions():
    """The base model (measured) and the loop direction per layer, in ONE pass.

    The direction is a normalised column of the delta D = W_edited - W_base, so the
    edited checkpoint plus the base's weight matrices are enough. MEMORY: the base is
    loaded ONCE and kept - loading it a second time cost 3.2 GB that torch's allocator
    never released, and the OOM killer took the process (exit 137).
    """
    print(f"loading edited from {EDITED}", flush=True)
    edited, _ = load_model(EDITED, device="cpu", dtype=torch.float32)
    directions = {}
    for layer in (15, 19):
        edited_weights = edited.model.layers[layer].mlp.down_proj.weight.detach()
        torch.save(edited_weights, os.path.join(ROOT, "lab", f"edited_L{layer}.pt"))
        del edited_weights
    del edited
    gc.collect()

    print(f"loading base from {BASE} (for the delta and the measurement)", flush=True)
    base, tok = load_model(BASE, device="cpu", dtype=torch.float32)
    for layer in (15, 19):
        edited_weights = torch.load(os.path.join(ROOT, "lab", f"edited_L{layer}.pt"),
                                    weights_only=True)
        base_weights = base.model.layers[layer].mlp.down_proj.weight.detach()
        delta = (edited_weights.float() - base_weights.float())
        columns = delta[:, delta.norm(dim=0) > 0.05]
        directions[layer] = loop_direction(columns.numpy())
        del delta, columns, edited_weights
    gc.collect()
    return base, tok, directions


def measure_problem(base, tok, probes, problem, tokens, stop_ids):
    """One problem: the loop signature over the prompt alone and over a short rollout."""
    encoded = prompt_ids(tok, problem["problem"])
    # prompt-only signature: ONE forward pass over the prompt, before any generation.
    # (Measuring it after probe.reset() but before a forward pass returns None for every
    # problem - which silently produced a meaningless AUC of 0.000.)
    for probe in probes.values():
        probe.reset()
    with torch.no_grad():
        base(encoded)
    prompt_only = {layer: summarise(probe.stack()) for layer, probe in probes.items()}
    # then the generated-trajectory signature
    for probe in probes.values():
        probe.reset()
    _, info = greedy_rollout(base, encoded, tokens, stop_ids, tokenizer=tok)
    row = dict(id=problem["id"], base_looped=str(problem.get("base_looped")),
               chars=info["chars"], closed=info["closed"], n=info["n"],
               prompt_only=prompt_only, generated={}, scores={})
    for layer, probe in probes.items():
        summary = summarise(probe.stack())
        row["generated"][layer] = summary
        row["scores"][str(layer)] = summary["mean_abs"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--out", default=os.path.join(ROOT, "lab", "loop_signal.json"))
    args = parser.parse_args()

    block_triton_kernels()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    # MEMORY: two fp32 0.8B checkpoints on this 6.3 GB box is an OOM kill, so the two
    # checkpoints are loaded ONE AT A TIME: the edited one only to extract the delta's
    # direction, then it is freed before the base model (the one we actually measure) is
    # loaded.
    base, tok, directions = load_base_and_directions()
    print(f"loop directions: "
          f"{ {k: [round(float(x), 4) for x in v[:3]] for k, v in directions.items()} }",
          flush=True)

    probes = {}
    for layer, direction in directions.items():
        probes[layer] = LoopSignalProbe(base.model.layers[layer].mlp.down_proj, direction)

    pairs = load_pairs(PAIRS)
    stop_ids = eos_token_ids(tok)
    rows = []
    for index, problem in enumerate(pairs):
        row = measure_problem(base, tok, probes, problem, args.tokens, stop_ids)
        rows.append(row)
        print(f"[{index+1}/{len(pairs)}] {row['id']}: looped={row['base_looped']} "
              f"closed={row['closed']} n={row['n']} "
              f"score={ {k: round(v, 4) for k, v in row['scores'].items()} }", flush=True)

    def as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    loopers = [r for r in rows if as_bool(r["base_looped"])]
    healthy = [r for r in rows if not as_bool(r["base_looped"])]
    results = {}
    for layer in directions:
        loop_scores = [r["generated"][layer] for r in loopers]
        healthy_scores = [r["generated"][layer] for r in healthy]
        prompt_loop = [r["prompt_only"][layer] for r in loopers]
        prompt_healthy = [r["prompt_only"][layer] for r in healthy]
        results[str(layer)] = dict(
            generated=separation(loop_scores, healthy_scores),
            prompt_only=separation(prompt_loop, prompt_healthy))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=rows, separation=results, tokens=args.tokens),
              open(args.out, "w"), indent=1)

    print("\n=== SEPARATION (loopers vs healthy) ===")
    for layer, entry in results.items():
        gen, prompt = entry["generated"], entry["prompt_only"]
        print(f"L{layer}: generated AUC {gen['auc']:.3f} "
              f"(loopers {gen['mean_looper']:.4f} vs healthy {gen['mean_healthy']:.4f}) | "
              f"prompt-only AUC {prompt['auc']:.3f} "
              f"(loopers {prompt['mean_looper']:.4f} vs healthy {prompt['mean_healthy']:.4f})")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
