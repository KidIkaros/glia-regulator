"""Family portability probe: does the thinking-loop failure exist in Qwen3.5-2B / 4B?

Ticket: .scratch/companion-substrate/issues/03-family-portability-run.md

Protocol-matched to the 0.8B measurement (greedy, presence penalty 1.5, 8192-token cap,
the same GSM8K/MATH problems). Records closed / close_position / text tail per
(model, problem). No lever, no activations - this is the base-behaviour probe that
decides whether edit work is even needed on the bigger members.

Pre-registered (from the ticket):
  prediction: 2B/4B loop less than 0.8B (16/18 looped); some closure failures expected.
  kill: if neither model shows closure failure above the base rate on the same protocol,
        the loop is a 0.8B-specific pathology - record it; portability narrows to
        "the method transfers, the failure does not".

Env: MODELS (space-separated HF ids), N_PROBLEMS, ROLLOUT, PROBLEM_CHUNK, RESULTS_PATH.

Structure (clean-code/functional-design): pure helpers live in runlog.py; this file holds
only the impure edges (model loading, generation, file append) behind a thin main().
Nothing executes at import, so tests can import the module.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from runlog import append_jsonl, as_bool, find_pairs, missing_keys, read_jsonl

DEFAULT_ROLLOUT = 8192          # the cap every published measurement used
TEXT_TAIL_CHARS = 2000          # enough to see the loop's shape, not the whole rollout
PRESENCE_PENALTY = 1.5          # protocol-matched to the published runs
STARTED = time.time()


def log(*parts):
    print(f"[probe {round(time.time() - STARTED)}s]", *parts, flush=True)


def measure(model, tok, problem, rollout=DEFAULT_ROLLOUT):
    """One (model, problem) rollout -> a result row. Impure: runs the model."""
    device = next(model.parameters()).device
    encoded = prompt_ids(tok, problem["problem"]).to(device)
    tokens, info = greedy_rollout(model, encoded, rollout, eos_token_ids(tok),
                                  tokenizer=tok, presence_penalty=PRESENCE_PENALTY)
    text = tok.decode(tokens, skip_special_tokens=True)
    return dict(base_looped=as_bool(problem.get("base_looped", False)),
                closed=bool(info["closed"]),
                close_position=info.get("close_position"),
                n=info["n"], chars=info["chars"], seconds=info["seconds"],
                text_tail=text[-TEXT_TAIL_CHARS:])


def default_loader(model_id):
    cuda = torch.cuda.is_available()
    return load_model(model_id, device="cuda" if cuda else "cpu",
                      dtype=torch.float16 if cuda else torch.float32)


def run_chunk(models, pairs, results_path, chunk, rollout=DEFAULT_ROLLOUT, loader=None):
    """Resume-aware execution of up to `chunk` work items; returns rows written.

    `loader(model_id) -> (model, tokenizer)` is injected so the loop can be exercised
    with a fake model in tests.
    """
    loader = loader or default_loader
    rows = read_jsonl(results_path)
    work = missing_keys(models, pairs, rows)
    log(f"{len(work)} items pending, chunk={chunk} -> {results_path}")
    written, current_model_id, current = 0, None, None
    for model_id, problem in work:
        if written >= chunk:
            log("chunk budget reached - run again to continue")
            break
        if current is None or model_id != current_model_id:
            current_model_id, current = model_id, loader(model_id)
            log(f"loaded {model_id}")
        model, tok = current
        key = f"{model_id}_{problem['id']}"
        row = measure(model, tok, problem, rollout)
        row.update(model=model_id, id=problem["id"])
        append_jsonl(results_path, row, key)
        log(f"recorded {key}: closed={row['closed']} n={row['n']} "
            f"close@{row['close_position']}")
        written += 1
    return written


def main():
    block_triton_kernels()
    models = os.environ.get("MODELS", "Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B").split()
    pairs = load_pairs(find_pairs(HERE, os.path.dirname(HERE)))[
        : int(os.environ.get("N_PROBLEMS", "6"))]
    run_chunk(models, pairs,
              results_path=os.environ.get("RESULTS_PATH",
                                          "/content/glia_out/family_probe.jsonl"),
              chunk=int(os.environ.get("PROBLEM_CHUNK", "2")),
              rollout=int(os.environ.get("ROLLOUT", str(DEFAULT_ROLLOUT))))


if __name__ == "__main__":
    main()
