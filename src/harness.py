"""Shared harness for the glial-regulator experiments.

Everything the CPU and GPU drivers need to touch a model the same way lives here:
one place to block the Triton kernels, one loader, one prompt builder, one rollout.
(Extracted because three files had copy-pasted it, and the copies had already drifted:
one of them still asked for CUDA on a CPU kernel.)
"""
import json
import os
import sys

import torch

MATH_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
EOS_IDS = ("<|im_end|>",)


def block_triton_kernels():
    """Make Qwen3.5's gated-delta-rule layers take the pure-torch path.

    The layers use flash-linear-attention's Triton kernels when the `fla` package is
    importable, and Triton cannot launch on CPU tensors ("Pointer argument cannot be
    accessed from Triton (cpu tensor?)"). transformers checks importlib.import_module
    ("fla") at import time, so blocking the import makes it fall back to its own torch
    implementation. Kaggle CPU images have no fla; this makes the local path work too.
    """
    sys.modules.setdefault("fla", None)


def load_model(model_id, device="cpu", dtype=None, attn_implementations=("eager", "sdpa")):
    """Load a causal LM, trying each attention implementation in turn.

    Returns (model, tokenizer). The attention implementation is explicit because
    Qwen3.5's linear-attention path is not covered by the usual `attn_implementation`
    knob -- it is the `fla` import that has to be blocked (see block_triton_kernels).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    last_error = None
    for implementation in attn_implementations:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, attn_implementation=implementation)
            model = model.to(device).eval()
            print(f"loaded {model_id} on {device} ({dtype}) "
                  f"with attn_implementation={implementation}", flush=True)
            return model, tokenizer
        except Exception as error:                       # noqa: BLE001
            last_error = error
            print(f"attn_implementation={implementation} failed: "
                  f"{type(error).__name__}: {str(error)[:160]}", flush=True)
    raise last_error


def eos_token_ids(tokenizer):
    ids = [tokenizer.convert_tokens_to_ids(e) for e in EOS_IDS]
    ids.append(tokenizer.eos_token_id)
    return [i for i in ids if i is not None]


def prompt_ids(tokenizer, problem, instruction=MATH_INSTR):
    """Chat prompt for one problem, with the instruction the reference protocol uses.

    Returns CPU token ids; the caller moves them to the model's device.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + instruction}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return tokenizer(text, return_tensors="pt").input_ids


def greedy_rollout(model, encoded, max_new_tokens, stop_ids, problem=None):
    """Greedy generation; returns (new_token_ids, info).

    `info` carries the observable behaviour: how many tokens were produced, whether the
    model closed its thinking, and how long the generation took.
    """
    import time

    started = time.time()
    with torch.no_grad():
        output = model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                eos_token_id=stop_ids, pad_token_id=stop_ids[-1])
    new_tokens = output[0, encoded.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True) \
        if hasattr(model, "tokenizer") else ""
    info = dict(n=int(new_tokens.shape[0]), closed="</think>" in text, chars=len(text),
                seconds=round(time.time() - started, 1))
    return new_tokens, info


def load_pairs(path, keys=("id", "problem", "base_looped")):
    """The paired problems, with the base model's loop label attached."""
    data = json.load(open(path))
    rows = data["pairs"] if isinstance(data, dict) else data
    return [{k: row[k] for k in keys if k in row} for row in rows]
