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


class PresencePenalty:
    """The reference protocol's presence penalty, matching the published driver exactly.

    The published work and our own stage-2 replication applied a presence penalty of
    1.5 to EVERY arm, base and fixed alike: subtract `penalty` once per UNIQUE token
    already GENERATED (the prompt is not penalised). An arm run without it is not
    protocol-matched and its closed-thinking label cannot be compared with the published
    numbers.

    This mirrors src/driver2.py's PresencePenalty verbatim - including using torch.unique
    rather than a Python set, which is the difference between O(unique) on the GPU and
    O(n) in the interpreter (the first version looped in Python and made a 16k-token
    generation take hours).
    """

    def __init__(self, prompt_lengths, penalty: float):
        self.prompt_lengths = prompt_lengths
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        import torch
        if self.penalty <= 0:
            return scores
        for row in range(input_ids.shape[0]):
            generated = input_ids[row, self.prompt_lengths[row]:]
            if generated.numel():
                scores[row, torch.unique(generated)] -= self.penalty
        return scores


def greedy_rollout(model, encoded, max_new_tokens, stop_ids, tokenizer=None,
                    presence_penalty=0.0):
    """Greedy generation; returns (new_token_ids, info).

    `info` carries the observable behaviour: how many tokens were produced, whether the
    model closed its thinking, and how long the generation took.
    """
    import time

    started = time.time()
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                  top_p=None, top_k=None, eos_token_id=stop_ids, pad_token_id=stop_ids[-1])
    if presence_penalty and presence_penalty > 0:
        # the prompt is excluded from the penalty (see the class docstring)
        prompt_lengths = [encoded.shape[1]] * encoded.shape[0]
        kwargs["logits_processor"] = [PresencePenalty(prompt_lengths, presence_penalty)]
    with torch.no_grad():
        output = model.generate(encoded, **kwargs)
    new_tokens = output[0, encoded.shape[1]:]
    if tokenizer is None:
        raise ValueError("greedy_rollout needs the tokenizer: the decoded text carries "
                         "the closed-thinking label, and model.tokenizer does not exist")
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    info = dict(n=int(new_tokens.shape[0]), closed="</think>" in text, chars=len(text),
                seconds=round(time.time() - started, 1),
                close_position=close_position(text))
    return new_tokens, info


def close_position(text):
    """Character offset of the closing marker, or None if the thinking never closed.

    "closed" alone is too coarse: a model can close its thinking at token 50 or at
    token 8,100 of an 8,192-token budget, and those are not the same outcome. The
    position makes the difference measurable.
    """
    marker = "</think>"
    index = text.find(marker)
    return None if index < 0 else index


def load_pairs(path, keys=("id", "problem", "base_looped")):
    """The paired problems, with the base model's loop label attached."""
    data = json.load(open(path))
    rows = data["pairs"] if isinstance(data, dict) else data
    return [{k: row[k] for k in keys if k in row} for row in rows]
