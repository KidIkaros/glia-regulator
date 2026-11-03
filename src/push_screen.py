"""Push screen for the language-drift behavior: push a candidate neuron at a dose,
greedy-generate on held-out drift prompts and ordinary prompts, score objectively.

Drift check (objective, no judge): the reply must contain Indonesian markers AND few
long English words. Side-effect panel: ppl delta on a fixed small corpus.
"""
import math
import re

import torch

INDO_MARKERS = [" yang", " dan", " saya", " untuk", " tidak", " bisa", " adalah"]
DRIFT_PROMPTS = [
    "What is the capital of France?",
    "Explain how rain forms.",
    "Tell me a short story about a dog.",
    "Why is the sky blue?",
    "What do cats eat?",
    "How do you boil an egg?",
    "Describe a beach.",
    "What is 2 + 2?",
    "Name three colors.",
    "How do plants grow?",
    "What season comes after summer?",
    "Describe a mountain.",
    "Why do we sleep?",
    "What is a river?",
    "Tell me about the moon.",
    "How does a bicycle work?",
    "What do bees make?",
    "Name a famous scientist.",
    "What is music?",
    "How do birds fly?",
]
ORDINARY_PROMPTS = [
    "Write a formal apology for missing a meeting.",
    "Draft an email requesting a day off.",
    "Explain recursion to a beginner.",
    "Summarize the plot of Hamlet.",
    "Give tips for public speaking.",
    "Describe your debugging process.",
]
CALIB_TEXT = ("The river runs past the old mill and into the valley, where the fields "
              "are green in spring and the roads are quiet. Farmers walk to the market "
              "at dawn, and the town keeps its records in long ledgers. ") * 60


def is_drift(text: str) -> bool:
    t = text.lower()
    hits = sum(m in t for m in INDO_MARKERS)
    ascii_words = len(re.findall(r"[A-Za-z]{3,}", text))
    return hits >= 2 and ascii_words < 5


def is_garbled(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]{3,}", text)) < 3 and len(text.strip()) > 40


@torch.no_grad()
def residual_norms(model, tok, layer_idx, text=CALIB_TEXT):
    """Median per-token norm of the residual leaving `layer_idx`, token 0 skipped."""
    import types
    captured = {}
    layer = model.model.layers[layer_idx]

    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["h"] = out.detach().float()

    h = layer.register_forward_hook(hook)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()
    model(ids)
    h.remove()
    norms = captured["h"][0].norm(dim=-1)
    return float(norms[1:].median())


@torch.no_grad()
def generate_with_push(model, layer_idx, tok, prompts, neuron, dose, r_L, max_new=150):
    """Greedy-generate with a constant push on `neuron` at `dose` (rel units), applied
    on every token via a pre-hook on down_proj's input."""
    down = model.model.layers[layer_idx].mlp.down_proj
    W = down.weight.detach()

    def pre_hook(module, args):
        act = args[0]
        scale = torch.ones(act.shape[-1], device=act.device, dtype=act.dtype)
        scale[neuron] = scale[neuron] + torch.tensor(
            dose * r_L / W[:, neuron].norm().clamp_min(1e-6).item(), dtype=act.dtype,
            device=act.device)
        return (act * scale,) + tuple(args[1:])

    handle = down.register_forward_pre_hook(pre_hook)
    outs = []
    try:
        for p in prompts:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            enc = tok(ids, return_tensors="pt").input_ids.cuda()
            gen = model.generate(enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, enc.shape[1]:], skip_special_tokens=True)
            outs.append(text)
    finally:
        handle.remove()
    return outs


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact for [[a,b],[c,d]] via hypergeometric (stdlib only)."""
    from math import comb
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    p = 0.0
    for k in range(lo, hi + 1):
        pr = comb(col1, k) * comb(n - col1, row1 - k) / comb(n, row1)
        if abs(k - a) <= 1e-9 + max(1, abs(a - (lo + hi) / 2)):
            p += pr
    return min(1.0, p)


@torch.no_grad()
def ppl_delta(model, tok, base_ppl, texts):
    ids = tok("".join(texts[:3]), return_tensors="pt", truncation=True,
              max_length=1024).input_ids.cuda()
    logits = model(ids).logits
    nll = torch.nn.functional.cross_entropy(
        logits[0, :-1].float(), ids[0, 1:], reduction="mean")
    return float(math.exp(nll) / base_ppl - 1)   # relative delta
