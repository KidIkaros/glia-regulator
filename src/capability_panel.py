"""A strengthened capability panel.

Why this exists. The current capability evidence is one KL sentence on 8 ordinary
prompts - far too thin to support a capability-preservation claim, and it cannot detect
the specific failure mode this project is at risk of: the intervention might not be
"removing a loop" but "making the model terse", which would improve every fluency
metric while destroying the ability to reason at length.

So the panel measures FIVE things, each chosen to break a different way:

  1. agreement   - on ordinary prompts, does the next-token argmax change at all?
                    (catches blunt distribution damage)
  2. kl          - the mean KL against the base model on those prompts
                    (catches subtle distortion the argmax hides)
  3. ppl         - perplexity on a fixed neutral passage (catches degradation on
                    ordinary text the model is not prompted for)
  4. verbosity   - mean generated length on the ordinary prompts, base vs intervened.
                    If the intervention mostly SHORTENS answers, it is not removing a
                    loop, it is truncating reasoning. This is the key check.
  5. specificity - on the loop-prone problems, the intervened model must close its
                    thinking while the *healthy* controls (which the base closes on its
                    own) must keep their content: we compare the intervened answer
                    against the base answer for the healthy controls and require the
                    final answer to still be present.

Every prompt set is embedded, so the panel is deterministic and needs no download, and
the same panel is used by the dose sweep and by the alpha(t) evaluation, so the two are
comparable.
"""
import json
import os

import numpy as np
import torch

ORDINARY_PROMPTS = [
    # general prose
    "Write a short paragraph explaining why bread dough rises.",
    "Explain how a bicycle gear system changes the effort needed to pedal.",
    "Describe what happens to a metal spoon left in a hot bowl of soup.",
    # factual / recall
    "What is the difference between weather and climate?",
    "Which planet has the shortest day in our solar system, and why?",
    "Who wrote the play 'A Doll's House'?",
    # procedural / how-to
    "Give three tips for someone learning to swim as an adult.",
    "How do you make a simple tomato sauce from scratch?",
    "What should someone check before a long drive in winter?",
    # maths with a worked answer
    "A shop sells pens for 3 dollars each. If Maya buys 7 pens and pays with a 50 "
    "dollar note, how much change does she get? Show your working.",
    "A tank holds 120 litres and is filled at 15 litres per minute. How many minutes "
    "does it take to fill? Show your working.",
    # code
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write a Python function that checks whether a string is a valid email address.",
    # translation / language mixing
    "Translate to French: 'The train leaves at nine in the morning.'",
    "Translate to German: 'We have been waiting for an hour.'",
    # instruction following with constraints
    "List exactly three primary colours, then explain each one in one sentence.",
    "Rewrite the following sentence in the passive voice: 'The chef broke the plate.'",
    # long-form generation (the regime the lever could plausibly damage)
    "Explain in detail, with several paragraphs, how a bicycle gear system works and "
    "why a cyclist might choose different gears on different terrain.",
    "Give a detailed walkthrough of how to prepare a week of meals for one person.",
]

LONG_PROMPTS = [
    "Explain in detail, with several paragraphs, how a bicycle gear system works and "
    "why a cyclist might choose different gears on different terrain.",
    "Give a detailed walkthrough of how to prepare a week of meals for one person.",
]

NEUTRAL_PASSAGE = (
    "Bread dough rises because yeast consumes sugars in the flour and releases carbon "
    "dioxide, which is trapped by the gluten network that has been stretched and "
    "folded. The gluten gives the dough its elasticity, the gas inflates the small "
    "cells of the crumb, and the oven sets the protein structure so the loaf keeps "
    "the shape it had when it left the pan. Resting the dough before baking lets the "
    "network relax, which is why a gentler handling produces a more even crumb. A "
    "sourdough culture works more slowly than commercial yeast, which is why the "
    "flavour is sharper and the dough needs a longer rest before shaping."
)

# Healthy controls: problems the BASE model already closes. The intervened model must
# still produce an answer containing the expected number, and must not get shorter.
HEALTHY_CONTROLS = [
    {"id": "healthy_1", "question": "A shop sells pens for 3 dollars each. If Maya "
     "buys 7 pens and pays with a 50 dollar note, how much change does she get?",
     "expect": "29"},
    {"id": "healthy_2", "question": "A tank holds 120 litres and is filled at 15 litres "
     "per minute. How many minutes does it take to fill?",
     "expect": "8"},
]


def build_reference(model, tok, device, max_new_tokens=96):
    """Base-model references for the panel: next-token distributions, lengths, answers."""
    references = {}
    for prompt in ORDINARY_PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        token = int(probs.argmax())
        text = tok.decode([token])
        references[prompt] = dict(probs=probs, token=token, text=text)
    neutral_ids = tok(NEUTRAL_PASSAGE, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        references["__passage__"] = float(
            model(neutral_ids, labels=neutral_ids).loss)
    return references


def measure_next_token(model, tok, device, references):
    """KL and argmax agreement between the current weights and the base references."""
    kls, agrees, token_texts = [], [], []
    for prompt in ORDINARY_PROMPTS:
        reference = references[prompt]
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
        kls.append(float((reference["probs"] * (reference["probs"].clamp_min(1e-12).log()
                                                - probs.clamp_min(1e-12).log())).sum()))
        agrees.append(float(int(probs.argmax()) == reference["token"]))
        token_texts.append(tok.decode([int(probs.argmax())]))
    return dict(ordinary_kl=float(np.mean(kls)), ordinary_agree=float(np.mean(agrees))), token_texts


def measure_passage(model, tok, device, base_nll):
    """Perplexity on the fixed passage, and its change against the base."""
    neutral_ids = tok(NEUTRAL_PASSAGE, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        nll = float(model(neutral_ids, labels=neutral_ids).loss)
    return dict(ppl_base=float(np.exp(base_nll)), ppl_at_alpha=float(np.exp(nll)),
                ppl_delta_pct=float((np.exp(nll) / np.exp(base_nll) - 1) * 100))


def measure_lengths(generate):
    """Mean generated length across the ordinary prompts, and the texts themselves.

    `generate` is a callable(prompt) -> (text, info) so the caller can reuse its own
    rollout path (with the presence penalty) rather than a second, different decoder.
    """
    lengths, texts = [], []
    for prompt in ORDINARY_PROMPTS:
        text, info = generate(prompt)
        lengths.append(info.get("chars", len(text)))
        texts.append(text)
    return float(np.mean(lengths)), texts


def measure(model, tok, device, references, generate):
    """The whole panel at the model's CURRENT weights (i.e. at the current alpha)."""
    next_token, token_texts = measure_next_token(model, tok, device, references)
    panel = dict(next_token)
    panel.update(measure_passage(model, tok, device, references["__passage__"]))
    mean_chars, texts = measure_lengths(generate)
    panel["mean_chars"] = mean_chars
    return panel, dict(texts=texts, token_texts=token_texts)


def check_healthy_answers(generate, controls=HEALTHY_CONTROLS):
    """For healthy controls: does the intervened model still produce the answer?

    A lever that shortens reasoning could destroy the final answer even while every
    fluency metric improves. This is the check that separates 'removed a loop' from
    'truncated the reasoning'.
    """
    results = []
    for control in controls:
        text, info = generate(control["question"])
        results.append(dict(id=control["id"], expect=control["expect"],
                            found=control["expect"] in text, chars=info.get("chars", len(text)),
                            closed=info.get("closed")))
    return results


def summarise(panel, healthy=None):
    """One line a human can read."""
    bits = [f"kl {panel['ordinary_kl']:.4f}",
            f"agree {panel['ordinary_agree']:.3f}",
            f"ppl {panel['ppl_delta_pct']:+.2f}%"]
    if "mean_chars" in panel:
        bits.append(f"chars {panel['mean_chars']:.0f}")
    if healthy is not None:
        found = sum(r["found"] for r in healthy)
        bits.append(f"healthy answers {found}/{len(healthy)}")
    return " | ".join(bits)


if __name__ == "__main__":
    print(json.dumps(dict(ordinary_prompts=len(ORDINARY_PROMPTS),
                          healthy_controls=len(HEALTHY_CONTROLS)), indent=1))
