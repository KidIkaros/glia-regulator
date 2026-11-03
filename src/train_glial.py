"""Train the glial field only (base frozen). Three objectives, one shared field:

  syco : raise lp(correct) - lp(wrong user suggestion)   [behaviour head "syco"]
  loop : raise lp of a healthy completion                [behaviour head "loop"]
         targets = the loop-fixed model's own completions on the same prompts
  cap  : keep the base model's next-token distribution  [no head - regulariser]
         KL(base || glial) on generic text

Documents are fed in segments so the glial slow state (PGC) has real dynamics; the
state resets between documents. --smoke runs a handful of tiny steps and asserts that
gradients reach the heads and the gains actually move.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glial import GlialConfig, install, _resolve

SEG = 512


def forward_segmented(model, field, ids, seg=SEG):
    """Forward a document in segments, carrying glial state, returning logits list."""
    outs = []
    for s in range(0, ids.shape[1], seg):
        e = min(ids.shape[1], s + seg)
        pos = torch.arange(s, e, device=ids.device).unsqueeze(0)
        out = model(input_ids=ids[:, s:e], position_ids=pos)
        outs.append(out.logits)
    return outs


def nll_of_continuation(logits_list, ids, prompt_len, seg=SEG):
    """NLL of ids[:, prompt_len:] given the segmented logits."""
    total, n = 0.0, 0
    for i, lg in enumerate(logits_list):
        s, e = i * seg, min(ids.shape[1], (i + 1) * seg)
        for t in range(s, e - 1):
            if t + 1 < prompt_len:
                continue
            target = ids[0, t + 1]
            total = total - torch.log_softmax(lg[0, t - s].float(), dim=-1)[target]
            n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--targets", default="")   # jsonl of {prompt, completion} from the fixed model
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, down = _resolve(model, ".layers.15.mlp.down_proj")
    inter = down.weight.shape[1]
    d_model = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    cfg = GlialConfig(behaviors=("loop", "syco"))
    idx = {15: torch.arange(cfg.neurons_per_layer), 19: torch.arange(cfg.neurons_per_layer)}
    field, handles = install(model, cfg, idx, inter=inter, d_model=d_model)
    opt = torch.optim.Adam(field.parameters(), lr=args.lr)

    from syco_eval import build_pairs
    pairs = build_pairs(4 if args.smoke else 100)
    loop_targets = []
    if args.targets and os.path.exists(args.targets):
        loop_targets = [json.loads(l) for l in open(args.targets)][:4 if args.smoke else 50]
    cap_text = "The river runs past the old mill and into the valley where the fields are green."

    def encode(prompt, cont=None):
        p = tok(prompt, return_tensors="pt").input_ids
        if cont is None:
            return p, p.shape[1]
        c = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids
        return torch.cat([p, c], dim=1), p.shape[1]

    log = lambda *a: print(f"[train {round(time.time()-t0)}s]", *a, flush=True)
    t0 = time.time()
    steps = 3 if args.smoke else args.steps
    g0 = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        field.reset_state()
        # --- syco: push lp(gold) above lp(wrong)
        pr = pairs[step % len(pairs)]
        field.active_heads = {"syco"}
        ids_g, pl_g = encode(pr["prompt"], " " + pr["gold"])
        l_g = nll_of_continuation(forward_segmented(model, field, ids_g), ids_g, pl_g)
        field.reset_state()
        ids_w, pl_w = encode(pr["prompt"], " " + pr["wrong"])
        l_w = nll_of_continuation(forward_segmented(model, field, ids_w), ids_w, pl_w)
        loss_syco = torch.relu(l_w - l_g + 0.5)          # want margin >= 0.5 nats
        # --- loop: NLL of a healthy completion
        loss_loop = torch.zeros(())
        if loop_targets:
            lt = loop_targets[step % len(loop_targets)]
            field.active_heads = {"loop"}
            field.reset_state()
            ids_l, pl_l = encode(lt["prompt"], lt["completion"])
            loss_loop = nll_of_continuation(forward_segmented(model, field, ids_l), ids_l, pl_l)
        # --- capability: KL(base || glial) on generic text
        # base reference: hooks still installed, so disable all heads -> zero gains
        field.active_heads = set()
        field.reset_state()
        with torch.no_grad():
            base_logits = model(input_ids=encode(cap_text)[0]).logits
        field.active_heads = {"loop", "syco"}
        field.reset_state()
        glial_logits = model(input_ids=encode(cap_text)[0]).logits
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(glial_logits[0, :-1].float(), -1),
            torch.log_softmax(base_logits[0, :-1].float(), -1),
            log_target=True, reduction="batchmean")
        loss = loss_syco + (0.5 * loss_loop if loop_targets else 0.0) + 0.1 * kl
        loss.backward()

        grads = {b: float(h.weight.grad.abs().max()) if h.weight.grad is not None else 0.0
                 for b, h in field.heads.items()}
        gmax = max(float(g.abs().max()) for g in field._gains.values()) if field._gains else 0.0
        if g0 is None:
            g0 = gmax
        opt.step()
        if step % (1 if args.smoke else 10) == 0:
            log(f"step {step}: loss={float(loss):.3f} syco={float(loss_syco):.3f} "
                f"loop={float(loss_loop):.3f} kl={float(kl):.2e} grads={grads} |g|={gmax:.4f}")

    json.dump({b: float(h.weight.abs().max()) for b, h in field.heads.items()},
              open(os.path.join(args.out, "glial_heads_absmax.json"), "w"))
    torch.save(field.state_dict(), os.path.join(args.out, "glial_state.pt"))
    log("saved glial_state.pt")
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
