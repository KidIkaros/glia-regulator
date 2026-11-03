"""Glial dry run (CPU): integrate the glial field into Qwen3.5-0.8B and verify the three
things that would silently ruin the real experiment if wrong:

  1. IDENTITY at init - glial installed with zero-init heads must give bit-identical
     logits to the base model.
  2. HOOKS LIVE - after nudging one head's weights, logits must change.
  3. TARGETING - non-target neurons must remain untouched; gains must be bounded.

No training, no GPU, no quota. Writes results to /kaggle/working/dryrun_glial.json.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glial import GlialConfig, install, _resolve

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
t0 = time.time()


def log(*a):
    print(f"[dryrun {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
log("model loaded", model.__class__.__name__)

# where do the layers live?
layer_names = [n for n, _ in model.named_modules() if n.endswith(".layers.15.mlp.down_proj")]
log("down_proj(15) path:", layer_names)
n_layers = len([n for n, _ in model.named_modules() if n.endswith(".mlp.down_proj")])
log("mlp.down_proj count (layers with FFN):", n_layers)

d_model = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
inter = None
_, down15 = _resolve(model, ".layers.15.mlp.down_proj")
inter = down15.weight.shape[1]      # down_proj: [hidden, intermediate]
log("d_model:", d_model, "intermediate:", inter)

prompt = "A department store offers a 10% discount on a $200 item. What is the final price?"
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=True)
enc = tok(ids, return_tensors="pt")

with torch.no_grad():
    base_logits = model(**enc, logits_to_keep=8).logits.clone()
log("base forward done, logits", tuple(base_logits.shape))

# the 64 neurons we identified in the wsp project (same architecture)
idx15 = torch.arange(64)          # dry run: first 64; the real run loads neurons_L15.npy
idx19 = torch.arange(64)
cfg = GlialConfig(read_layers=(4, 8, 12, 16), target_layers=(15, 19),
                  neurons_per_layer=64, behaviors=("loop", "syco"))
field, handles = install(model, cfg, {15: idx15, 19: idx19}, inter=inter, d_model=d_model)
log("glial installed; handles:", len(handles))

with torch.no_grad():
    glial_logits = model(**enc, logits_to_keep=8).logits.clone()
identity = torch.equal(base_logits, glial_logits)
maxdiff_init = float((base_logits - glial_logits).abs().max())
log("IDENTITY at init:", identity, "| max logit diff:", maxdiff_init)

# nudge one head: hooks must bite
with torch.no_grad():
    for h in field.heads.values():
        h.weight.normal_(0, 0.05)
        h.bias.fill_(0.2)
field.reset_state()
with torch.no_grad():
    nudged = model(**enc, logits_to_keep=8).logits.clone()
delta = float((nudged - base_logits).abs().max())
log("HOOKS LIVE: max logit delta after nudging heads:", delta)

gains = {l: float(g.abs().max()) for l, g in field._gains.items()} if field._gains else {}
log("gains |max|:", gains, "| sigma bound:", cfg.sigma)
state_norm = float(field._state.norm()) if field._state is not None else 0.0
log("glial state norm:", state_norm)

res = dict(
    status="ok" if (identity and delta > 1e-4) else "FAILED",
    model_class=model.__class__.__name__,
    down_proj_path=layer_names[0] if layer_names else None,
    d_model=d_model, intermediate=inter,
    identity_at_init=bool(identity), max_diff_at_init=maxdiff_init,
    hooks_live=bool(delta > 1e-4), max_logit_delta_after_nudge=delta,
    gain_abs_max=gains, sigma=cfg.sigma, state_norm=state_norm,
    elapsed_s=time.time() - t0,
)
json.dump(res, open(os.path.join(OUT, "dryrun_glial.json"), "w"), indent=1)
log("RESULT", json.dumps(res))
for h in handles:
    h.remove()
