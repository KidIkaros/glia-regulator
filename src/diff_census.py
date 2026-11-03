"""Diff census: exactly which tensors differ between Qwen/Qwen3.5-0.8B and the
authors' edited checkpoint, and by how much?

The logit-equivalence check failed (max|delta| ~ 10), which means the authors' edit is
not only the 62+63 down_proj columns we extracted. This census lists every tensor
that differs, so the lever can be built from the complete set of changes.

Output: tensors_differing.json = [{name, n_changed, max_abs_diff, rms_diff}, ...]
(sorted by max_abs_diff) plus a summary of how many tensors differ at all.
"""
import json
import os
import sys
import time

import torch

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
EDITED_ID = os.environ.get("EDITED_ID", "RikiyaT/Qwen3.5-0.8B-weight-space-programming")
t0 = time.time()


def log(*a):
    print(f"[census {round(time.time()-t0)}s]", *a, flush=True)


from transformers import AutoModelForCausalLM

log("loading base")
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
log("loading edited")
edited = AutoModelForCausalLM.from_pretrained(EDITED_ID, dtype=torch.float16)

sb, se = base.state_dict(), edited.state_dict()
log("tensors:", len(sb), len(se))
missing = set(sb) ^ set(se)
if missing:
    log("KEY MISMATCH:", sorted(missing)[:10])

rows = []
for name in sorted(set(sb) & set(se)):
    a, b = sb[name].float(), se[name].float()
    if a.shape != b.shape:
        rows.append(dict(name=name, shape_changed=True, n_changed=int(a.numel()),
                         max_abs_diff=float("nan"), rms_diff=float("nan")))
        continue
    d = (b - a).abs()
    n_changed = int((d > 0).sum())
    if n_changed:
        rows.append(dict(name=name, n_changed=n_changed, numel=int(a.numel()),
                         max_abs_diff=float(d.max()),
                         rms_diff=float(d.pow(2).mean().sqrt())))

rows.sort(key=lambda r: (-(r.get("max_abs_diff") or 0), -r.get("n_changed", 0)))
log(f"tensors differing: {len(rows)} / {len(sb)}")
for r in rows[:25]:
    log(f"  {r['name']}: changed {r.get('n_changed')} / {r.get('numel')} "
        f"max {r.get('max_abs_diff'):.5f} rms {r.get('rms_diff'):.6f}")

out = dict(status="complete", n_tensors=len(sb), n_differing=len(rows), rows=rows)
json.dump(out, open(os.path.join(OUT, "tensors_differing.json"), "w"), indent=1)
log("SUMMARY differing:", len(rows))
