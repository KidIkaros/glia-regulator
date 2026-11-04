"""Local (CPU, zero-quota) diff census and delta extraction.

Does on the local machine what the Kaggle census kernel did on GPU, and produces the
payload the next GPU kernel needs, so no future kernel has to download the 1.6 GB
edited checkpoint:

  1. census: which tensors differ between the hub's CURRENT base and the authors'
     edited checkpoint, and by how much;
  2. extract the delta D = W_edited - W_base on the changed columns of layers 15/19,
     with the exact column indices;
  3. verify that base + D == edited EXACTLY (bitwise in the stored dtype);
  4. report the overlap between the authors' edited columns and OUR independently
     diagnosed neuron set (which was derived on an OLDER revision of the model, so this
     also measures how far the two revisions have drifted);
  5. write delta_L15.npy / delta_L19.npy / delta_idx.json / delta_manifest.json.

Usage: python3 scripts/extract_delta.py
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "base-hub-current")
EDITED = os.path.join(ROOT, "models", "edited-authors")
OUT = os.path.join(ROOT, "ground_truth")
WSP = "/home/ikaaros/Coding/kaggle/wsp"


def load_all(d):
    tensors = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        tensors.update(load_file(f))
    return tensors


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("loading base   :", BASE, flush=True)
    sb = load_all(BASE)
    print("loading edited :", EDITED, flush=True)
    se = load_all(EDITED)
    print(f"tensors: base {len(sb)}  edited {len(se)}")

    only_base = sorted(set(sb) - set(se))
    only_edited = sorted(set(se) - set(sb))
    if only_base or only_edited:
        print(f"KEY MISMATCH: only-base {len(only_base)} (e.g. {only_base[:3]}) "
              f"only-edited {len(only_edited)} (e.g. {only_edited[:3]})")

    rows = []
    for name in sorted(set(sb) & set(se)):
        a, b = sb[name], se[name]
        if a.shape != b.shape:
            rows.append(dict(name=name, shape_changed=True))
            continue
        a32, b32 = a.float(), b.float()
        d = (b32 - a32).abs()
        n_changed = int((d > 0).sum())
        if n_changed:
            rows.append(dict(name=name, n_changed=n_changed, numel=int(a.numel()),
                             max_abs_diff=float(d.max()),
                             rms_diff=float(d.pow(2).mean().sqrt())))
    rows.sort(key=lambda r: -(r.get("max_abs_diff") or 0))
    print(f"\nDIFFERING TENSORS: {len(rows)} / {len(sb)}")
    for r in rows[:10]:
        print(f"  {r['name']}: changed {r['n_changed']}/{r['numel']} "
              f"max {r.get('max_abs_diff'):.6f} rms {r.get('rms_diff'):.6f}")

    os.makedirs(OUT, exist_ok=True)
    payload = {"layers": {}, "manifest": {}}
    for L in (15, 19):
        key = f"model.layers.{L}.mlp.down_proj.weight"
        if key not in sb or key not in se:
            cand = [k for k in sb if k.endswith(f"layers.{L}.mlp.down_proj.weight")]
            if not cand:
                print(f"L{L}: no down_proj found, skipping")
                continue
            key = cand[0]
        Wb = sb[key].float().numpy()
        We = se[key].float().numpy()
        D = We - Wb
        norms = np.linalg.norm(D, axis=0)
        changed = np.where(norms > 0.05)[0]
        idx = np.sort(changed)
        delta = D[:, idx]
        # exactness check: base + delta must equal edited on those columns
        recon = Wb[:, idx] + delta
        err = float(np.abs(recon - We[:, idx]).max())
        # our independently diagnosed neurons (computed on an OLDER revision)
        ours = np.load(os.path.join(WSP, f"neurons_L{L}.npy")).astype(int)
        overlap = len(set(idx.tolist()) & set(ours.tolist()))
        d_hat = delta[:, 0] / (np.linalg.norm(delta[:, 0]) + 1e-9)
        proj_current = Wb[:, idx].T @ d_hat
        print(f"\nL{L}: {len(idx)} changed columns | recon max err {err:.2e} | "
              f"our-neuron overlap {overlap}/{len(ours)}")
        print(f"     |delta| mean {np.linalg.norm(delta, axis=0).mean():.4f} | "
              f"proj(current weights) mean {np.abs(proj_current).mean():.5f}")
        old_proj = np.load(os.path.join(WSP, f"proj_L{L}.npy")).astype(np.float32)
        print(f"     stored proj (old revision) mean {np.abs(old_proj).mean():.5f} | "
              f"ratio old/current {np.abs(old_proj).mean()/max(np.abs(proj_current).mean(),1e-9):.2f}x")
        np.save(os.path.join(OUT, f"delta_L{L}.npy"), delta.astype(np.float32))
        payload["layers"][str(L)] = dict(idx=[int(i) for i in idx],
                                         n_cols=int(len(idx)),
                                         delta_file=f"delta_L{L}.npy",
                                         direction=[float(x) for x in d_hat],
                                         recon_max_err=err,
                                         our_neuron_overlap=int(overlap))
        payload["manifest"][key] = dict(
            edited_sha256=sha256(os.path.join(EDITED, os.path.basename(key)))
            if os.path.exists(os.path.join(EDITED, os.path.basename(key))) else None,
            n_changed=int(sum(1 for c in range(Wb.shape[1]) if norms[c] > 0)),
            max_abs_diff=float(norms.max()))

    json.dump(payload, open(os.path.join(OUT, "delta_idx.json"), "w"), indent=1)
    json.dump(dict(census=rows, base_dir=os.path.basename(BASE),
                   edited_dir=os.path.basename(EDITED)),
              open(os.path.join(OUT, "delta_manifest.json"), "w"), indent=1)
    print("\nwrote", OUT + "/delta_L15.npy, delta_L19.npy, delta_idx.json, delta_manifest.json")


if __name__ == "__main__":
    main()
