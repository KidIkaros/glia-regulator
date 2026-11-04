"""Local, zero-quota validation of the dose-response payload.

The Sunday kernel validates its payload against the live weights before generating
anything: every changed column must be parallel to ONE direction, and the implied edit
strength must be ~6. This test runs exactly those checks against the LOCAL copies of
both checkpoints, so a payload that would fail in the kernel fails here first.

    python3 scripts/validate_delta.py
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "base-hub-current")
EDITED = os.path.join(ROOT, "models", "edited-authors")
GT = os.path.join(ROOT, "ground_truth")


def load_all(d):
    import glob
    tensors = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        tensors.update(load_file(f))
    return tensors


def main():
    if not os.path.isdir(BASE) or not os.path.isdir(EDITED):
        print("missing local checkpoints; run the snapshot_download step first")
        return 2
    sb = load_all(BASE)
    se = load_all(EDITED)
    meta = json.load(open(os.path.join(GT, "delta_idx.json")))
    ok = True
    for L in (15, 19):
        info = meta["layers"][str(L)]
        delta = np.load(os.path.join(GT, info["delta_file"])).astype(np.float32)
        idx = [int(i) for i in info["idx"]]
        key = f"model.layers.{L}.mlp.down_proj.weight"
        if key not in sb:
            cand = [k for k in sb if k.endswith(f"layers.{L}.mlp.down_proj.weight")]
            key = cand[0] if cand else None
        assert key, f"L{L}: down_proj not found"
        Wb = sb[key].float().numpy()
        We = se[key].float().numpy()

        # payload matches the live edited checkpoint exactly
        recon = Wb[:, idx] + delta
        err = float(np.abs(recon - We[:, idx]).max())
        # structural signature 1: columns parallel to one direction
        cols = torch.as_tensor(delta)
        d_hat = torch.as_tensor(info["direction"], dtype=torch.float32)
        unit = cols / (cols.norm(dim=0, keepdim=True) + 1e-9)
        cos = (unit.T @ d_hat).abs()
        # structural signature 2: implied edit strength ~6
        proj = torch.as_tensor(Wb[:, idx]).T @ d_hat
        dp = cols.T @ d_hat
        implied_median = float((dp @ proj) / (proj @ proj).clamp_min(1e-12))

        checks = [
            ("payload reconstructs the edited weights", err < 1e-4, f"max err {err:.2e}"),
            ("columns parallel to one direction", float(cos.min()) > 0.999,
             f"min cos {float(cos.min()):.6f}"),
            ("implied edit strength ~6", 4.0 < abs(implied_median) < 8.0,
             f"LSQ estimate {implied_median:.3f} (negative = the edit removes the direction)"),
            ("column count matches the census", len(idx) == info["n_cols"],
             f"{len(idx)} vs {info['n_cols']}"),
        ]
        for name, passed, detail in checks:
            print(f"L{L}: {'PASS' if passed else 'FAIL'}  {name} ({detail})")
            ok = ok and passed
        print(f"L{L}: our-neuron overlap with the authors' columns: "
              f"{info.get('our_neuron_overlap')}/{info.get('n_cols')}")
    print("\nRESULT:", "payload is valid" if ok else "PAYLOAD INVALID - do not launch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
