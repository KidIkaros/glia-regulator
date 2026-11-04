"""Local ground-truth check (no GPU, no quota): is proj_L*.npy really W[:,j].d_hat?

The stage-2 edit algebra is  W[:,j] += d_unit * (-6 * (W[:,j].d_unit))  , so the saved
proj array must equal W[:,j] . d_unit with d_unit = loopdir / ||loopdir||. If instead it
was computed against the RAW (un-normalised) direction, then building the lever's delta
as -6 * proj * d_unit applies the wrong magnitude and the lever does not reproduce the
edit - which is exactly what the 'ours' arm showed (still looping, and looping LONGER).

Runs against the local base checkpoint; prints both comparisons.
"""
import glob
import os

import numpy as np
import torch
from safetensors.torch import load_file

WSP = "/home/ikaaros/Coding/kaggle/wsp"
pattern = os.path.join(WSP, "models", "Qwen3.5-0.8B", "*.safetensors")
files = sorted(glob.glob(pattern))
assert files, f"no checkpoint at {pattern}"
sd = load_file(files[0])
print("checkpoint:", files[0], "| tensors:", len(sd))

key_for = {}
for k in sd:
    if k.endswith("mlp.down_proj.weight") and ".layers." in k:
        key_for[k.split(".layers.")[1].split(".")[0].split(".")[-1]] = k
# fall back: build from layer index
def down_weight(L):
    for k in sd:
        if k.endswith(f"layers.{L}.mlp.down_proj.weight"):
            return sd[k].float().numpy()
    raise KeyError(L)


for L in (15, 19):
    W = down_weight(L)
    neurons = np.load(os.path.join(WSP, f"neurons_L{L}.npy"))
    d_raw = np.load(os.path.join(WSP, f"loopdir_L{L}.npy")).astype(np.float32)
    proj = np.load(os.path.join(WSP, f"proj_L{L}.npy")).astype(np.float32)
    d_unit = d_raw / np.linalg.norm(d_raw)
    c_unit = W[:, neurons].T @ d_unit
    c_raw = W[:, neurons].T @ d_raw
    print(f"L{L}: |d_raw| = {np.linalg.norm(d_raw):.5f}  neurons {len(neurons)}")
    print(f"   max|c_unit - proj| = {np.abs(c_unit - proj).max():.6f}"
          f"   corr = {np.corrcoef(c_unit, proj)[0, 1]:.6f}")
    print(f"   max|c_raw  - proj| = {np.abs(c_raw - proj).max():.6f}"
          f"   corr = {np.corrcoef(c_raw, proj)[0, 1]:.6f}")
    print(f"   proj[:3]  = {np.round(proj[:3], 5)}")
    print(f"   c_unit[:3]= {np.round(c_unit[:3], 5)}")
    print(f"   c_raw[:3] = {np.round(c_raw[:3], 5)}")
    # what magnitude does each convention imply for the edit?
    print(f"   implied |delta| mean: unit-conv {np.abs(6*c_unit).mean():.4f} "
          f"raw-conv {np.abs(6*c_raw).mean():.4f}  stored-proj {np.abs(6*proj).mean():.4f}")
