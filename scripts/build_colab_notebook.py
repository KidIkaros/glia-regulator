"""Emit a Colab-ready notebook for the fp32 lever-fidelity experiment.

The notebook carries the driver and its two helper modules as base64 payloads (the
same pattern the Kaggle builders use, so the code that runs is the code in the repo),
mounts Drive for resumable results, and runs the arms in priority order.

    python3 scripts/build_colab_notebook.py            -> glia_colab_fidelity.ipynb
    KAGGLE_KERNEL_ID-style override: COLAB_NOTEBOOK=path
"""
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_FILES = {
    "src/harness.py": os.path.join(ROOT, "src", "harness.py"),
    "src/lever.py": os.path.join(ROOT, "src", "lever.py"),
    "src/colab_dose.py": os.path.join(ROOT, "src", "colab_dose.py"),
    "s1_pairs.json": os.path.join(ROOT, "s1_pairs.json"),
}
for name, path in PAYLOAD_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing payload file {name} -> {path}")
payload = {name: base64.b64encode(open(path, "rb").read()).decode()
           for name, path in PAYLOAD_FILES.items()}

SETUP = '''"""fp32 lever-fidelity run (Colab).

Question: the runtime lever reproduces the authors' weight edit at ONE step and for
875 of 1024 tokens, but on a T4 in fp16 it did not close a loop the weight edit closes.
Is that fp16 arithmetic, or is the runtime lever simply not the fix?

This notebook works in fp32 and runs, in priority order:
  1. fix_reference  - the edited checkpoint's own 16k-token rollout (the target)
  2. lever_alpha1   - base + lever(alpha=1)
  3. lever_alpha1_5 - base + lever(alpha=1.5): does over-correction break the loop?

Each arm is appended to Drive as soon as it finishes, and a recorded arm is skipped on
relaunch, so a dropped Colab session costs minutes rather than hours.
"""
import base64, json, os, subprocess, sys, time

PAYLOAD = __PAYLOAD__
root = "/content/glia_work"
for name, b64 in PAYLOAD.items():
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("payload written to", root, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.22.1"], check=True)
import torch
print("cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
      flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no GPU in this session - Colab free tier may not have one right now")

# The Drive mount can raise a permission prompt that nobody clicks, which would stall
# the run before a single token is generated. So: try to mount, and if it is not
# available, write the results into the runtime's own filesystem and print them.
RESULTS_DIR = "/content/glia_out"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("/content/drive/glia_colab", exist_ok=True)
    RESULTS_DIR = "/content/drive/glia_colab"
    print("drive mounted; results ->", RESULTS_DIR, flush=True)
except Exception as error:
    print("drive not mounted (", error, ") - results stay in the runtime filesystem",
          flush=True)
os.environ["RESULTS_DIR"] = RESULTS_DIR
'''

RUN = '''os.environ["RESULTS_PATH"] = os.path.join(
    os.environ.get("RESULTS_DIR", "/content/glia_out"), "results.jsonl")
# also drop a copy into the Drive folder the desktop is linked to, so the results can
# be read from the local machine with: gio copy "google-drive://.../glia_results.jsonl" .
os.environ.setdefault("MIRROR_FOLDER_ID", "0AM4A_9P6W9cBUk9PVA")
os.environ.setdefault("ROLLOUT", "16384")
os.environ.setdefault("ALPHAS", "1.0 1.5")
subprocess.run([sys.executable, os.path.join(root, "src", "colab_dose.py")], check=True)

import json, os
path = os.environ["RESULTS_PATH"]
rows = [json.loads(line) for line in open(path) if line.strip()]
summary = {r["arm"]: dict(n=r.get("info", {}).get("n"),
                           closed=r.get("info", {}).get("closed"),
                           chars=r.get("info", {}).get("chars"),
                           matched_prefix=r.get("matched_prefix")) for r in rows}
print("\\n=== RESULTS ON DRIVE ===")
print(json.dumps(summary, indent=1))
print("results file:", path)

import shutil
mirror = os.path.join("/content/drive", os.environ["MIRROR_FOLDER_ID"], "glia_results.jsonl")
try:
    shutil.copy(path, mirror)
    print("mirrored to", mirror)
except Exception as error:
    print("mirror failed:", error)
'''

SETUP = SETUP.replace("__PAYLOAD__", repr(payload))

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Glial lever: fp32 fidelity run\n",
                                                          "\n",
                                                          "Free-T4 experiment: does the runtime lever reproduce the ",
                                                          "weight-space fix's behaviour over a full 16k-token generation?\n",
                                                          "\n",
                                                          "Resumable: each arm is written to Drive as it finishes; re-running ",
                                                          "skips completed arms.\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
     "source": SETUP.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
     "source": RUN.splitlines(keepends=True)},
]
notebook = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                         "kernelspec": {"name": "python3"},
                                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 0}

out = os.environ.get("COLAB_NOTEBOOK", os.path.join(ROOT, "glia_colab_fidelity.ipynb"))
json.dump(notebook, open(out, "w"), indent=1)
print("wrote", out, f"({os.path.getsize(out)/1024:.1f} KB)")
