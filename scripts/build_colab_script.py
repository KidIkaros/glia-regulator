"""Emit a single self-contained script for `colab exec` (stdin), with the payload embedded.

File staging through `colab upload` does not land in the VM filesystem the driver reads
(upload reported success, but /content held only .config and sample_data), so the
robust route is to send everything in one stdin blob: the payload is base64-embedded,
written out by the script itself, and then executed.

    python3 scripts/build_colab_script.py            -> colab_fidelity_run.py
    cat colab_fidelity_run.py | colab exec -s <session> --timeout 5400
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

SCRIPT = '''"""fp32 lever-fidelity run on a Colab GPU (self-contained; payload embedded).

Question: the runtime lever reproduces the authors' weight edit at ONE step and for 875
of 1024 tokens, but on a T4 in fp16 it did not close a loop the weight edit closes after
4,555 characters. Is that fp16 arithmetic, or is the runtime lever simply not the fix?

Arms, in priority order (a dropped session must not lose the important ones):
  1. fix_reference   - the edited checkpoint's own 16k-token rollout (the target)
  2. lever_alpha1    - base + lever(alpha=1)
  3. lever_alpha1_5 - base + lever(alpha=1.5): does over-correction break the loop?

Each arm is appended to the results file as soon as it finishes, and an arm already
present is SKIPPED, so re-running after a disconnect resumes instead of restarting.
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
    raise SystemExit("no GPU in this session")

RESULTS_DIR = "/content/glia_out"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["RESULTS_DIR"] = RESULTS_DIR
os.environ["RESULTS_PATH"] = os.path.join(RESULTS_DIR, "results.jsonl")
os.environ.setdefault("ROLLOUT", "16384")
os.environ.setdefault("ALPHAS", "1.0 1.5")
print("results ->", os.environ["RESULTS_PATH"], flush=True)

subprocess.run([sys.executable, os.path.join(root, "src", "colab_dose.py")], check=True)

rows = [json.loads(line) for line in open(os.environ["RESULTS_PATH"]) if line.strip()]
summary = {r["arm"]: dict(n=r.get("info", {}).get("n"),
                           closed=r.get("info", {}).get("closed"),
                           chars=r.get("info", {}).get("chars"),
                           matched_prefix=r.get("matched_prefix")) for r in rows}
print("\\n=== RESULTS ===")
print(json.dumps(summary, indent=1))
print("results file:", os.environ["RESULTS_PATH"])
'''

out = os.environ.get("COLAB_SCRIPT", os.path.join(ROOT, "colab_fidelity_run.py"))
open(out, "w").write(SCRIPT.replace("__PAYLOAD__", repr(payload)))
print("wrote", out, f"({os.path.getsize(out)/1024:.1f} KB)")
