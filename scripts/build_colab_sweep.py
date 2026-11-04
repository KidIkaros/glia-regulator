"""Build the Colab script for the dose-response sweep (payload embedded, sent over stdin).

    python3 scripts/build_colab_sweep.py
    cat colab_sweep_run.py | colab exec -s <session> --timeout 1200 \
        --env N_PROBLEMS=18 --env ALPHAS="1.0 1.25 1.5 1.75 2.0" --env MAX_ARMS_PER_RUN=1
"""
import base64
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "src/harness.py": os.path.join(ROOT, "src", "harness.py"),
    "src/lever.py": os.path.join(ROOT, "src", "lever.py"),
    "src/dose_sweep.py": os.path.join(ROOT, "src", "dose_sweep.py"),
    "s1_pairs.json": os.path.join(ROOT, "s1_pairs.json"),
}
for name, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing payload file {name} -> {path}")
payload = {n: base64.b64encode(open(p, "rb").read()).decode() for n, p in FILES.items()}

SCRIPT = '''"""Dose-response sweep with a capability panel (Colab GPU, self-contained)."""
import base64, os, subprocess, sys

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

os.makedirs("/content/glia_out", exist_ok=True)
os.environ.setdefault("RESULTS_PATH", "/content/glia_out/sweep.jsonl")
subprocess.run([sys.executable, os.path.join(root, "src", "dose_sweep.py")], check=True)
'''

out = os.path.join(ROOT, "colab_sweep_run.py")
open(out, "w").write(SCRIPT.replace("__PAYLOAD__", repr(payload)))
print("wrote", out, f"({os.path.getsize(out)/1024:.1f} KB)")
