"""Build the Colab script for Stage A2 (the identity foothold).

    python3 scripts/build_colab_identity.py
    # then, per phase (each invocation is one phase; every phase is resumable):
    cat colab_identity_run.py | colab exec -s <session> --timeout 1300 \
        --env PHASE=fit_r --env DOCS=200
"""
import base64
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "src/harness.py": os.path.join(ROOT, "src", "harness.py"),
    "src/lever.py": os.path.join(ROOT, "src", "lever.py"),
    "src/identity.py": os.path.join(ROOT, "src", "identity.py"),
    "s1_pairs.json": os.path.join(ROOT, "s1_pairs.json"),
}
for name, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing payload file {name} -> {path}")
payload = {n: base64.b64encode(open(p, "rb").read()).decode() for n, p in FILES.items()}

SCRIPT = '''"""Stage A2: identity for the 128 neurons and the loop direction (Colab GPU)."""
import base64, os, subprocess, sys

PAYLOAD = __PAYLOAD__
root = "/content/glia_work"
for name, b64 in PAYLOAD.items():
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("payload written to", root, flush=True)

if not os.path.isdir("/content/neuron-lens"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/stdlogits/neuron-lens", "/content/neuron-lens"],
                   check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-e", "/content/neuron-lens", "datasets"], check=True)

import torch
print("cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
      flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no GPU in this session")

os.makedirs("/content/glia_out/identity", exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = "/content/neuron-lens:" + env.get("PYTHONPATH", "")
# the T4 (sm_75) has 64 KB of shared memory; a triton config the estimator wants needs
# 96 KB, so torch.compile dies with InductorError. Eager mode is slower but works.
env["TORCHDYNAMO_DISABLE"] = "1"
os.environ.update(env)
subprocess.run([sys.executable, os.path.join(root, "src", "identity.py")],
               check=True, env=env)
'''

out = os.path.join(ROOT, "colab_identity_run.py")
open(out, "w").write(SCRIPT.replace("__PAYLOAD__", repr(payload)))
print("wrote", out, f"({os.path.getsize(out)/1024:.1f} KB)")
