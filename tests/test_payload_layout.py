"""Regression test for the failure class that killed two kernels: payload files written
to a layout the driver does not read from (today: s1_pairs.json / gt*.npy looked up in
src/ while the payload puts them at the root).

Reproduces build_stageA.py's payload -> extraction, then asserts:
  1. every payload file the driver opens exists in the extracted layout;
  2. the driver resolves those files against the payload ROOT (dirname(dirname(__file__))),
     not against src/ (dirname(__file__)).

Pure file ops: no GPU, no model, no network.

Run:  python3 tests/test_payload_layout.py   (exit 0 = pass)
"""
import ast
import base64
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["KAGGLE_KERNEL_ID"] = "mauricew/glia-stageA-test"
import importlib
import build_stageA
importlib.reload(build_stageA)

kernel_main = os.path.join(ROOT, "kernel_stageA", "main.py")
assert os.path.exists(kernel_main), "builder did not emit kernel main.py"

src = open(kernel_main).read()
m = re.search(r"PAYLOAD = (\{.*?\})\n", src, re.S)
payload = {k: base64.b64decode(v) for k, v in ast.literal_eval(m.group(1)).items()}

root = tempfile.mkdtemp(prefix="glia_payload_")
for name, b in payload.items():
    p = os.path.join(root, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(b)

driver_src = payload["src/stageA.py"].decode()

# 1. classify every os.path.join(...,"file") in the driver: payload READ vs OUTPUT WRITE
writes = set()
reads = {}
for line in driver_src.splitlines():
    if "os.path.join(" not in line:
        continue
    for prefix, fname in re.findall(r"os\.path\.join\(([^,]+(?:\([^()]*\))?),\s*[\"']([\w./-]+\.(?:json|npy|py|pt))[\"']\)",
                                     line):
        if "OUT" in prefix:
            writes.add(fname)          # written to /kaggle/working, not a payload read
        else:
            reads[fname] = prefix.strip()

print("driver payload reads:", reads)
print("driver output writes (ignored):", sorted(writes))
assert reads, "no payload reads found in driver (regex drift - update the test)"
missing = [f for f in reads if not os.path.exists(os.path.join(root, f))]
assert not missing, f"PATH BUG: driver reads {missing}, absent from payload root"

# 2. the lookup must use the ROOT expression verbatim (exact string check, no regex)
for fname in reads:
    ok = (f'os.path.join(ROOT, "{fname}")' in driver_src
          or f"os.path.join(ROOT, '{fname}')" in driver_src)
    assert ok, (f"PATH BUG: {fname} is not joined with the payload ROOT expression; "
                f"it must read os.path.join(ROOT, \"{fname}\")")

# 3. payload JSON shape matches what the driver assumes
data = json.load(open(os.path.join(root, "s1_pairs.json")))
assert isinstance(data, dict) and isinstance(data.get("pairs"), list), \
    "s1_pairs.json must be {\"pairs\": [...]}"
assert data["pairs"] and isinstance(data["pairs"][0], dict), "pairs entries must be dicts"
for key in ("id", "problem", "healthy_completion"):
    assert key in data["pairs"][0], f"pairs entries missing {key!r}"

print("PASS: every referenced payload file exists at the root the driver resolves it to, "
      "and payload JSON shapes match the driver's assumptions")
