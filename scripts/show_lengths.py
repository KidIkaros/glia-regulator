import json

rows = {}
for line in open("/home/ikaaros/Coding/kaggle/wsp/artifacts/s2/results_ours.jsonl"):
    r = json.loads(line)
    rows[r["id"]] = r
for pid in ["gsm8k_train_0", "gsm8k_train_1", "gsm8k_train_3"]:
    r = rows[pid]
    out = r.get("output", "")
    print(pid, "| closed:", "</think>" in out, "| chars:", len(out),
          "| n_gen:", r.get("n_gen"), "| cap_hit:", r.get("cap_hit"))
