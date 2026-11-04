"""Collect the dose-sweep rows from the Colab sessions and grade them against the gates.

    python3 scripts/collect_sweep.py a8d8a8 d3b480

Each session's VM holds a JSONL of one row per (alpha, problem) plus one panel row per
alpha. The rows are pulled back over the authenticated CLI, merged, and graded against
the pre-registered gates:

  * the best alpha must close >= 70% of the problems;
  * and at that alpha, ordinary_kl <= 0.05 and ppl_delta <= +20%.

The healthy controls (problems the base model already closes) are reported separately:
the intervention must not break them either, and their closed rate is the sharpest
available check that the lever is not simply making the model terse.
"""
import json
import os
import subprocess
import sys

SNIPPET = r'''
import os, json
rows = []
for name in ("sweep_a.jsonl", "sweep_b.jsonl"):
    p = os.path.join("/content/glia_out", name)
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
print("@@ROWS@@" + json.dumps(rows))
'''


def pull(session):
    out = subprocess.run(
        ["colab", "exec", "-s", session, "--timeout", "600"],
        input=SNIPPET, capture_output=True, text=True, timeout=900)
    text = out.stdout + out.stderr
    marker = "@@ROWS@@"
    if marker not in text:
        print(f"[{session}] no rows returned (session busy or expired)")
        return []
    payload = text.split(marker, 1)[1].strip()
    payload = payload.split("\n", 1)[0] if "\n" in payload else payload
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError:
        print(f"[{session}] row payload was not valid JSON")
        return []
    print(f"[{session}] {len(rows)} rows")
    return rows


def grade(rows):
    summary = {}
    for row in rows:
        alpha = row.get("alpha")
        if alpha is None:
            continue
        entry = summary.setdefault(str(alpha), dict(closed=0, n=0, healthy_closed=0,
                                                     healthy_n=0, panel=None))
        if row.get("kind") == "panel":
            entry["panel"] = dict(ordinary_kl=row["ordinary_kl"],
                                  ordinary_agree=row["ordinary_agree"],
                                  ppl_base=row["ppl_base"], ppl_at_alpha=row["ppl_at_alpha"],
                                  ppl_delta_pct=row["ppl_delta_pct"])
        else:
            entry["n"] += 1
            entry["closed"] += int(bool(row["closed"]))
            if not row.get("base_looped", False):
                entry["healthy_n"] += 1
                entry["healthy_closed"] += int(bool(row["closed"]))
    for entry in summary.values():
        entry["closed_rate"] = (entry["closed"] / entry["n"]) if entry["n"] else None
        entry["healthy_closed_rate"] = ((entry["healthy_closed"] / entry["healthy_n"])
                                        if entry["healthy_n"] else None)
        panel = entry.get("panel") or {}
        entry["gate_closed_rate"] = bool(entry["closed_rate"] is not None
                                         and entry["closed_rate"] >= 0.70)
        entry["gate_capability"] = bool(panel.get("ordinary_kl", 1) <= 0.05
                                        and panel.get("ppl_delta_pct", 1e9) <= 20.0)
        entry["gate_pass"] = entry["gate_closed_rate"] and entry["gate_capability"]
    return summary


def render(summary):
    """The one table a human reads."""
    print("\n=== DOSE-RESPONSE (alpha: closed rate | healthy | ordinary_kl | ppl%) ===")
    for alpha in sorted(summary, key=float):
        entry = summary[alpha]
        panel = entry.get("panel") or {}
        print(f"  alpha {alpha:>4}: {entry['closed']}/{entry['n']} closed "
              f"({entry['closed_rate']:.2f}) | healthy {entry['healthy_closed']}/"
              f"{entry['healthy_n']} | kl {panel.get('ordinary_kl', float('nan')):.4f} | "
              f"agree {panel.get('ordinary_agree', float('nan')):.3f} | "
              f"ppl {panel.get('ppl_delta_pct', float('nan')):+.2f}% | "
              f"gate {'PASS' if entry['gate_pass'] else 'fail'}")


def main():
    sessions = sys.argv[1:] or ["a8d8a8", "d3b480"]
    rows = []
    for session in sessions:
        rows.extend(pull(session))
    if not rows:
        print("no rows collected")
        return 1
    seen, unique = set(), []
    for row in rows:
        if row.get("key") not in seen:
            seen.add(row.get("key"))
            unique.append(row)
    summary = grade(unique)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "lab", "colab")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(rows=unique, summary=summary),
              open(os.path.join(out_dir, "sweep_results.json"), "w"), indent=1)
    render(summary)
    print("\nsaved:", os.path.join(out_dir, "sweep_results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
