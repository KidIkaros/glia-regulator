"""Shared pure helpers for the experiment drivers: parsing, jsonl, problem lookup.

Extracted after a clean-code/functional-design review found `as_bool` duplicated in four
drivers, `find_pairs` in three, and `read_rows`/`record` in two - the same bug (the
string-"False" truthiness trap) was fixed independently in each copy, which is exactly
how one copy silently stays broken.

Functional-design rules applied here:
  * every function is pure unless its name says otherwise; the impure ones
    (`read_jsonl`, `append_jsonl`) are thin edges over the filesystem and do no logic;
  * no module-level execution - importing this file has no effects;
  * values in, values out - no mutation of arguments, no global state.
"""
import json
import os


def as_bool(value):
    """Parse a JSON-ish boolean.

    s1_pairs.json stores `base_looped` as the STRING "False", which is truthy for both
    values. Truth-testing it silently classifies every problem as a looper; this parser
    is the single place that knowledge lives.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def find_pairs(*roots):
    """The paired-problems file, looked up next to the driver and one level up.

    `roots` are candidate directories in priority order; raises if none contains
    `s1_pairs.json`, so a mislaid payload fails loudly instead of silently measuring
    nothing.
    """
    for root in roots:
        candidate = os.path.join(root, "s1_pairs.json")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"s1_pairs.json not found in any of: {list(roots)}")


def read_jsonl(path):
    """Rows of a jsonl file, keyed by their `key` field; bad lines are skipped.

    A truncated or partially written line must not destroy the whole results file - the
    drivers append after every unit of work precisely so a crash loses at most one row.
    """
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" in row:
            rows[row["key"]] = row
    return rows


def append_jsonl(path, payload, key):
    """Append one row, stamping it with its key. The only write in this module."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(dict(payload, key=key)) + "\n")


def missing_keys(models, pairs, rows):
    """The (model, problem) work items not yet recorded, in deterministic order.

    Pure: takes the current rows as a value and returns the work list, so the resume
    logic is testable without a filesystem or a GPU.
    """
    return [(model, problem) for model in models for problem in pairs
            if f"{model}_{problem['id']}" not in rows]
