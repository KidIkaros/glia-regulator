"""Pure helpers the experiment drivers share - tested without a GPU or a filesystem.

Two bugs motivated these tests, both found in the drivers this file now serves:

1. `base_looped` is stored in s1_pairs.json as the STRING "False", which is truthy for
   both values. A truth-test silently classified every problem as a looper; the shared
   `as_bool` is now the single place that knowledge lives.
2. A results file can be truncated mid-line by a killed process. `read_jsonl` must skip
   the broken line and keep every good row - a resume that loses all prior work is worse
   than no resume.

    ~/.venvs/glia/bin/python tests/test_runlog.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from runlog import append_jsonl, as_bool, find_pairs, missing_keys, read_jsonl


def test_as_bool_parses_the_string_false():
    """The exact bug: "False" is truthy in Python."""
    assert as_bool("False") is False
    assert as_bool("false") is False
    assert as_bool("True") is True
    assert as_bool(False) is False
    assert as_bool(True) is True


def test_read_jsonl_survives_a_truncated_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.jsonl")
        with open(path, "w") as handle:
            handle.write(json.dumps({"key": "a", "closed": True}) + "\n")
            handle.write('{"key": "b", "closed": fal')          # killed mid-write
            handle.write("\n" + json.dumps({"key": "c", "closed": False}) + "\n")
        rows = read_jsonl(path)
        assert set(rows) == {"a", "c"}, f"lost good rows or kept the broken one: {set(rows)}"
        assert rows["a"]["closed"] is True


def test_read_jsonl_on_missing_file_is_empty_not_an_error():
    assert read_jsonl("/nonexistent/path/rows.jsonl") == {}


def test_append_then_read_roundtrips_and_keys_are_stamped():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "rows.jsonl")
        append_jsonl(path, {"closed": True, "n": 10}, key="m_p1")
        rows = read_jsonl(path)
        assert rows["m_p1"]["closed"] is True and rows["m_p1"]["n"] == 10


def test_missing_keys_resumes_and_preserves_order():
    """The resume logic is pure: recorded work is skipped, order is deterministic."""
    pairs = [{"id": "p1"}, {"id": "p2"}]
    rows = {"m1_p1": {}}
    work = missing_keys(["m1", "m2"], pairs, rows)
    assert work == [("m1", {"id": "p2"}), ("m2", {"id": "p1"}), ("m2", {"id": "p2"})]


def test_find_pairs_reports_every_root_it_tried():
    try:
        find_pairs("/definitely/not/here", "/nor/here")
    except FileNotFoundError as error:
        assert "/definitely/not/here" in str(error)
    else:
        raise AssertionError("find_pairs must raise when no root contains s1_pairs.json")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("runlog: 6 tests green")
