"""The rollout must record WHERE the thinking closed, not just that it did.

A generation can close its thinking at token 50 or at token 8,100 of an 8,192-token
budget, and those are not the same outcome: "closed" is a boolean that hides the very
difference the dose-response needs. The close position makes it measurable.

    ~/.venvs/glia/bin/python tests/test_close_position.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from harness import close_position


def test_returns_none_when_the_thinking_never_closed():
    assert close_position("the model kept re-deriving the answer...") is None


def test_returns_the_character_offset_of_the_marker():
    text = "step one, step two</think> and here is the final answer"
    assert close_position(text) == text.index("</think>")


def test_takes_the_FIRST_marker_not_the_last():
    """A model can emit the marker more than once; the first close is the event."""
    text = "</think> thinking again</think> and the answer"
    assert close_position(text) == 0


def test_handles_a_marker_at_the_very_start():
    assert close_position("</think>") == 0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("close position: verified")
