"""The capability panel must be able to run without torch installed.

The panel's measurement code needs torch and numpy, but its PROMPT SETS and its
scoring logic are pure Python - and they are the part that encodes what we mean by
"capability preserved". Testing them locally (with the heavy imports stubbed) keeps
them honest: a prompt set that silently shrinks, or a summary that silently drops the
verbosity or healthy-answer checks, is exactly the kind of quiet weakening that makes
a capability claim hollow.

Style note: like the other tests in this directory, this is a plain script of asserts
that can be run directly (python3 tests/test_capability_panel.py) or inside a kernel.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub(name):
    if name not in sys.modules:
        module = types.ModuleType(name)
        setattr(module, "float32", "float32")
        sys.modules[name] = module


for _name in ("numpy", "torch"):
    _stub(_name)
sys.path.insert(0, os.path.join(ROOT, "src"))
import capability_panel as panel  # noqa: E402


def test_ordinary_prompts_cover_distinct_capabilities():
    """The panel must not be one kind of prompt wearing 20 hats.

    If every prompt were "explain a concept", a lever that damaged, say, code or
    arithmetic would go unnoticed. We therefore require coverage of prose, factual
    recall, arithmetic with a worked answer, code, translation, constrained
    instruction following, and long-form generation.
    """
    prompts = " ".join(panel.ORDINARY_PROMPTS).lower()
    assert len(panel.ORDINARY_PROMPTS) >= 18, "too few prompts to support a claim"
    for marker, label in [
        ("dollars", "arithmetic"),
        ("python", "code"),
        ("translate", "translation"),
        ("exactly three", "constrained instruction following"),
        ("several paragraphs", "long-form generation"),
        ("who wrote", "factual recall"),
    ]:
        assert marker in prompts, f"the panel lost its {label} coverage"


def test_long_form_prompts_are_a_subset_of_the_ordinary_set():
    """The long-form prompts must also be exercised by the panel, so the verbosity
    check is measured on the regime the lever could plausibly damage."""
    for prompt in panel.LONG_PROMPTS:
        assert prompt in panel.ORDINARY_PROMPTS


def test_healthy_controls_carry_their_expected_answers():
    """A healthy control without an expected answer cannot detect a broken answer."""
    assert len(panel.HEALTHY_CONTROLS) >= 2
    for control in panel.HEALTHY_CONTROLS:
        assert control["expect"], "a healthy control must state what it expects"
        assert control["expect"].isdigit()
        assert control["question"].strip()


def test_summarise_reports_every_dimension():
    """The summary line a human reads must include the KL, the argmax agreement, the
    perplexity delta, the verbosity, and the healthy-answer count. Dropping any one
    of them would let a regression pass unnoticed."""
    metrics = dict(ordinary_kl=0.0047, ordinary_agree=1.0, ppl_base=9.1,
                   ppl_at_alpha=9.4, ppl_delta_pct=3.07, mean_chars=410.0)
    healthy = [dict(found=True), dict(found=False)]
    line = panel.summarise(metrics, healthy)
    for fragment in ("kl", "agree", "ppl", "chars", "healthy answers"):
        assert fragment in line, f"the summary dropped '{fragment}'"
    assert "1/2" in line, "the healthy-answer count must be visible as a fraction"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("capability panel: prompt sets and scoring logic verified")
