"""The resume loop must be testable without a GPU, a model, or the network.

The probe's whole reliability claim is "kill it any time; re-run and it continues where
it stopped". That claim lives in `run_chunk`, so it gets a test with a fake model: the
loop must (a) write the expected rows, (b) skip already-recorded work on a second run,
(c) stop at the chunk budget, and (d) never reload the model for consecutive items of
the same model.

torch and harness are stubbed at import: this file is about control flow, not numerics.
"""
import json
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _FakeTensor:
    def to(self, _device):
        return self


class _FakeParam:
    device = "cpu"


class _FakeModel:
    def parameters(self):
        return iter([_FakeParam()])


class _FakeTok:
    def decode(self, tokens, skip_special_tokens=True):
        return "decoded text"


torch = _stub("torch", float16="float16", float32="float32",
              cuda=_stub("torch.cuda", is_available=lambda: False, empty_cache=lambda: None))
harness = _stub(
    "harness",
    block_triton_kernels=lambda: None,
    eos_token_ids=lambda tok: [0],
    prompt_ids=lambda tok, problem: _FakeTensor(),
    load_pairs=lambda path: [],
    load_model=lambda model_id, device=None, dtype=None: (_FakeModel(), _FakeTok()),
    greedy_rollout=lambda model, encoded, rollout, stop_ids, tokenizer=None,
    presence_penalty=0.0: ([1, 2], dict(n=2, closed=False, chars=10, seconds=0.1,
                                        close_position=None)),
)
sys.path.insert(0, os.path.join(ROOT, "src"))
import family_probe  # noqa: E402

CALLS = {"rollouts": 0}


def _counting_rollout(*args, **kwargs):
    CALLS["rollouts"] += 1
    return [1, 2], dict(n=2, closed=True, chars=10, seconds=0.1, close_position=0)


def test_run_chunk_writes_rows_then_resumes_without_repeating():
    family_probe.greedy_rollout = _counting_rollout
    pairs = [{"id": "p1", "problem": "q1", "base_looped": "False"},
                 {"id": "p2", "problem": "q2", "base_looped": "True"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.jsonl")
        loader_calls = []

        def loader(model_id):
            loader_calls.append(model_id)
            return _FakeModel(), _FakeTok()

        written = family_probe.run_chunk(["m1"], pairs, path, chunk=2, loader=loader)
        assert written == 2, f"expected 2 rows, got {written}"
        rows = [json.loads(line) for line in open(path)]
        assert [row["key"] for row in rows] == ["m1_p1", "m1_p2"]
        assert rows[0]["base_looped"] is False, "string 'False' must parse to False"
        assert rows[1]["base_looped"] is True
        assert rows[0]["close_position"] == 0
        assert loader_calls == ["m1"], "model must load once for consecutive same-model items"

        # second run: everything recorded -> nothing to do, no extra rollout
        before = CALLS["rollouts"]
        assert family_probe.run_chunk(["m1"], pairs, path, chunk=2, loader=loader) == 0
        assert CALLS["rollouts"] == before, "resume must not re-run recorded work"


def test_run_chunk_respects_the_chunk_budget():
    family_probe.greedy_rollout = _counting_rollout
    pairs = [{"id": f"p{i}", "problem": f"q{i}", "base_looped": False} for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.jsonl")
        written = family_probe.run_chunk(["m1"], pairs, path, chunk=2,
                                         loader=lambda m: (_FakeModel(), _FakeTok()))
        assert written == 2
        assert len([l for l in open(path) if l.strip()]) == 2


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("family_probe: resume loop verified")
