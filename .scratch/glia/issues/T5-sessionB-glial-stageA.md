# T5 — Session B: glial Stage A (train + eval)
status: in progress
Blocks: none. Depends on: T2 (rule recovered), ground-truth neuron indices.
Done so far: src/stageA.py (train/eval/selectivity/knockout), build_stageA.py,
tests/test_payload_layout.py (RED/GREEN proven against the path bug that killed v1).
Next: push v2, verify GATE 2 (identity at init), train 3 epochs, eval 18 paired
problems, selectivity, knockout. Bands: closed >= 70% and knockout <= 25% = SUCCESS.
