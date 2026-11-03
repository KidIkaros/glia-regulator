# T2 — gradient attribution search (src/attribute.py)
status: done (with negative result)
Outcome: implementation shipped; three bug classes found and fixed (zero-gradient graph
attach, multiplicative fp16 overflow, arange-weighted objective overflow). Corrected
pipeline (fp32 + additive push + centered objective) runs clean.
