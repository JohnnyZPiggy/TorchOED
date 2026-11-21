# W15 held-out H40–H42 (#512)

Frozen calls and seeds from docs/WEEK_GOAL.md, run as-is at main d869c1a; no tuning.

| Case | TorchOED | Reference | Verdict | Δ | Feasible / maximum count |
|---|---|---|---|---|---|
| H40 | 2097152/441 | 2097152/441 | MATCH | 0 | 4304 / 128 |
| H41 | 2097152/315 | 2097152/315 | MATCH | 0 | 5952 / 48 |
| H42 | WLP [0.0, 0.0, 0.0, 38.0, 0.0, 52.0, 0.0, 33.0, 0.0, 4.0, 0.0, 0.0] | WLP [0.0, 0.0, 0.0, 38.0, 0.0, 52.0, 0.0, 33.0, 0.0, 4.0, 0.0, 0.0] | MATCH | no differing word length | N/A |

H40/H41: enumerate all 16,384 frozen level assignments per case, retaining multiplicity ≤ 2.
Length-16 integer A/B level labels use the documented API; contiguous nested strata / row-major strip grid.
Exact Fraction dense GLS determinant equals integer Bareiss on scaled V⁻¹ for both returned designs (2 PASS, 0 FAIL). Feasibility 2 PASS, 0 FAIL.

H42: one [r-oracle run 44868066239](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44868066239), recipe committed at def9885d8bb1e72b487ad034901dfa36181ad052; R version 4.5.2 (2025-10-31), FrF2 2.3-5. sessionInfo.txt and runtime.json retained.
R CSV is scored with the same word_length_pattern as TorchOED. No reference failures.

Reproduce: `PYTHONPATH=src python reports/w15-heldout/run.py` (byte-identical results.json).
Local CPU float64, one thread; exact versions in results.json. No GPU or source changes.
