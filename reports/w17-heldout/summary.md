# W17 frozen held-out H46–H48

All three frozen cases MATCH, recorded as-is with defaults; no tuning or changed calls.

| Case | TorchOED | Reference | Frozen verdict |
|---|---|---|---|
| H46 npk least-squares | R model matrix and term order; max relative 8.253e-15, max absolute p 2.693e-15 | R lm/anova; threshold 1e-8 | MATCH |
| H47 mixed-level OA | 16 runs, strength 2, A3 = 34 | DoE.base: 16 runs, strength 2, A3 = 34 | MATCH |
| H48 constrained mixture | 13 vertices | Exact Fraction enumeration: 13 vertices; missing 0, extra 0 at 1e-12 | MATCH |

One R dispatch: [45924802098](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45924802098), source 1af147210d168a8e834305e630a22b1bb452289f, SUCCESS.
R 4.5.2, DoE.base 1.2-5; Python 3.14.6, torch 2.15.0+cpu, CPU float64, one thread. Full sessionInfo and workflow runtime retained.
`npk.csv`, `r.json` and `sessionInfo.txt` match downloaded bytes; runtime.json retains all downloaded fields and adds the one-time TorchOED outputs/timing.
`PYTHONPATH=src python reports/w17-heldout/run.py`: sandbox A3 3 PASS, Fraction toy 1 PASS, 0 FAIL. First execution 4.221 s including interpreter startup; scientific work 0.130 s.
Reproduction from frozen R outputs and cached one-time TorchOED values: byte-identical results/runtime PASS, 3.922 s including startup (scoring 0.051 s). No held-out constructor/fit rerun.
The OA scorer is imported unchanged from w17-oa-design. H48 decimal-string call accepted as frozen; independent Fraction Gaussian elimination enumerates all boundary triples.
Eight report files, 86 Python/R lines (S100); no source, tests, docs, tools, matrix or website changes. No GPU in this task.
Next: Architect review; week-18 docs task may cite these results for the matrix HO cells.
