# B4: G2 scale measurement

**G2 MET**: three fixed seeds; target <=120 s each with a checked certified lower bound.

Command: `python tools/bench_g2.py` in the existing dev venv. random2000 seed 0, quadratic p=10, n=20, restarts=10; all other parameters default, one thread, one timed call per seed. Timing includes copying existing restart diagnostics; independent bench_b0.score_design rescoring is excluded. No PyDOE search or GPU run. Bounds are floors, not achieved efficiencies.

| Seed | Wall s | Objective | Stop / certificate | Returned bound | Independent bound | Absolute difference | Check |
|---|---|---|---|---|---|---|---|
| 0 | 9.83178 | 20.4956078974 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL | 0.843561018467 | 0.843561018467 | 1.11e-16 | True |
| 1 | 8.69558 | 20.4956078974 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL | 0.843561018467 | 0.843561018467 | 0 | True |
| 2 | 11.6931 | 20.4956078974 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL | 0.843561018467 | 0.843561018467 | 0 | True |
