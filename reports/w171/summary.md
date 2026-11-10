# W171: A/I dual-floor amendment

Measured head `69b53990f5985119d2e778bf705dfe7078d75a1d` vs base `e0d131f9bbc43b184ecc832ad49833c77f38c0a7`; Refs #171, PR #172.
Command: `../.sae-w95-venv/Scripts/python.exe scratch/run_paired.py`, then `scratch/aggregate.py` (exact drivers in results.json). Unmodified full bench_b0, same-session base/head/base/head as w162; all cases/seeds/repeats; medians across five seed medians then two runs. Separate untimed outputs/floors, no concurrent tests.
Environment: Python3.14.6, torch2.15.0+cpu, NumPy2.5.4, SciPy1.18.1, PyDOE1.6.0; one thread, same frozen wheel. No GPU/R/Julia/held-out or remote sessions.
D: 160/160 timed rows and 40/40 untimed IDs/objectives/bounds byte-identical; timing 8/8 within durable 1.10x allowance. A/I: 80/80 untimed IDs/objectives identical; expected infeasible errors 15/15 identical.
C1: both A/I <=2D on 5/8 cases (was 4/8); individual criteria 11/16. Remaining slow cases are reported as-is. A/I bound new/old 0.999196132–0.999618070, no tuning.
Partial initial baseline interrupted while pytest was finishing; discarded before completion. First complete four-run sequence retained, no third round. Raw logs/checkouts are in task-owned .sae-171-20261110/scratch until merge.
Source tests: 116 passed, 0 failed; independent dense exponent-half reference converged to phi/qmax >=1-1e-12 for grid3 n20 A/I, lower/upper inequalities pass. Final-SHA tests and CI in handoff.

| Case | Base D s | Head D s | D head/base | Head A:D | Head I:D | Floor A new/old | Floor I new/old |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid2 n=6 | 0.00602784997 | 0.00626559998 | 1.03944192 | 1.07371999 | 0.926168288 | 0.999458704 | 0.999219569 |
| grid2 n=7 | 0.00611005002 | 0.00571125001 | 0.934730484 | 1.00580434 | 0.865581086 | 0.999458704 | 0.999219569 |
| grid3 n=10 | 0.01221085 | 0.01178665 | 0.965260406 | 0.951483242 | 0.889429988 | 0.99961807 | 0.999387094 |
| grid3 n=20 | 0.0117907 | 0.0112618501 | 0.955146857 | 0.777181364 | 0.814977111 | 0.99961807 | 0.999387094 |
| grid4 n=15 | 0.0261909501 | 0.0255845001 | 0.976845055 | 3.57429498 | 1.70288065 | 0.999530417 | 0.999350404 |
| grid4 n=30 | 0.02579695 | 0.02531305 | 0.98124197 | 3.54178971 | 3.2659636 | 0.999530417 | 0.999350404 |
| random500 n=10 | 0.11765675 | 0.1162948 | 0.988424378 | 1.03965912 | 1.11759081 | 0.9995242 | 0.999196132 |
| random500 n=20 | 0.1661858 | 0.1671078 | 1.00554801 | 5.33196416 | 4.2722539 | 0.9995242 | 0.999196132 |
