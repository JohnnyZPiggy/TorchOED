# W131: exact pair bookkeeping

Base4a5c604; measured headf83993c; PR#135. Same Windows dev venv/wheel as #126; one thread. Earlier9e7e755 measurement superseded by the linear-prefix memory correction.
Command: `python CAPTURE.py ROOT <pydoe-1.6.0-py3-none-any.whl> OUTPUT`; results.json/capture_adapter calls unchanged tools/bench_b0.py and captures IDs/objectives in memory.
D:3 calls/seed; A/I:1; seeds0..4, restarts1. Capture overhead is inside both fresh timed wrappers. Default10-restart D/PyDOE rows also retained.
Profile: pair_tiles49.60%, move bookkeeping1.23%, det/solve18.22%, gather/matmul/assembly/dispatch residual30.95% of N2. Top15, command and script: results.json/profile.

| Case | #126 A:D / I:D | Fresh base A:D / I:D | Head A:D / I:D | D seconds base -> head |
|---|---|---|---|---|
| grid2 n=6 | 0.774 / 0.834 | 0.858 / 1.051 | 0.883 / 0.978 | 0.006031 -> 0.005777 |
| grid2 n=7 | 0.724 / 0.856 | 0.998 / 0.971 | 0.888 / 0.951 | 0.005359 -> 0.005563 |
| grid3 n=10 | 2.766 / 1.946 | 2.891 / 2.053 | 2.339 / 1.622 | 0.011160 -> 0.011582 |
| grid3 n=20 | 1.408 / 1.490 | 1.398 / 1.547 | 1.118 / 1.234 | 0.010940 -> 0.012061 |
| grid4 n=15 | 39.011 / 20.822 | 38.369 / 20.089 | 29.873 / 15.680 | 0.025517 -> 0.024691 |
| grid4 n=30 | 96.189 / 96.981 | 90.157 / 90.988 | 71.095 / 72.250 | 0.026170 -> 0.024825 |
| random500 n=10 | 180.745 / 259.233 | 175.630 / 263.508 | 123.538 / 185.391 | 0.114574 -> 0.115830 |
| random500 n=20 | 517.716 / 261.011 | 521.172 / 262.992 | 357.867 / 181.230 | 0.156766 -> 0.164880 |

Equality:120/120 D/A/I IDs and objectives exact;15/15 infeasible errors exact;40/40 default D reference rows reproduced.
D no-regression condition NOT MET as measured: grid2 n=7 +3.81%, grid3 n=10 +3.78%, grid3 n=20 +10.25%, random500 n=10 +1.10%, random500 n=20 +5.18%. Other three D medians lower. Single sequential runs; no statistical no-regression claim.
C1 <=2x target still NOT MET. Random500 n20 A81.701981->59.005186s; I41.228229->29.881185s. No held-out tuning.
Local sourcef83993c: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`:100 passed,0 failed,137.87s; one optional NumPy warning. Final CI linked in ENGINEER_HANDOFF.
Only per-tile row decode and best/cutoff bookkeeping changed; linalg, tile boundaries, capacity, tolerances and search order unchanged. Linear prefix storage; one independent itertools test:width7 and1.
