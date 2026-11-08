# W131: exact pair bookkeeping

Base4a5c604; measured head9e7e755; PR#135. Same Windows dev venv/wheel as #126; one thread.
Command: `python CAPTURE.py ROOT <pydoe-1.6.0-py3-none-any.whl> OUTPUT`; results.json/capture_adapter calls unchanged tools/bench_b0.py and captures IDs/objectives in memory.
D:3 calls/seed; A/I:1; seeds0..4, restarts1. Capture overhead is inside both fresh timed wrappers. Default10-restart D/PyDOE rows also retained.
Profile: pair_tiles49.60%, move bookkeeping1.23%, det/solve18.22%, gather/matmul/assembly/dispatch residual30.95% of N2. Top15, command and script: results.json/profile.

| Case | #126 A:D / I:D | Fresh base A:D / I:D | Head A:D / I:D | D seconds base -> head |
|---|---|---|---|---|
| grid2 n=6 | 0.774 / 0.834 | 0.858 / 1.051 | 0.815 / 0.960 | 0.006031 -> 0.005740 |
| grid2 n=7 | 0.724 / 0.856 | 0.998 / 0.971 | 0.829 / 0.948 | 0.005359 -> 0.005345 |
| grid3 n=10 | 2.766 / 1.946 | 2.891 / 2.053 | 2.310 / 1.571 | 0.011160 -> 0.011685 |
| grid3 n=20 | 1.408 / 1.490 | 1.398 / 1.547 | 1.249 / 1.312 | 0.010940 -> 0.010918 |
| grid4 n=15 | 39.011 / 20.822 | 38.369 / 20.089 | 29.617 / 15.634 | 0.025517 -> 0.024903 |
| grid4 n=30 | 96.189 / 96.981 | 90.157 / 90.988 | 68.977 / 70.092 | 0.026170 -> 0.025739 |
| random500 n=10 | 180.745 / 259.233 | 175.630 / 263.508 | 125.443 / 184.134 | 0.114574 -> 0.116343 |
| random500 n=20 | 517.716 / 261.011 | 521.172 / 262.992 | 355.502 / 181.353 | 0.156766 -> 0.164499 |

Equality:120/120 D/A/I IDs and objectives exact;15/15 infeasible errors exact;40/40 default D reference rows reproduced.
D no-regression condition NOT MET as measured: grid3 n10 +4.71%, random500 n10 +1.54%, random500 n20 +4.93%; other five D medians lower. Single sequential runs; no statistical no-regression claim.
C1 <=2x target still NOT MET. Random500 n20 A81.701981->58.479704s; I41.228229->29.832401s. No held-out tuning.
Local source9e7e755: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`:100 passed,0 failed,145.07s; one optional NumPy warning. Final CI linked in ENGINEER_HANDOFF.
Only row decode and best/cutoff bookkeeping changed; linalg, tile boundaries, capacity, tolerances and search order unchanged. One independent itertools test:width7 and1.
