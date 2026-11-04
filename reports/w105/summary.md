# W1 A/I two-swap: paired CPU benchmark

Base `9f2bab790508bf8270d0c597e3ebc4833dc09d60`; timed source `e1f0f38dc5a1b880a87447aab12e5de0a5764fca`. One same-session base/head tools/bench_b0.py pair; no held-out inputs or reruns. Raw timing gate FAIL on grid4 n=15; no tuning or additional benchmark attempt.
40/40 feasible TorchOED D rows have exactly identical IDs, logdet, certificates and stop reasons. Grid2 n=12 is the unchanged infeasible control.
D function and timed callees are unchanged; Architect ruling is requested for the raw timing exception (the earlier #92 byte-identical-path precedent may apply). This report does not claim measured timing PASS.

| Case | n | Base s | Head s | Head/base |
|---|---|---|---|---|
| grid2 | 6 | 0.037257569 | 0.038742474 | 1.039855123 |
| grid2 | 7 | 0.034714574 | 0.035893218 | 1.033952426 |
| grid3 | 10 | 0.091263719 | 0.087768668 | 0.961703829 |
| grid3 | 20 | 0.096392753 | 0.100386948 | 1.041436673 |
| grid4 | 15 | 0.249582185 | 0.282494430 | 1.131869368 |
| grid4 | 30 | 0.327272817 | 0.317833276 | 0.971156966 |
| random500 | 10 | 0.608444758 | 0.604128214 | 0.992905611 |
| random500 | 20 | 1.506596278 | 1.477368622 | 0.980600207 |

A/I on predetermined grid3 n=20, default restarts=10 and seed0, three-call median: {"A": {"seconds": 0.17250387699999692, "objective": -1.592582833162543, "floor": 0.588018530727401, "certificate": "LOCAL_2_NUMERICAL"}, "I": {"seconds": 0.18874507099997118, "objective": -0.5034202240473651, "floor": 0.5111935092195257, "certificate": "LOCAL_2_NUMERICAL"}}
Colab T4 host CPU only, Python 3.13.16, torch 2.11.0+cu130, PyDOE1.6.0, one thread. Tesla T4 observed but unused by the benchmark. Cleanup PASS, assignments0→0. A/I ≤2× D is a hypothesis, not a gate.
