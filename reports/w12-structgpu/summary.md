# W12 structured GPU evidence: BLOCKED

Refs #362. No GPU result or speed claim. The single permitted controller allocation was interrupted after a local adapter mistake was discovered; cleanup is confirmed. A replacement corrected session needs an Architect ruling. The P2 Dev cells remain open.

Corrected local driver source: `cd0e0868c2a975a5ebdc7d548ecec7f438cdcbcb`; local torch 2.15.0+cpu, one thread, CPU float64. Candidate grid 3^4, degree 2, n=24, default 10 restarts, seed 0. B4 removes the intercept; S4 hard labels use C[:,0].long(), matching tests/test_blocked_design.py::test_split_plot_matches_frozen_exhaustive_optimum. One B4 D warm-up. Wall time is whole-call / 10; no per-restart loop change.

| Case | Criterion | Device | Objective | Floor | IDs equal | Seconds/restart | Status |
|---|---|---|---|---|---|---|---|
| B4 | D | cpu | 32.95803107277601 | 0.8910042544570046 | CPU self only | 0.1392580600004294 | PASS |
| B4 | A | cpu | -1.6387385605965425 | 0.930435359688778 | CPU self only | 0.21935935000001336 | PASS |
| B4 | I | cpu | -0.5895762572514831 | 0.8836851617191598 | CPU self only | 0.2087047300003178 | PASS |
| S4 | D | cpu | 31.02750714031874 | 0.6747751931300998 | CPU self only | 0.28974453999981054 | PASS |
| S4 | A | cpu | -3.592435785724454 | 0.5082189996903211 | CPU self only | 0.25416236000019127 | PASS |
| S4 | I | cpu | -1.1799983713055593 | 0.4979594918513877 | CPU self only | 0.2409156499998062 | PASS |
| B4 | D | cuda | — | — | NR | — | NOT_RUN |
| B4 | A | cuda | — | — | NR | — | NOT_RUN |
| B4 | I | cuda | — | — | NR | — | NOT_RUN |
| S4 | D | cuda | — | — | NR | — | NOT_RUN |
| S4 | A | cuda | — | — | NR | — | NOT_RUN |
| S4 | I | cuda | — | — | NR | — | NOT_RUN |

CPU rows are local validation only, not same-host Colab measurements. CUDA rows NOT_RUN: no remote scientific artifact downloaded; therefore there is no per-row CUDA traceback or CPU/CUDA comparison. Required local-to-Colab CPU reproduction check is NOT_RUN.

Failure: initial float64 hard labels raised `ValueError: hard must be a length-m integer tensor on F's device`; corrected before the six local calls passed. Premature controller start interrupted after allocation; `runtime.json` records attempted source and command. `cleanup.json`: manual stop PASS, active assignments 1 → 0, rate 0.00/hr, balance 49.76 → 49.76. No second session, no source fixes.
