# W12 structured GPU evidence: BLOCKED

Refs #362. The authorized replacement session failed before science: controller `ALLOCATION_FAILED`, operation `new` status `TIMEOUT`, exit code -1. No artifact or observed device; Colab CPU and CUDA objectives and local-to-Colab reproduction remain NOT_RUN. No GPU speed or P2 Dev completion claim. No third session per Architect ruling.

First attempt interrupted: premature controller start preceded inspection of a float64 hard-label error; corrected to C[:,0].long() citing tests/test_blocked_design.py::test_split_plot_matches_frozen_exhaustive_optimum. Manual cleanup confirmed zero active assignments and unchanged 49.76 units. The Architect authorized one replacement at corrected head 6ee0a2967818a2d774898d0c28e2a79a7307b000.

Before replacement, all six local CPU calls at that exact SHA passed and their output was inspected. Local torch 2.15.0+cpu, one thread, float64, 3^4 grid, degree 2, n=24, default 10 restarts, seed 0. One B4 D warm-up. The following CPU rows are LOCAL ONLY; timings are whole-call / 10 and are not comparable to an unmeasured GPU.

| Case | Criterion | Device | Objective | Floor | IDs equal | Seconds/restart | Status |
|---|---|---|---|---|---|---|---|
| B4 | D | cpu | 32.95803107277601 | 0.8910042544570046 | CPU self only | 0.12262179999961517 | PASS |
| B4 | A | cpu | -1.6387385605965425 | 0.930435359688778 | CPU self only | 0.19710550000017973 | PASS |
| B4 | I | cpu | -0.5895762572514831 | 0.8836851617191598 | CPU self only | 0.1915285999995831 | PASS |
| S4 | D | cpu | 31.02750714031874 | 0.6747751931300998 | CPU self only | 0.2563787599996431 | PASS |
| S4 | A | cpu | -3.592435785724454 | 0.5082189996903211 | CPU self only | 0.21977416000008815 | PASS |
| S4 | I | cpu | -1.1799983713055593 | 0.4979594918513877 | CPU self only | 0.21577596999995877 | PASS |
| B4 | D | cuda | — | — | NR | — | NOT_RUN |
| B4 | A | cuda | — | — | NR | — | NOT_RUN |
| B4 | I | cuda | — | — | NR | — | NOT_RUN |
| S4 | D | cuda | — | — | NR | — | NOT_RUN |
| S4 | A | cuda | — | — | NR | — | NOT_RUN |
| S4 | I | cuda | — | — | NR | — | NOT_RUN |

Replacement controller log is preserved in runtime.json (137.917 seconds, pinned SAE 4129001); preflight PASS, allocation TIMEOUT, stop PASS, fresh usage PASS. cleanup.json confirms assignments 0 → 0, rate 0.00/hr, balance 49.76 → 49.76. There is no per-row CUDA traceback because the kernel was never reached. Required local-to-Colab CPU objective check: NOT_RUN. Local objectives reproduce the prior corrected local run exactly (6/6, Δ=0). No driver/src change, tuning, extra cases or third session. Next: Architect ruling on allocation failure.
