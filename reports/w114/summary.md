# W114 amendment 2: trusted D verification

Binding amendment: Issue #114 comment 6673114837. Dense scoring reverted; helper replaces repeated public validation only for internally generated candidate verification and D restart feasibility. Public k2 entry unchanged; F validated once at D entry.

Base `bd1623f5c7491a23c892371bde6ae0c54dbf628e`; timed head `4951d259a547fcc03ebd3e30cdc809239e983311`. Environment: {"python": "3.14.6", "platform": "Windows-11-10.0.26200-SP0", "cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD", "torch": "2.15.0+cpu", "numpy": "2.5.4", "scipy": "1.18.1", "pydoe": "1.6.0", "threads": 1}.

Same-session local command: `python ../.sae-w114-control-20261106/run_pair.py`; bench_b0 plus returned-floor capture, three calls/config, seeds 0..4, ten restarts, one thread. Profile: separate cProfile d_optimal_design(grid2,n=7,seed=0), cumulative top 10 before/after.

| Case | base s | head s | head/base | head / fastest PyDOE exchange | <=3x |
|---|---|---|---|---|---|
| grid2 n=6 | 0.0448688 | 0.0384846 | 0.857714047 | 11.2373639 | False |
| grid2 n=7 | 0.0423892 | 0.0365965001 | 0.863344912 | 14.121199 | False |
| grid3 n=10 | 0.0811734 | 0.0685393 | 0.844356648 | 3.94727507 | False |
| grid3 n=20 | 0.0953609 | 0.0796782 | 0.835543707 | 2.80451379 | True |
| grid4 n=15 | 0.2226351 | 0.187301 | 0.841291423 | 1.64634891 | True |
| grid4 n=30 | 0.279763 | 0.2258777 | 0.807389469 | 0.510510457 | True |
| random500 n=10 | 0.5566577 | 0.5169304 | 0.928632443 | 0.440722627 | True |
| random500 n=20 | 1.2725878 | 1.2156359 | 0.955247174 | 0.772932777 | True |

Equality: 40/40 IDs/logdet/certificates/stops/returned dual floors exact. No-regression <=1.10: True. Time target 5/8 <=3x (baseline 4/8); full 8/8 target NOT MET. Remaining cases: grid2 n6/n7, grid3 n10. No tuning or repeated amendment-2 benchmark.

Singular PyDOE reference designs count for time as ruled; grid2 n12 is infeasible. No GPU/R/Julia/remote/held-out runs. Previous dense experiment remains in git history; these three files replace its evidence per the ruling.
