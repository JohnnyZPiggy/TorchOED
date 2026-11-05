# W114 dense D scoring: measured as-is

Base `bd1623f5c7491a23c892371bde6ae0c54dbf628e`; head `35270c449283ae6a9a63f0f0a6b79e16bd76cc0f`. Same local dev environment per Architect amendment 6669829366: {"python": "3.14.6", "platform": "Windows-11-10.0.26200-SP0", "cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD", "torch": "2.15.0+cpu", "numpy": "2.5.4", "scipy": "1.18.1", "pydoe": "1.6.0", "threads": 1}.

Command: task scratch `python run_pair.py`, invoking `python -B -c <bench_b0 wrapper> <PyDOE 1.6.0 wheel>` for each checkout. Wrapper only captures the returned floor outside search; bench_b0 keeps three timed calls/config, seeds 0..4, ten restarts and one thread. Profiles: one separate cProfile call of d_optimal_design(grid2,n=7,seed=0), cumulative top 10. No held-out runs.

| Case | base s | head s | head/base | fastest head PyDOE exchange s | head/PyDOE | <=3x |
|---|---|---|---|---|---|---|
| grid2 n=6 | 0.0471797 | 0.0418921 | 0.887926375 | 0.00367420004 | 11.4016928 | False |
| grid2 n=7 | 0.0409678 | 0.0455123 | 1.11092858 | 0.00267369999 | 17.0222165 | False |
| grid3 n=10 | 0.0815872001 | 0.0830649999 | 1.01811313 | 0.0176609 | 4.70332768 | False |
| grid3 n=20 | 0.0927198 | 0.1169985 | 1.26185022 | 0.027916 | 4.19109113 | False |
| grid4 n=15 | 0.219133 | 0.2227583 | 1.01654383 | 0.1144391 | 1.94652265 | True |
| grid4 n=30 | 0.2689192 | 0.2715026 | 1.0096066 | 0.4376109 | 0.620420104 | True |
| random500 n=10 | 0.578362 | 0.5645201 | 0.976067065 | 1.1789752 | 0.47882271 | True |
| random500 n=20 | 1.2779825 | 1.2769768 | 0.999213057 | 1.5812762 | 0.80756088 | True |

Output equality: 40/40 IDs/logdet/certificates/stops/returned §4b floors exact. Target: 4/8 <=3x (baseline 4/8), NOT MET. The four failing cases remain grid2 n6/n7 and grid3 n10/n20. No restarts, max_iter, tolerances or defaults changed; no tuning or benchmark rerun. Architect rules on next step.

Singular PyDOE reference designs still count for time per the amendment. grid2 n12 remains the infeasible control. GPU/R/Julia/remote allocation NOT_RUN.
