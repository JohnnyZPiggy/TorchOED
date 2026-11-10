# Final W4 C1 paired result (Refs #168)

Sources: baseline `39b8c56cf14866fa347accc4340c9d3f77aac03c`; final main `6e20954b2392003e089a54bc5e47ba336ab34bff`.
First complete paired run. Alternate baseline/head for each seed 0..4, separate processes importing each source own src and unchanged bench_b0 pools/features. Final bench_b0 per-restart calls: restarts=1, D three repeats, A/I one; median over five per-seed medians. No ten-restart or PyDOE runs (Not required). Baseline bench_b0 predates per_restart, so identical final per-restart loop supplied in scratch adapter; tools unchanged. Exact equality includes binary float hex for objectives and D bound. Infeasible grid2 n12 errors compared separately.
Environment: {'python': '3.14.6', 'platform': 'Windows-11-10.0.26200-SP0', 'cpu': 'AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD', 'torch': '2.15.0+cpu', 'numpy': '2.5.4', 'scipy': '1.18.1', 'pydoe': '1.6.0', 'threads': 1}
Exact checks: 135/135 pass (120 feasible designs, 15 expected infeasible errors); no new pytest (evidence only).
Final C1: **5/8** cases with both A:D and I:D <=2; baseline 3/8. C1 not met; reported as-is.

| Case | Base D s | Base A s | Base I s | Base A:D | Base I:D | Final D s | Final A s | Final I s | Final A:D | Final I:D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid2 n6 | 0.00537070003 | 0.00475960004 | 0.00543909997 | 0.886215951 | 1.01273576 | 0.00565860001 | 0.0073556999 | 0.00558519992 | 1.29991515 | 0.987028577 |
| grid2 n7 | 0.00558469992 | 0.00429810002 | 0.00524949993 | 0.769620585 | 0.939978872 | 0.00575760007 | 0.00591230008 | 0.0052461999 | 1.02686884 | 0.91117824 |
| grid3 n10 | 0.0113335 | 0.0350235 | 0.0223995 | 3.09026337 | 1.9763974 | 0.0119895 | 0.0116651 | 0.0105164 | 0.97294299 | 0.877134156 |
| grid3 n20 | 0.0110127 | 0.0174526001 | 0.0175074 | 1.58477031 | 1.58974638 | 0.0112893999 | 0.00880449999 | 0.00929899991 | 0.779890875 | 0.823693019 |
| grid4 n15 | 0.0256916999 | 0.9919485 | 0.516186 | 38.6096873 | 20.0915471 | 0.0257424 | 0.0884188 | 0.0451557999 | 3.43475356 | 1.75414103 |
| grid4 n30 | 0.0247435999 | 2.4246375 | 2.3885868 | 97.9904907 | 96.53352 | 0.0266542 | 0.0893570001 | 0.0770888 | 3.35245477 | 2.89218209 |
| random500 n10 | 0.1264317 | 20.5123803 | 30.4621666 | 162.240801 | 240.937728 | 0.1174492 | 0.1176114 | 0.1358063 | 1.00138102 | 1.15629821 |
| random500 n20 | 0.1717923 | 81.4382899 | 41.5777377 | 474.050873 | 242.02329 | 0.1694098 | 0.8821839 | 0.7109333 | 5.20739591 | 4.19652995 |

No src/tests/tools changes; no tuning, held-out, GPU, R/Julia, release or remote sessions. Drivers and all raw per-seed times/outputs retained in results.json.
