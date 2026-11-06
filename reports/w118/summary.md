# W118 lazy candidate rows and combined host transfer

Base `6232e43576a6fe8ba0b138163b97984d66b17bac`; timed head `ad1155b9f5957ab0657441fc4fe21c28fdf6cbf5`. Environment: {"python": "3.14.6", "platform": "Windows-11-10.0.26200-SP0", "cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD", "torch": "2.15.0+cpu", "numpy": "2.5.4", "scipy": "1.18.1", "pydoe": "1.6.0", "threads": 1}.

One driver run was aborted for a wrong head path before producing results; its files were quarantined, not deleted or read (Architect rulings 6676759225 / 6677590375).

Single valid same-session local pair: `python ../.sae-w118-control2-20261106/run_pair.py`, base/head folder SHAs verified first, fresh reports/w118-pair2 outputs. bench_b0 plus returned-floor capture, 3 calls/config, seeds0..4, ten restarts, one thread. Profile: grid3 n10 seed0 separate cProfile, cumulative top10 before/after.

| Case | base s | head s | head/base | head / fastest PyDOE exchange | <=3x |
|---|---|---|---|---|---|
| grid2 n=6 | 0.0400816 | 0.037368 | 0.932298113 | 10.7943843 | False |
| grid2 n=7 | 0.0377878 | 0.0341894 | 0.9047735 | 12.8040598 | False |
| grid3 n=10 | 0.0732231 | 0.0680298 | 0.92907566 | 3.82439116 | False |
| grid3 n=20 | 0.0867022 | 0.0795949 | 0.918026302 | 2.84851446 | True |
| grid4 n=15 | 0.1906838 | 0.1697726 | 0.890335729 | 1.47522773 | True |
| grid4 n=30 | 0.2392343 | 0.2131485 | 0.890961287 | 0.483585651 | True |
| random500 n=10 | 0.5315155 | 0.4795584 | 0.902247253 | 0.404495383 | True |
| random500 n=20 | 1.2697099 | 1.1449072 | 0.901707705 | 0.72660135 | True |

Exact IDs/logdet/certificates/stops/returned floors: 40/40. No-regression gate <=1.10: PASS all eight. Time count: 5/8 <=3x (baseline 5/8).

No defaults, tolerances, target or bench protocol changed. Raw results reported as-is; no valid-pair rerun or tuning. Singular PyDOE rows count for time as ruled; grid2 n12 stays infeasible. No held-out/GPU/R/Julia/remote runs.
