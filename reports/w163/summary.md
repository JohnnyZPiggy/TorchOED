# W163: workspace-budget probe

Source `3606ab175e8828d3c4d7fc935aab18ac32f991f1`; fixed B0 environment, one CPU thread, seeds0..4, restarts1. D3 calls/seed; A/I1 call/seed. Times are medians of per-seed medians. No source, tests, tools or defaults changed.

| Budget (elements) | Case | D s | A s | I s | A:D | I:D |
|---:|---|---:|---:|---:|---:|---:|
| 1,000,000 | grid2 n=6 | 0.0067705 | 0.0084948 | 0.0079963 | 1.2547 | 1.1811 |
| 1,000,000 | grid2 n=7 | 0.0063354 | 0.0065319 | 0.0073164 | 1.031 | 1.1548 |
| 1,000,000 | grid3 n=10 | 0.0148963 | 0.0173625 | 0.0202805 | 1.1656 | 1.3614 |
| 1,000,000 | grid3 n=20 | 0.0142952 | 0.0131803 | 0.014061 | 0.92201 | 0.98362 |
| 1,000,000 | grid4 n=15 | 0.0312467 | 0.122541 | 0.059005 | 3.9217 | 1.8884 |
| 1,000,000 | grid4 n=30 | 0.037535 | 0.120498 | 0.113236 | 3.2103 | 3.0168 |
| 1,000,000 | random500 n=10 | 0.145248 | 0.354636 | 0.349901 | 2.4416 | 2.409 |
| 1,000,000 | random500 n=20 | 0.219105 | 1.31806 | 1.04713 | 6.0157 | 4.7791 |
| 4,000,000 | grid2 n=6 | 0.0068274 | 0.0067825 | 0.007396 | 0.99342 | 1.0833 |
| 4,000,000 | grid2 n=7 | 0.0059819 | 0.0057739 | 0.0066464 | 0.96523 | 1.1111 |
| 4,000,000 | grid3 n=10 | 0.0125295 | 0.0167618 | 0.0147268 | 1.3378 | 1.1754 |
| 4,000,000 | grid3 n=20 | 0.0146037 | 0.0140835 | 0.0175295 | 0.96438 | 1.2003 |
| 4,000,000 | grid4 n=15 | 0.0340683 | 0.111642 | 0.0559241 | 3.277 | 1.6415 |
| 4,000,000 | grid4 n=30 | 0.0449522 | 0.112717 | 0.154164 | 2.5075 | 3.4295 |
| 4,000,000 | random500 n=10 | 0.152706 | 0.442541 | 0.378091 | 2.898 | 2.4759 |
| 4,000,000 | random500 n=20 | 0.198566 | 0.782738 | 0.768009 | 3.942 | 3.8678 |
| 16,000,000 | grid2 n=6 | 0.0062429 | 0.0067593 | 0.0068724 | 1.0827 | 1.1008 |
| 16,000,000 | grid2 n=7 | 0.005704 | 0.0054994 | 0.0062785 | 0.96413 | 1.1007 |
| 16,000,000 | grid3 n=10 | 0.013666 | 0.0146702 | 0.0155335 | 1.0735 | 1.1367 |
| 16,000,000 | grid3 n=20 | 0.0158098 | 0.0122305 | 0.0147806 | 0.7736 | 0.9349 |
| 16,000,000 | grid4 n=15 | 0.0248707 | 0.0981706 | 0.0494143 | 3.9472 | 1.9868 |
| 16,000,000 | grid4 n=30 | 0.028058 | 0.0831236 | 0.0695765 | 2.9626 | 2.4797 |
| 16,000,000 | random500 n=10 | 0.148393 | 0.447672 | 0.426059 | 3.0168 | 2.8712 |
| 16,000,000 | random500 n=20 | 0.213059 | 0.670025 | 0.545137 | 3.1448 | 2.5586 |

| Budget | Peak timed RSS MiB | Process peak MiB | Seed0 random500 n20 gather calls A / I | Both A/I ≤2× D |
|---:|---:|---:|---:|---:|
| 1,000,000 | 286.840 | 286.840 | 0 / 4 | 4/8 |
| 4,000,000 | 305.836 | 305.836 | 0 / 4 | 4/8 |
| 16,000,000 | 367.574 | 367.574 | 0 / 4 | 4/8 |

**Equality PASS:** 240/240 larger-budget vs1M ID/objective comparisons; 120 case/seed/criterion configurations at each budget, all360 outputs retained. D repeat outputs are also exact. Gathered rows A0/I14 and empty calls0 at every budget; separate untimed calls reproduce their timed outputs.

RSS is Windows PeakWorkingSetSize from GetProcessMemoryInfo in a fresh process per budget, including imports/features/allocator caches; timed peak is read before the count pass, process peak afterward. Baseline1M times are fresh, not a reproduction guarantee: exact time ratios vs w162 are retained in results.json. No rerolls or paired causal speedup claim; sequential budget order is fixed.

Command: `../.sae-w95-venv/Scripts/python.exe ../.sae-w163-control-20261109/run_probe.py`. Exact driver/adapter, every timing, IDs/objectives and checks: [results.json](results.json). C1 remains4/8; no default change proposed here. No new pytest required, GPU/held-out/R/Julia/remote sessions or queued #165 work.
