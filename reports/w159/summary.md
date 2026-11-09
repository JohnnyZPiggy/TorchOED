# W159: base A/I counts and kernel profile

Source `0213928bdb96f5e7b9bda8be3ccacaeb686a7100`; seed 0, restarts=1, four B0 cases only. Environment unchanged since w155 (Python3.14.6, torch2.15.0+cpu, one CPU thread). No source/test/tool changes.

Command: `../.sae-w95-venv/Scripts/python.exe ../.sae-w159-control-20261109/capture_profile.py`. Exact adapter, outputs, per-scan counts/means and cProfile function data are in [profile.json](profile.json).

Counts are totals over all N₂ scans. Neighbourhood = C(free,2)·C(m−n,2) per scan; screened = screen input rows; kept = final mask rows; gathered = gather4 input rows (equal to kept here).

| Case | L | Scans | Neighbourhood | Screened | Kept = gathered | Screened / neighbourhood | Kept / screened |
|---|---|---:|---:|---:|---:|---:|---:|
| grid4 n=15 | A | 3 | 675,675 | 24,931 | 197 | 3.69% | 0.7902% |
| grid4 n=15 | I | 3 | 675,675 | 2,239 | 14 | 0.3314% | 0.6253% |
| grid4 n=30 | A | 2 | 1,109,250 | 107,175 | 2 | 9.662% | 0.001866% |
| grid4 n=30 | I | 1 | 554,625 | 55,640 | 0 | 10.03% | 0% |
| random500 n=10 | A | 2 | 10,782,450 | 61,168 | 32 | 0.5673% | 0.05231% |
| random500 n=10 | I | 3 | 16,173,675 | 33,772 | 31 | 0.2088% | 0.09179% |
| random500 n=20 | A | 1 | 21,842,400 | 122,979 | 0 | 0.563% | 0% |
| random500 n=20 | I | 4 | 87,369,600 | 1,477,207 | 14 | 1.691% | 0.0009477% |

Seconds below: original-function cProfile cumulative time, with unprofiled D three-call median beside it. Tile union excludes nested _pair_tiles fallback to avoid double counting.

| Case | L | D s | A/I wall s | N₂ s | Screen s | Gather4 s (calls) | Tiles s | N₂ remainder s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| grid4 n=15 | A | 0.02757 | 0.14006 | 0.12527 | 0.03658 | 0.01635 (54) | 0.05173 | 0.02062 |
| grid4 n=15 | I | 0.02757 | 0.11909 | 0.09577 | 0.02423 | 0.01092 (43) | 0.04411 | 0.01650 |
| grid4 n=30 | A | 0.04041 | 0.10932 | 0.08132 | 0.03228 | 0.00601 (22) | 0.02738 | 0.01565 |
| grid4 n=30 | I | 0.04041 | 0.06055 | 0.03686 | 0.01464 | 0.00271 (11) | 0.01253 | 0.00698 |
| random500 n=10 | A | 0.14457 | 0.34805 | 0.05768 | 0.02452 | 0.00504 (14) | 0.01798 | 0.01015 |
| random500 n=10 | I | 0.14457 | 0.30340 | 0.04529 | 0.01575 | 0.00352 (10) | 0.01778 | 0.00825 |
| random500 n=20 | A | 0.17277 | 0.82456 | 0.55980 | 0.22970 | 0.06539 (273) | 0.17932 | 0.08539 |
| random500 n=20 | I | 0.17277 | 4.23452 | 3.93638 | 1.70683 | 0.44211 (1688) | 1.16612 | 0.62132 |

Counts/oracles: **8/8 PASS** exact IDs/objectives vs same-SHA unprofiled B0 per-restart subset calls through the unchanged imported entry point; also **8/8** vs w155 and **8/8** seed-0 screened-count cross-checks vs w157. The full B0 main driver and other seeds/cases were not run.

Gather4 time includes empty calls (e.g. random500 n=20 A has zero gathered rows). N₂ includes counter wrappers; original kernel times exclude wrapper self time. Remainder includes setup, indexing/masking, native torch operations and instrumentation, not just Python. No overhead subtraction or speedup claim. This single-seed profile does not change C1 (4/8).

One setup failure: adapter filename profile.py shadowed stdlib profile; renamed before the successful run. No new tests requested. No GPU, held-out cases, R/Julia or remote sessions.
