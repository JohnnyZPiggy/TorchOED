# W2 augmentation: paired default-path benchmark

Base `2dfc2e50e3c0da915aa6e5eb87cdf757eec0d08a`; timed head `07c085898a846b3ed3e2ffe82e66413f14363bcf`. One completed back-to-back pair of `python -B tools/bench_b0.py <PyDOE-1.6.0-wheel> w95-base|w95-head`, PYTHONPATH=src, one BLAS thread; no concurrent tests.
Native Windows 11 build 26200; AMD Ryzen 7 4800H; Python 3.14.6, torch 2.15.0+cpu, NumPy 2.5.4, SciPy 1.18.1, PyDOE 1.6.0. GPU and Colab not used.
40/40 feasible TorchOED rows have exactly identical IDs and logdet. grid2 n=12 remains infeasible. All eight head/base case-median ratios are <1.10 (PASS); these are repeat timings, not a speedup claim.

| Case | n | Base s | Head s | Head/base |
|---|---|---|---|---|
| grid2 | 6 | 0.041826700 | 0.042313200 | 1.011631327 |
| grid2 | 7 | 0.038945100 | 0.042635300 | 1.094753897 |
| grid3 | 10 | 0.087232300 | 0.078207300 | 0.896540617 |
| grid3 | 20 | 0.093281900 | 0.089613100 | 0.960669755 |
| grid4 | 15 | 0.212586800 | 0.214668700 | 1.009793176 |
| grid4 | 30 | 0.283807900 | 0.272122200 | 0.958825318 |
| random500 | 10 | 0.517738600 | 0.506841400 | 0.978952313 |
| random500 | 20 | 1.291688900 | 1.255007000 | 0.971601599 |
