# B1: fixed CPU comparison

All seven feasible primary cases and the grid2 n=7 oracle completed under the unchanged defaults: 40/40 TorchOED seed configurations, each timed three times, all LOCAL_2_NUMERICAL. Per the [Architect ruling](https://github.com/JohnnyZPiggy/TorchOED/issues/43#issuecomment-6541609358), grid2 n=12 remains an infeasible control and is not a B1 failure. The case list is unchanged.

Against B0, all grid2/grid3 returned logdets match exactly in results.json (required tolerance 1e-9). TorchOED median time falls from 0.375901 to 0.038554 seconds for grid2 n=6 (9.75x), 0.342987 to 0.0402174 for n=7 (8.53x), 32.0395 to 0.115969 for grid3 n=10 (276x), and 36.0457 to 0.105878 for n=20 (340x). grid4/random500 previously failed the default output budget, so no B0 speed ratio exists for them.

Against PyDOE, TorchOED quality is equal on grid2 except better than sequential at n=6. It is better on grid3, grid4 and random500 n=10; the saturated grid3/grid4 PyDOE designs are exactly singular. At random500 n=20 it matches fedorov/detmax and beats sequential; modified_fedorov repeats one ID and is not comparable. TorchOED CPU time is worse than every comparable PyDOE row. These are observed design scores and end-to-end times, not global-optimality or kernel-speed claims.

The full rerun returned 72 configurations (216 timed calls); nine expected exceptions are confined to the infeasible control. Eight returned PyDOE designs have exact determinant zero, and the one repeated-ID result is retained. Benchmark source: `6d292d4e25f76dc0022406bedc0cd56c934e6fb2`; raw IDs, coordinates, scores, timings and environment are in results.json.

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

B1 command appends `b1` to the command above; only the output directory/report changes. B0 median seconds and B0/current time ratios are shown beside the rerun timings for both libraries. grid2 n=12 remains infeasible (12 distinct IDs from 9 candidates); it is retained unchanged.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | Median s | B0 median s (B0/current) | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.038554 | 0.375901 (9.75x) | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0017715 | 0.0019456 (1.1x) | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.0034654 | 0.0035365 (1.02x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.0044306 | 0.0042697 (0.964x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.0045469 | 0.004346 (0.956x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.0402174 | 0.342987 (8.53x) | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.0020024 | 0.0018302 (0.914x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.003072 | 0.0027688 (0.901x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.0028369 | 0.0027507 (0.97x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.00391 | 0.0038702 (0.99x) | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | Median s | B0 median s (B0/current) | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.115969 | 32.0395 (276x) | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | 0.0101685 | 0.0102784 (1.01x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | 0.0174683 | 0.0179087 (1.03x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | 0.0174449 | 0.0175982 (1.01x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | 0.0205236 | 0.0206365 (1.01x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.105878 | 36.0457 (340x) | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.0156688 | 0.0157524 (1.01x) | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.0287976 | 0.0286574 (0.995x) | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.0287316 | 0.0290109 (1.01x) | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.0278785 | 0.0282754 (1.01x) | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | Median s | B0 median s (B0/current) | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 2.30894 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | 0.0626598 | 0.0619376 (0.988x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | 0.112956 | 0.113385 (1x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | 0.113286 | 0.113816 (1x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | 0.124763 | 0.124824 (1x) | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 5.45109 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.111477 | 0.111527 (1x) | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.599894 | 0.610502 (1.02x) | OK /  repeats=0 | better (delta 0.0973242) / worse CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.432867 | 0.438002 (1.01x) | OK /  repeats=0 | better (delta 0.052747) / worse CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.466037 | 0.474138 (1.02x) | OK /  repeats=0 | better (delta 0.052747) / worse CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | Median s | B0 median s (B0/current) | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 41.3691 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.211478 | 0.214297 (1.01x) | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 3.05367 | 3.08751 (1.01x) | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 1.16059 | 1.16678 (1.01x) | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 3.31203 | 3.34438 (1.01x) | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 175.684 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.419344 | 0.424471 (1.01x) | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 5.32026 | 5.32792 (1x) | OK /  repeats=0 | equal (delta 3.55271e-15) / worse CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | 1.56067 | 1.55455 (0.996x) | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 3.61902 | 3.60474 (0.996x) | OK /  repeats=0 | equal (delta 7.10543e-15) / worse CPU time |
