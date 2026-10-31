# B0: fixed CPU comparison

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | Median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.375901 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0019456 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.0035365 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.0042697 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.004346 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.342987 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.0018302 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.0027688 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.0027507 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.0038702 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | Median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 32.0395 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | 0.0102784 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | 0.0179087 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | 0.0175982 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | 0.0206365 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 36.0457 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.0157524 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.0286574 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.0290109 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.0282754 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | Median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|
| 15 | TorchOED | n/a | n/a | n/a | ValueError: dimension exceeds output element budget/int64 limit | reference |
| 15 | sequential | -inf (exact singular) | n/a | 0.0619376 | M must be positive definite (Cholesky failed) /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 15 | fedorov | -inf (exact singular) | n/a | 0.113385 | M must be positive definite (Cholesky failed) /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 15 | modified_fedorov | -inf (exact singular) | n/a | 0.113816 | M must be positive definite (Cholesky failed) /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 15 | detmax | -inf (exact singular) | n/a | 0.124824 | M must be positive definite (Cholesky failed) /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 30 | TorchOED | n/a | n/a | n/a | ValueError: dimension exceeds output element budget/int64 limit | reference |
| 30 | sequential | 39.2951 | 0.952902 | 0.111527 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 30 | fedorov | 39.9742 | 0.997033 | 0.610502 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 30 | modified_fedorov | 40.0188 | 1 | 0.438002 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 30 | detmax | 40.0188 | 1 | 0.474138 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |

## random500: m=500, p=10

| n | Method | logdet | D-eff | Median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|
| 10 | TorchOED | n/a | n/a | n/a | ValueError: dimension exceeds output element budget/int64 limit | reference |
| 10 | sequential | 7.91298 | 0.70037 | 0.214297 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 10 | fedorov | 11.4745 | 1 | 3.08751 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 10 | modified_fedorov | 11.4745 | 1 | 1.16678 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 10 | detmax | 11.4745 | 1 | 3.34438 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | TorchOED | n/a | n/a | n/a | ValueError: dimension exceeds output element budget/int64 limit | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.424471 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | fedorov | 19.2347 | 1 | 5.32792 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | modified_fedorov | 19.2127 | 0.997802 | 1.55455 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 3.60474 | OK /  repeats=0 | not comparable (error, repeats, or unavailable TorchOED) |
