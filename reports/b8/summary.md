# B8: fixed CPU comparison

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

Append `b8` to the command above. Baseline: B6; grid2 n=12 remains an unchanged infeasible control.

## Fixed-run acceptance result

Source `8f0b2ced165fc85e7680dcf32de34e924cdbf33e`. All 40 feasible TorchOED logdets match B6 exactly (max absolute difference 0). **DoD 3 regression gate FAILED**: two case medians exceed B6 by more than 10%. No parameters were tuned and this fixed run was not repeated to select a passing result. Architect ruling required.

| Case | B6 seconds | B8 seconds | B8 / B6 | <=10% regression |
|---|---|---|---|---|
| grid2 n=6 | 0.042413 | 0.044599 | 1.051529x | PASS |
| grid2 n=7 | 0.038804 | 0.036845 | 0.949536x | PASS |
| grid3 n=10 | 0.078220 | 0.087600 | 1.119920x | FAIL |
| grid3 n=20 | 0.090416 | 0.095775 | 1.059272x | PASS |
| grid4 n=15 | 0.218448 | 0.211796 | 0.969547x | PASS |
| grid4 n=30 | 0.257904 | 0.297177 | 1.152276x | FAIL |
| random500 n=10 | 0.501039 | 0.524098 | 1.046022x | PASS |
| random500 n=20 | 1.330804 | 1.219047 | 0.916023x | PASS |

G1 ratios versus fastest finite distinct-ID equal-or-lower-logdet PyDOE (sequential): grid4 n=30: 2.603805x (0.297177/0.114132s); random500 n=10: 2.416479x (0.524098/0.216885s); random500 n=20: 2.849132x (1.219047/0.427866s). All three speed targets are MET in this run, but the regression gate fails. TorchOED has higher logdet and takes longer than sequential in these cases. Certified efficiency floors do not establish global optimality.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff â‰¥ | Median s | B6 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0445989 | 0.0424134 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.0019504 | 0.0017978 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0036378 | 0.0036108 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.0044615 | 0.0044604 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0046695 | 0.0046715 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0368454 | 0.0388036 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.001945 | 0.0018408 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0027766 | 0.0027616 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.0027299 | 0.0026677 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.003919 | 0.0036224 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff â‰¥ | Median s | B6 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.0876004 | 0.0782202 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.010488 | 0.0106289 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0178003 | 0.0179436 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.0179111 | 0.0180589 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.0209166 | 0.0205851 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.0957747 | 0.0904156 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0161139 | 0.0159188 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0291586 | 0.0288691 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.029646 | 0.0287057 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.0286219 | 0.0279958 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff â‰¥ | Median s | B6 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 0.211796 | 0.218448 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0636056 | 0.0626405 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.11795 | 0.115558 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.115761 | 0.114194 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.12557 | 0.123917 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 0.297177 | 0.257904 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.114132 | 0.111866 | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.606275 | 0.601654 | OK /  repeats=0 | better (delta 0.0973242) / better CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.441771 | 0.440548 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.476462 | 0.474081 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff â‰¥ | Median s | B6 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 0.524098 | 0.501039 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.216885 | 0.212561 | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.15988 | 3.09909 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.19374 | 1.16987 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.45064 | 3.3563 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 1.21905 | 1.3308 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.427866 | 0.424786 | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.39618 | 5.37721 | OK /  repeats=0 | equal (delta 3.55271e-15) / better CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.57418 | 1.56678 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.65307 | 3.66258 | OK /  repeats=0 | equal (delta 7.10543e-15) / better CPU time |
