# B3: fixed CPU comparison

DoD: all 40 feasible TorchOED seed configurations complete as LOCAL_2_NUMERICAL; every finite TorchOED logdet matches B2 exactly (max absolute difference 0). No case median regresses more than 10%: the largest B3/B2 ratio is 1.02309 (grid4 n=15, +2.31%). The infeasible grid2 n=12 control is unchanged.

Observed B2-to-B3 median times: grid4 n=30, 5.46450 -> 0.263692 s (20.7x); random500 n=20, 174.723 -> 1.47361 s (118.6x). Saturated random500 n=10 remains 42.0114 s versus B2 41.9831 s: rho is below the guard, so it deliberately follows the original unpruned scan.

G1 target is not fully met. The comparator below is the fastest finite, distinct-ID PyDOE method with equal-or-lower observed logdet (sequential in these three cases); repeated-ID rows are excluded. This states the literal START metric rather than selecting a slower reference.

| Case | B2 s | B3 s | Fastest eligible PyDOE s | B3/PyDOE | Target <=3x |
|---|---|---|---|---|---|
| grid4 n=30 | 5.46450 | 0.263692 | 0.114107 | 2.31x | met |
| random500 n=10 | 41.9831 | 42.0114 | 0.214685 | 195.69x | not met |
| random500 n=20 | 174.723 | 1.47361 | 0.424265 | 3.47x | not met |

Pruning uses exactly rho>=1e-2 and w(c)w(d)>(1+tau/2)rho to retain survivors. Prefixes and their gather4 scores are tiled; all other outgoing pairs keep the previous full scan. Eligibility, near-tie ordering, verification and neighborhood-size coverage counts are unchanged. The certified D-eff column remains a floor against the true exact optimum, not achieved efficiency or a gap estimate; float64 values are not interval certificates.

Benchmark source: `3bfcebf1f8a85f12743d3fe38c9dea931b1c9213`. Cases, seeds, restarts, tolerances and parameters are unchanged; no bound for the saturated path or new algorithm was added.

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

Append `b3` to the command above. Baseline: B2; grid2 n=12 remains an unchanged infeasible control.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B2 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0384751 | 0.0383451 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.001848 | 0.0018754 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0037723 | 0.0035088 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.004401 | 0.0043653 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0045567 | 0.0044046 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0383876 | 0.0385087 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.0018747 | 0.0018893 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0026263 | 0.0027169 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.0028289 | 0.0026791 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.0038133 | 0.0040256 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B2 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.115013 | 0.115446 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.0103217 | 0.0102756 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0174552 | 0.0177567 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.018119 | 0.0176646 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.020353 | 0.0209501 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.0833021 | 0.10863 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0156276 | 0.0158435 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0286333 | 0.0290878 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0287734 | 0.028961 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.0278835 | 0.0279584 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B2 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 2.29066 | 2.23896 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0630147 | 0.0630224 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.113874 | 0.114446 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.115966 | 0.115226 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.126734 | 0.125923 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 0.263692 | 5.4645 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.114107 | 0.113581 | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.596746 | 0.600115 | OK /  repeats=0 | better (delta 0.0973242) / better CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.444118 | 0.440217 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.480693 | 0.474943 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B2 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 42.0114 | 41.9831 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.214685 | 0.213136 | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.09181 | 3.10105 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.1667 | 1.17691 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.34039 | 3.34017 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 1.47361 | 174.723 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.424265 | 0.422849 | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.3662 | 5.36898 | OK /  repeats=0 | equal (delta 3.55271e-15) / better CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.56993 | 1.57568 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.67097 | 3.66012 | OK /  repeats=0 | equal (delta 7.10543e-15) / better CPU time |
