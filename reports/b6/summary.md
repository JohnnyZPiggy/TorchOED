# B6: fixed CPU comparison

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

Append `b6` to the command above. Baseline: B5; grid2 n=12 remains an unchanged infeasible control.

## B6 result against B5

Source: `c6a86d1743963b753537c6b5715e88351d498b5a`. All 40 feasible TorchOED seed results preserve B5 logdet (maximum absolute difference 0); no case median regresses by more than 10% (maximum ratio 1.060155). The infeasible grid2 n=12 control remains unchanged.

| G1 case | B5 seconds | B6 seconds | Fastest eligible PyDOE seconds | B6 / PyDOE | <=3x target |
|---|---|---|---|---|---|
| grid4 n=30 | 0.257061 | 0.257904 | sequential 0.111866 | 2.3055x | MET |
| random500 n=10 | 0.490861 | 0.501039 | sequential 0.212561 | 2.3572x | MET |
| random500 n=20 | 1.505780 | 1.330804 | sequential 0.424786 | 3.1329x | NOT MET |

Random500 n=20 improves from 1.505780 to 1.330804 seconds, but remains above the 3x target. No parameters or extra cases were tuned. The other two G1 targets remain met. TorchOED returns higher logdet than sequential in these three cases, and takes longer; comparisons against each method are below. Certified efficiency floors are lower bounds, not proofs of global optimality. No G2 rerun.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B5 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0424134 | 0.0419896 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.0017978 | 0.0017939 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0036108 | 0.0036023 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.0044604 | 0.0044419 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0046715 | 0.0044424 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0388036 | 0.038048 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.0018408 | 0.0018831 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0027616 | 0.0026512 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.0026677 | 0.0026514 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.0036224 | 0.0038634 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B5 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.0782202 | 0.0802273 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.0106289 | 0.0102522 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0179436 | 0.0174779 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.0180589 | 0.0176886 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.0205851 | 0.020917 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.0904156 | 0.0872208 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0159188 | 0.015722 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0288691 | 0.0288822 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0287057 | 0.0290822 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.0279958 | 0.0281121 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B5 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 0.218448 | 0.206053 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0626405 | 0.06226 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.115558 | 0.114937 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.114194 | 0.114255 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.123917 | 0.124335 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 0.257904 | 0.257061 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.111866 | 0.111557 | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.601654 | 0.607861 | OK /  repeats=0 | better (delta 0.0973242) / better CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.440548 | 0.436998 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.474081 | 0.469167 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B5 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 0.501039 | 0.490861 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.212561 | 0.21512 | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.09909 | 3.09438 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.16987 | 1.17632 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.3563 | 3.34214 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 1.3308 | 1.50578 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.424786 | 0.427273 | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.37721 | 5.38537 | OK /  repeats=0 | equal (delta 3.55271e-15) / better CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.56678 | 1.57528 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.66258 | 3.65291 | OK /  repeats=0 | equal (delta 7.10543e-15) / better CPU time |
