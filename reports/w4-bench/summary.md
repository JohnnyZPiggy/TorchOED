# W4-BENCH: fixed CPU comparison

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B0 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0325213 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.0017947 | n/a | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0032523 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.004292 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0041534 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0334993 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.0018469 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0027592 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.0026358 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.003707 | n/a | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B0 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.06591 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.0101788 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0171665 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.0174028 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.0202519 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.0739499 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0154846 | n/a | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0282779 | n/a | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.028204 | n/a | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.0276913 | n/a | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B0 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 0.166026 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0606736 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.111305 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.112917 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.123454 | n/a | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 0.207837 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.110938 | n/a | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.596985 | n/a | OK /  repeats=0 | better (delta 0.0973242) / better CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.428157 | n/a | OK /  repeats=0 | better (delta 0.052747) / better CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.466078 | n/a | OK /  repeats=0 | better (delta 0.052747) / better CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B0 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 0.44376 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.207503 | n/a | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.1055 | n/a | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.18265 | n/a | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.35018 | n/a | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 1.0935 | n/a | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.410082 | n/a | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.23429 | n/a | OK /  repeats=0 | equal (delta 3.55271e-15) / better CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.52349 | n/a | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.5568 | n/a | OK /  repeats=0 | equal (delta 7.10543e-15) / better CPU time |

## Measured per-restart timing (Scientist rule)

Each row uses restarts=1, seeds 0..4: D has 3 calls/seed, A/I 1 call/seed. Values are medians of per-seed medians, not total time divided by 10. A/I timings are reported as-is; no A/I quality comparison.
10-restart D reproduction: {"matched": 40, "total": 40, "reference": "reports/w118/pair.json runs.head"}

| Case | D total (10) s | D (1) s | A (1) s | I (1) s | A:D | I:D | fastest PyDOE exchange s | D (1) / PyDOE |
|---|---|---|---|---|---|---|---|---|
| grid2 n=6 | 0.0325213 | 0.00603209995 | 0.00467000005 | 0.0050305 | 0.774191425 | 0.833955015 | 0.00325230008 | 1.85471814 |
| grid2 n=12 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| grid2 n=7 | 0.0334993 | 0.00568950002 | 0.00411900005 | 0.00486799993 | 0.723965205 | 0.855611199 | 0.00263579993 | 2.15854775 |
| grid3 n=10 | 0.0659099999 | 0.0115094 | 0.0318298 | 0.0223927 | 2.76554817 | 1.94560099 | 0.0171665 | 0.670456994 |
| grid3 n=20 | 0.0739499 | 0.0110523 | 0.0155575001 | 0.0164710999 | 1.40762557 | 1.49028708 | 0.0276913 | 0.399125356 |
| grid4 n=15 | 0.1660259 | 0.0244370999 | 0.9533126 | 0.5088305 | 39.0108729 | 20.8220493 | 0.1113046 | 0.219551572 |
| grid4 n=30 | 0.207837 | 0.0248421 | 2.3895336 | 2.4092214 | 96.188873 | 96.9813905 | 0.4281574 | 0.0580209521 |
| random500 n=10 | 0.4437598 | 0.1153214 | 20.8438129 | 29.895165 | 180.745403 | 259.233455 | 1.182652 | 0.0975108485 |
| random500 n=20 | 1.0935012 | 0.1562964 | 80.9171772 | 40.7951284 | 517.716193 | 261.011312 | 3.5567975 | 0.0439430133 |
