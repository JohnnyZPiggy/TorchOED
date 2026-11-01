# B8: fixed CPU comparison

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

Append `b8` to the command above. Baseline: B6R; grid2 n=12 remains an unchanged infeasible control.

## Paired-run acceptance (Architect ruling #68)

**PASS**. Exactly one pre-registered pair on the same machine, back-to-back: B6R source `9aae0f74bf2d72c5c16618f52c05d9b1b1ceac82`, then B8 source `81168b339afcaf5991bec264727e170fcf783adc`. Baseline is [base.json](base.json). All 40 feasible B8 logdets equal the original B6 results exactly (max absolute difference 0). Every case median is below 1.10 times its paired baseline; maximum ratio 0.975263. Tables below show paired B6R and B8 medians.

| G1 case | B8 seconds | Fastest eligible PyDOE seconds | B8 / PyDOE | <=3x target |
|---|---|---|---|---|
| grid4 n=30 | 0.261486 | 0.113135 | 2.311279x | MET |
| random500 n=10 | 0.490854 | 0.215949 | 2.273006x | MET |
| random500 n=20 | 1.207935 | 0.425065 | 2.841768x | MET |

All three G1 targets are **MET** in this paired run. The fastest eligible PyDOE is sequential in each case (finite logdet, distinct IDs, equal-or-lower observed logdet). TorchOED returns higher observed logdet and takes longer than sequential; per-method comparisons below show ties or differences against the other methods. Certified efficiency floors do not prove global optimality. No tuning or further reruns.

The initial cross-session comparison failed the 10% gate on grid3 n=10 (+11.992%) and grid4 n=30 (+15.228%), and was reported BLOCKED. Its complete results remain in commit `970c90f`. The Architect authorized this single paired remeasurement because the original gate mixed session drift with the code effect; the threshold and cases did not change.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B6R median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0393745 | 0.0429024 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.0017547 | 0.0017621 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0034729 | 0.0036215 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.0043432 | 0.0044641 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0043473 | 0.0044363 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0374701 | 0.0384205 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.0019184 | 0.0021412 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0028112 | 0.0026994 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.002767 | 0.0027363 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.0037743 | 0.0039506 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B6R median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.0774883 | 0.0851004 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.0105091 | 0.0103232 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0177487 | 0.0175345 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.0180667 | 0.0176473 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.0210933 | 0.020784 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.0858769 | 0.0971751 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0159314 | 0.0159863 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0291829 | 0.0293992 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0294646 | 0.0290238 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.028301 | 0.0289743 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B6R median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 0.205167 | 0.22088 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0632356 | 0.0624115 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.115791 | 0.116398 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.115174 | 0.115236 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.12676 | 0.126069 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 0.261486 | 0.276112 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.113135 | 0.113937 | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.599275 | 0.614166 | OK /  repeats=0 | better (delta 0.0973242) / better CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.440248 | 0.450111 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.474755 | 0.480696 | OK /  repeats=0 | better (delta 0.052747) / better CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B6R median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 0.490854 | 0.524585 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.215949 | 0.221066 | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.10061 | 3.16243 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.18582 | 1.20344 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.36603 | 3.41821 | OK /  repeats=0 | better (delta 0.162129) / better CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 1.20794 | 1.41445 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.425065 | 0.439161 | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.39026 | 5.53187 | OK /  repeats=0 | equal (delta 3.55271e-15) / better CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.56846 | 1.59019 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.64332 | 3.69189 | OK /  repeats=0 | equal (delta 7.10543e-15) / better CPU time |
