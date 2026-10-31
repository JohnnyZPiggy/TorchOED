# B2: fixed CPU comparison

Search is unchanged from B1: all 72 returned designs have identical IDs, and every finite logdet matches exactly (max absolute difference 0). All 40 feasible TorchOED seed configurations complete with LOCAL_2_NUMERICAL. B1/current timings below are repeat measurements of the same search, not a speedup claim; CPU time remains worse than PyDOE in every comparable row.

The new column contains 63 certified lower bounds: 40 TorchOED and 23 PyDOE designs. Eight exactly singular PyDOE designs and the one repeated-ID design are n/a. The nine exceptions belong only to the unchanged infeasible grid2 n=12 control. All 81 configurations and 216 timed returns are retained in results.json.

For TorchOED, grid4 n=15 lower bounds range from 0.419355 to 0.452231; n=30 gives 0.835730. random500 n=10/20 gives 0.522260/0.856244. These bounds are against the true exact optimum, not the best found design, and are not achieved efficiencies. They can be loose: grid2 n=6 has exact achieved efficiency 1 but certified lower bound 0.363636. No tightening or global-optimality claim is made.

Benchmark source: `327ed8d040159c0f55a4678920f454358adcaaff`. Only the two new return fields, benchmark rescoring/reporting and explicitly authorized redundant-check cleanup changed; cases, seeds, restarts and search parameters are unchanged.

Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.
Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.
TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.
Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.

Append `b2` to the command above. Baseline: B1; grid2 n=12 remains an unchanged infeasible control.

B2 search/timing protocol is unchanged from B1; timings are repeat measurements, not a speedup claim. Certified D-eff >= is p/(n max_j G_jj), a lower bound against the true exact optimum, not achieved efficiency. It is recomputed from each finite, distinct-ID design using shared float64 features and tiled whitening, outside timing. Singular or repeated-ID rows are n/a. The theorem is exact-arithmetic; float64 evaluations are not interval certificates.

## grid2: m=9, p=6; n=12 exceeds candidate count; exact optima: n=6 det=256, n=7 det=960

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B1 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 6 | TorchOED | 5.54518/5.54518/5.54518 | 1/1/1 | 0.363636/0.363636/0.363636 | 0.0383451 | 0.038554 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 6 | sequential | 2.77259 | 0.629961 | 0.0169492 | 0.0018754 | 0.0017715 | OK /  repeats=0 / exact=0/1 | better (delta 2.77259) / worse CPU time |
| 6 | fedorov | 5.54518 | 1 | 0.363636 | 0.0035088 | 0.0034654 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | modified_fedorov | 5.54518 | 1 | 0.363636 | 0.0043653 | 0.0044306 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 6 | detmax | 5.54518 | 1 | 0.363636 | 0.0044046 | 0.0045469 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 12 | TorchOED | n/a | n/a | n/a | n/a | n/a | ValueError: n must be an int with m>=n>=p | reference |
| 12 | sequential | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | modified_fedorov | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 12 | detmax | n/a | n/a | n/a | n/a | n/a | IndexError: index 0 is out of bounds for axis 0 with size 0 | not comparable (error, repeats, or unavailable TorchOED) |
| 7 | TorchOED | 6.86693/6.86693/6.86693 | 1/1/1 | 0.612245/0.612245/0.612245 | 0.0385087 | 0.0402174 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 / exact=5/5 | reference |
| 7 | sequential | 6.86693 | 1 | 0.612245 | 0.0018893 | 0.0020024 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | fedorov | 6.86693 | 1 | 0.612245 | 0.0027169 | 0.003072 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | modified_fedorov | 6.86693 | 1 | 0.612245 | 0.0026791 | 0.0028369 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |
| 7 | detmax | 6.86693 | 1 | 0.612245 | 0.0040256 | 0.00391 | OK /  repeats=0 / exact=1/1 | equal (delta 0) / worse CPU time |

## grid3: m=27, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B1 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 14.0985/14.0985/14.0985 | 1/1/1 | 0.290323/0.290323/0.363636 | 0.115446 | 0.115969 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | -inf (exact singular) | 0 | n/a | 0.0102756 | 0.0101685 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | fedorov | -inf (exact singular) | 0 | n/a | 0.0177567 | 0.0174683 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.0176646 | 0.0174449 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 10 | detmax | -inf (exact singular) | 0 | n/a | 0.0209501 | 0.0205236 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 20 | TorchOED | 22.2586/22.2586/22.2586 | 1/1/1 | 0.66129/0.66129/0.66129 | 0.10863 | 0.105878 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 21.9859 | 0.973091 | 0.617454 | 0.0158435 | 0.0156688 | OK /  repeats=0 | better (delta 0.272778) / worse CPU time |
| 20 | fedorov | 22.1201 | 0.986238 | 0.610169 | 0.0290878 | 0.0287976 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | modified_fedorov | 22.1201 | 0.986238 | 0.610169 | 0.028961 | 0.0287316 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |
| 20 | detmax | 22.1201 | 0.986238 | 0.610169 | 0.0279584 | 0.0278785 | OK /  repeats=0 | better (delta 0.138573) / worse CPU time |

## grid4: m=81, p=15

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B1 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 15 | TorchOED | 27.7817/27.7874/27.7874 | 0.999621/1/1 | 0.419355/0.452231/0.452231 | 2.23896 | 2.30894 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 15 | sequential | -inf (exact singular) | 0 | n/a | 0.0630224 | 0.0626598 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | fedorov | -inf (exact singular) | 0 | n/a | 0.114446 | 0.112956 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | modified_fedorov | -inf (exact singular) | 0 | n/a | 0.115226 | 0.113286 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 15 | detmax | -inf (exact singular) | 0 | n/a | 0.125923 | 0.124763 | M must be positive definite (Cholesky failed) /  repeats=0 | better: finite vs -inf / worse CPU time |
| 30 | TorchOED | 40.0715/40.0715/40.0715 | 1/1/1 | 0.83573/0.83573/0.83573 | 5.4645 | 5.45109 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 30 | sequential | 39.2951 | 0.949557 | 0.59935 | 0.113581 | 0.111477 | OK /  repeats=0 | better (delta 0.776396) / worse CPU time |
| 30 | fedorov | 39.9742 | 0.993533 | 0.773501 | 0.600115 | 0.599894 | OK /  repeats=0 | better (delta 0.0973242) / worse CPU time |
| 30 | modified_fedorov | 40.0188 | 0.99649 | 0.815952 | 0.440217 | 0.432867 | OK /  repeats=0 | better (delta 0.052747) / worse CPU time |
| 30 | detmax | 40.0188 | 0.99649 | 0.815952 | 0.474943 | 0.466037 | OK /  repeats=0 | better (delta 0.052747) / worse CPU time |

## random500: m=500, p=10

| n | Method | logdet | D-eff | certified D-eff ≥ | Median s | B1 median s | Status / repeats / exact optimum | TorchOED quality / time versus row |
|---|---|---|---|---|---|---|---|---|
| 10 | TorchOED | 11.6366/11.6366/11.6366 | 1/1/1 | 0.52226/0.52226/0.52226 | 41.9831 | 41.3691 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 10 | sequential | 7.91298 | 0.689106 | 0.153141 | 0.213136 | 0.211478 | OK /  repeats=0 | better (delta 3.7236) / worse CPU time |
| 10 | fedorov | 11.4745 | 0.983918 | 0.555332 | 3.10105 | 3.05367 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | modified_fedorov | 11.4745 | 0.983918 | 0.555332 | 1.17691 | 1.16059 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 10 | detmax | 11.4745 | 0.983918 | 0.555332 | 3.34017 | 3.31203 | OK /  repeats=0 | better (delta 0.162129) / worse CPU time |
| 20 | TorchOED | 19.2347/19.2347/19.2347 | 1/1/1 | 0.856244/0.856244/0.856244 | 174.723 | 175.684 | LOCAL_2_NUMERICAL / LOCAL_2_NUMERICAL repeats=0 | reference |
| 20 | sequential | 18.5044 | 0.929577 | 0.647859 | 0.422849 | 0.419344 | OK /  repeats=0 | better (delta 0.730251) / worse CPU time |
| 20 | fedorov | 19.2347 | 1 | 0.856244 | 5.36898 | 5.32026 | OK /  repeats=0 | equal (delta 3.55271e-15) / worse CPU time |
| 20 | modified_fedorov | 19.2127 | 0.997802 | n/a | 1.57568 | 1.56067 | OK /  repeats=1 | not comparable (error, repeats, or unavailable TorchOED) |
| 20 | detmax | 19.2347 | 1 | 0.856244 | 3.66012 | 3.61902 | OK /  repeats=0 | equal (delta 7.10543e-15) / worse CPU time |
