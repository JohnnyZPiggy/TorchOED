# Comparison with PyDOE

The following measured-result cells are copied verbatim from the [Beta report at source 6c9d63d](https://github.com/JohnnyZPiggy/TorchOED/blob/6c9d63d3b48b5afe6a3189364e19614d9897854d/reports/beta/summary.md). That report records each experiment's own source SHA, methodology and limitations. No new comparison is run for this page.

| Goal | Measured result |
|---|---|
| G1 | grid4 n=30: 2.3113×; random500 n=10: 2.2730×; n=20: **2.8418×** (1.207935 / 0.425065 s); paired regression gate PASS |
| G2 | Seeds 0/1/2: 8.046611 / 7.454587 / 10.242654 s; certified floors 0.843561018467; max independent difference 1.11e−16 |
| G3 | 63 floors: 40 TorchOED + 23 PyDOE; 18 n/a: 9 infeasible-control errors, 8 singular, 1 repeated-ID result |

The G1 comparator is the fastest finite, distinct-ID, equal-or-lower-logdet PyDOE row. TorchOED achieves higher observed logdet but takes longer than sequential in those cases. These are fixed CPU float64 experiments with PyDOE 1.6.0; they do not establish a general speed advantage, GPU performance or global optimality. Certified floors are lower bounds, not achieved efficiency.

The separate [W3 held-out results at source 6c9d63d](https://github.com/JohnnyZPiggy/TorchOED/blob/6c9d63d3b48b5afe6a3189364e19614d9897854d/reports/w3-heldout/summary.md) are reported as-is in the repository.

## Constructor mapping

| PyDOE function | TorchOED function | Differences |
| --- | --- | --- |
| `fullfact(levels)` | `full_factorial(levels, *, device=None, max_elements=1_000_000)` | Each entry of `levels` counts the levels of one factor. Both use zero-based level indices. PyDOE varies the first factor fastest and TorchOED varies the last factor fastest. |
| `ff2n(n_factors)` | `ff2n(k, *, device=None, dtype=torch.float64, max_elements=1_000_000)` | Both arguments count factors. Both code the levels as −1 and +1. |
| `fracfact(gen)` | `fractional_factorial(m, generators, *, dtype=torch.float64, device=None, max_elements=1_000_000)` | PyDOE takes a letter string such as `"a b ab"` and accepts signed terms. TorchOED takes the base-factor count `m` and one zero-based column list per unsigned product, such as `[[0, 1]]`. |
| `fracfact_by_res(n, res)` | `fractional_factorial_generators(k, runs)` | PyDOE takes a factor count and desired resolution. TorchOED has no by-resolution search. The nearest function takes factor and run counts and returns generators from the first entries of the frozen FrF2 catalogue (P2 gate §34). |
| `fold(H, columns=None)` | `fold_over(X, factor=None)` | PyDOE swaps the two levels in a list of selected columns. TorchOED negates one selected zero-based column of a signed design. Omit the selector to fold all columns in either API. |
| `pbdesign(n)` | `plackett_burman(runs, *, dtype=torch.float64, device=None)` | PyDOE takes a factor count and uses the next strictly higher multiple of four for the run count. TorchOED takes a run count and returns `runs - 1` columns coded −1 and +1. |
| `ccdesign(n, center=(4, 4), alpha="orthogonal", face="circumscribed")` | `central_composite(k, *, alpha="orthogonal", center=(4, 4), dtype=torch.float64, device=None, max_elements=1_000_000)` | Both take a factor count and two center counts, one per block. PyDOE selects face-centred coding with `face="faced"`. TorchOED uses `alpha="faces"`, or a positive numeric `alpha` for a specified axial distance. |
| `bbdesign(n, center=None)` | `box_behnken(k, *, center=3, dtype=torch.float64, device=None)` | Both take a factor count and a center-point count. PyDOE chooses the center count from the factor count when omitted. For TorchOED, [see the API for supported k](api.md#candidate-construction). |
| `lhs(n, samples=None, criterion=None, iterations=None, random_state=None, correlation_matrix=None, seed=None)` | `lhs(n_samples, n_factors, *, generator=None, device=None, dtype=torch.float64, mode="random", max_elements=1_000_000)` | PyDOE takes the factor count first and defaults `samples` to `n`. TorchOED requires the sample count first and the factor count second. Both sample in the unit interval. |
| `gsd(levels, reduction, n=1)` | not available | PyDOE takes the level counts, reduction factor and number of complementary designs. |

TorchOED does not claim identical outputs to PyDOE for these mappings.
