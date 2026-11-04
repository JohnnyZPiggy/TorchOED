# Comparison with PyDOE

The following measured-result cells are copied verbatim from the [Beta report at source 6c9d63d](https://github.com/JohnnyZPiggy/TorchOED/blob/6c9d63d3b48b5afe6a3189364e19614d9897854d/reports/beta/summary.md). That report records each experiment's own source SHA, methodology and limitations. No new comparison is run for this page.

| Goal | Measured result |
|---|---|
| G1 | grid4 n=30: 2.3113×; random500 n=10: 2.2730×; n=20: **2.8418×** (1.207935 / 0.425065 s); paired regression gate PASS |
| G2 | Seeds 0/1/2: 8.046611 / 7.454587 / 10.242654 s; certified floors 0.843561018467; max independent difference 1.11e−16 |
| G3 | 63 floors: 40 TorchOED + 23 PyDOE; 18 n/a: 9 infeasible-control errors, 8 singular, 1 repeated-ID result |

The G1 comparator is the fastest finite, distinct-ID, equal-or-lower-logdet PyDOE row. TorchOED achieves higher observed logdet but takes longer than sequential in those cases. These are fixed CPU float64 experiments with PyDOE 1.6.0; they do not establish a general speed advantage, GPU performance or global optimality. Certified floors are lower bounds, not achieved efficiency.

The separate [W3 held-out results at source 6c9d63d](https://github.com/JohnnyZPiggy/TorchOED/blob/6c9d63d3b48b5afe6a3189364e19614d9897854d/reports/w3-heldout/summary.md) are reported as-is in the repository.
