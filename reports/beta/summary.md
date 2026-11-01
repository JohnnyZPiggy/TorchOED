# TorchOED 0.5.0 Beta: G1–G4

CPU float64; PyDOE 1.6.0 comparison. G1 uses B8 and G3 reuses B6 (one thread, fixed cases, seeds 0–4, ten restarts). G2 is remeasured with unchanged search code on the release source. No GPU or global-optimality claim.

| Goal / target | Measured result | Source file and source SHA | Status |
|---|---|---|---|
| G1: time ≤3× fastest eligible PyDOE | grid4 n=30: 2.3113×; random500 n=10: 2.2730×; n=20: **2.8418×** (1.207935 / 0.425065 s); paired regression gate PASS | [B8 JSON](../b8/results.json), `81168b3` | **MET** |
| G2: random2000 p=10 n=20, ten restarts, ≤120 s each; checked bound | Seeds 0/1/2: 8.046611 / 7.454587 / 10.242654 s; certified floors 0.843561018467; max independent difference 1.11e−16 | [G2 JSON](g2.json), `823749a` | **MET** |
| G3: certified floor on finite distinct-ID designs | 63 floors: 40 TorchOED + 23 PyDOE; 18 n/a: 9 infeasible-control errors, 8 singular, 1 repeated-ID result | [B6 JSON](../b6/results.json), `c6a86d1` | **MET** |
| G4: one-page consolidated Beta report and release preparation | This report, version bump and CHANGELOG; `v0.5.0` pre-release tagged on `9aae0f7` (before B8, so its notes still say G1 NOT MET) | This report and [v0.5.0 release](https://github.com/JohnnyZPiggy/TorchOED/releases/tag/v0.5.0) | **MET** |

G1 comparator: fastest finite, distinct-ID, equal-or-lower-logdet PyDOE row (sequential in these three cases). TorchOED achieves higher observed logdet but takes longer than sequential. Random500 n=20 improved from B5's 1.505780 s (B6: 1.330804 s) to 1.207935 s in B8 (#68), which meets the ≤3× target. B8 logdets equal B6 exactly, and B8 passed a same-session paired ≤1.10× regression gate against B6. Per-method comparisons: [B8 summary](../b8/summary.md).

G3 uses the Kiefer–Wolfowitz floor **D-eff ≥ p / (n max_j G_jj)**, G = F M⁻¹ Fᵀ, M = X_SᵀX_S. This is a certified lower bound, not achieved efficiency. On grid2 n=6 the floor is loose: 0.364 versus achieved efficiency 1 (exact determinant 256). Numerical local certificates do not prove global optimality.

G2 command: `python -B tools/bench_g2.py beta`; one timed call per seed, one thread, default search parameters. Timing includes existing diagnostic capture; independent rescoring is excluded. Search source is unchanged from B6; only package version metadata differs. No tag, release or PyPI publication is performed by this preparation PR.
