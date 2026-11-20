# Benchmarks and validation

## Scale vs PyDOE

**Scale vs PyDOE 1.6.0** (D-optimal, full quadratic, random candidates; one Colab host with a Tesla T4; seconds per restart; [full report](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w5-crossover/run2-summary.md)):

| factors | candidates | PyDOE (fastest of 3) | TorchOED CPU | TorchOED GPU |
|---|---|---|---|---|
| 3 | 10³ | 4.5 s | **0.7 s** | 1.0 s |
| 5 | 10³ | 7.3 s | **0.8 s** | 1.7 s |
| 5 | 10⁴ | 199 s | 41 s | **9.2 s** |
| 5 | 10⁵ | timeout (> 30 min) | timeout | **188 s** |

Where both finished, TorchOED's design was as good or better (higher log-determinant) in 4 of 5 cells; at 3 factors and 10⁵ candidates it was 0.02 lower than PyDOE's `fedorov` (21.75 vs 21.77), reported as measured. Each TorchOED design also carries a certified efficiency floor of about 0.97.

**Space filling on one GPU host** (NVIDIA L4 and the same host's CPU, float64, one restart with `max_iter=20`, seconds; [full report](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-sfgpu/summary.md)): for 400 × 5 Latin hypercubes, maximin took 111.9 s on CPU vs 9.7 s on GPU, MaxPro 115.1 vs 7.3 s and uniform (CD2) 84.4 vs 4.6 s; at 50 runs every call took under 0.3 s on either device. Fast flexible filling of 50 points from 2¹⁷ candidates took 1.08 vs 0.32 s. CPU and GPU use different random generators, so the designs differ; all 22 designs passed the Latin-hypercube or constraint check.

## Published examples

14 of 16 published examples match or are beaten on the shared metric (P2 exit target ≥ 8 met; week 13 target ≥ 12 met).

| ID | Package and source | Shared metric | Verdict |
|---|---|---|---|
| alg-d | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | logdet(F.T @ F) | MATCH; see results.json |
| alg-a | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | trace(inverse(F.T @ F)) | NO MATCH; see results.json |
| alg-i | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | trace(inverse(F.T @ F) @ (F.T @ F / len(F))) | MATCH; see results.json |
| mix-sld32 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SLD.Rd#L30-L30) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-sld43 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SLD.Rd#L32-L32) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd3 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L29-L29) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd5 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L31-L31) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd12 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L33-L33) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 3.3e-16, [w13-published](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-published/summary.md) |
| mix-xvert | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/Xvert.Rd#L39-L40) | Hausdorff on dimen=0 vertex sets; vertex count | NO MATCH; Hausdorff 3.0e-8 > 1e-12 (R returns float32-rounded vertices), [w13-published](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-published/summary.md) |
| fr-pb8 | [FrF2 2.3-5](https://github.com/cran/FrF2/blob/2.3-5/man/pb.Rd#L210-L210) | X.T @ X = nruns * I after mapping factor low/high levels to -1/+1 | MATCH; Gram error 0, [w13-published](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-published/summary.md) |
| fr-pb40 | [FrF2 2.3-5](https://github.com/cran/FrF2/blob/2.3-5/man/pb.Rd#L211-L211) | X.T @ X = nruns * I after mapping factor low/high levels to -1/+1 | MATCH; Gram error 0, [w13-published](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-published/summary.md) |
| rsm-ccd3-inscribed | [rsm 2.10.6](https://github.com/cran/rsm/blob/2.10.6/man/ccd.Rd#L120-L123) | dense D: logdet(F.T @ F) on coded full quadratic; row count | MATCH; see results.json |
| skpr-splitplot | [skpr 1.9.2](https://github.com/cran/skpr/blob/1.9.2/man/gen_design.Rd#L217-L230) | dense GLS logdet | MATCH; both 13.513636250491656, delta 0 ([w8-skpr](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w8-skpr/summary.md)) |
| lhs-maximin43 | [lhs 1.3.0](https://github.com/cran/lhs/blob/1.3.0/man/maximinLHS.Rd#L74-L74) | phi_15 on rank midpoints (lower is better) | TorchOED better; 1.6452337368527556 vs 2.310300628135955, [w13-spacefill-ext](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-spacefill-ext/summary.md) |
| maxpro-104 | [MaxPro 4.1-2](https://github.com/cran/MaxPro/blob/4.1-2/man/MaxProLHD.Rd#L65-L65) | MaxPro psi on rank midpoints (lower is better) | TorchOED better; 13.227729408335211 vs 13.43815350925327, [w13-spacefill-ext](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-spacefill-ext/summary.md) |
| dice-cd2-402 | [DiceDesign 1.10](https://github.com/cran/DiceDesign/blob/1.10/man/discrepancyCriteria.Rd#L68-L68) | CD2 of the R design, abs diff <= 1e-12 | MATCH; delta 8.0e-15, [w13-spacefill-ext](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w13-spacefill-ext/summary.md) |

New results: alg-d MATCH; alg-a NO MATCH (2.4240924092409237 vs reference 2.3, lower is better); alg-i MATCH; rsm-ccd3-inscribed MATCH. mix-scd12, fr-pb8 and fr-pb40 MATCH and mix-xvert NO MATCH on the R oracle workflow (R 4.5.2, mixexp 1.2.7.1, FrF2 2.3-5, run 44613389076). Existing five MATCH results reused without rerun. No broader parity claim. lhs-maximin43 and maxpro-104 TorchOED better and dice-cd2-402 MATCH on R 4.5.2 (lhs 1.3.0, MaxPro 4.1-2, DiceDesign 1.10, run 44641039986).

## Against R and held-out cases

**Against R** (re-scored with one shared metric): FrF2 fractional-factorial catalogue 22/22 and Plackett–Burman 3/3 identical; all rsm central-composite and Box–Behnken references matched; a blocked response-surface case reached a higher D-criterion than AlgDesign `optBlock` (log det 18.03 vs 16.78); the skpr split-plot documentation example matched exactly. On 8 × 3 space-filling designs (R 4.5.2: lhs 1.3.0, MaxPro 4.1-2, DiceDesign 1.10), `space_filling_diagnostics` agreed with the R values to 1e-12 in 16/16 comparisons; TorchOED's maximin LHS had a lower φ₁₅ than `maximinLHS` (2.073 vs 3.278), while `MaxProLHD` found a lower MaxPro ψ than TorchOED (11.120 vs 12.246, default settings, seed 0). For a 10-point fast flexible filling design on the 511 feasible Sobol points of x₁ + x₂ ≤ 1 (R support 0.1.7), TorchOED had lower φ₁₅, MaxPro ψ and CD2 than `support::sp` (5.524 vs 5.646, 26.21 vs 133.56, 0.2295 vs 0.2775), while `sp` had the larger minimum distance and the lower energy criterion it minimises (0.1862 vs 0.1839, 0.4233 vs 0.4310). Every external run records its exact R and package versions; references that could not be run are reported as such. See [`reports/`](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/README.md).

See the [full report index and held-out reports](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/README.md).
