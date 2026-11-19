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

## Published examples

8 of 13 published examples match or are beaten on the shared metric (target ≥ 8 for the P2 exit).

| ID | Package and source | Shared metric | Verdict |
|---|---|---|---|
| alg-d | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | logdet(F.T @ F) | MATCH; see results.json |
| alg-a | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | trace(inverse(F.T @ F)) | NO MATCH; see results.json |
| alg-i | [AlgDesign 1.2.1.2](https://github.com/cran/AlgDesign/blob/1.2.1.2/man/Federov.Rd#L127-L131) | trace(inverse(F.T @ F) @ (F.T @ F / len(F))) | MATCH; see results.json |
| mix-sld32 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SLD.Rd#L30-L30) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-sld43 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SLD.Rd#L32-L32) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd3 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L29-L29) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd5 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L31-L31) | sorted row multiset max_abs <= 1e-12; row sums | MATCH; max_abs 0, [existing result](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w11-mixlat/summary.md) |
| mix-scd12 | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/SCD.Rd#L33-L33) | sorted row multiset max_abs <= 1e-12; row sums | NR; mix-scd12: there is no package called ‘mixexp’ |
| mix-xvert | [mixexp 1.2.7.1](https://github.com/cran/mixexp/blob/1.2.7.1/man/Xvert.Rd#L39-L40) | Hausdorff on dimen=0 vertex sets; vertex count | NR; mix-xvert: there is no package called ‘mixexp’ |
| fr-pb8 | [FrF2 2.3-5](https://github.com/cran/FrF2/blob/2.3-5/man/pb.Rd#L210-L210) | X.T @ X = nruns * I after mapping factor low/high levels to -1/+1 | NR; fr-pb8: there is no package called ‘FrF2’ |
| fr-pb40 | [FrF2 2.3-5](https://github.com/cran/FrF2/blob/2.3-5/man/pb.Rd#L211-L211) | X.T @ X = nruns * I after mapping factor low/high levels to -1/+1 | NR; fr-pb40: there is no package called ‘FrF2’ |
| rsm-ccd3-inscribed | [rsm 2.10.6](https://github.com/cran/rsm/blob/2.10.6/man/ccd.Rd#L120-L123) | dense D: logdet(F.T @ F) on coded full quadratic; row count | MATCH; see results.json |
| skpr-splitplot | [skpr 1.9.2](https://github.com/cran/skpr/blob/1.9.2/man/gen_design.Rd#L217-L230) | dense GLS logdet | MATCH; both 13.513636250491656, delta 0 ([w8-skpr](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/w8-skpr/summary.md)) |

New results: alg-d MATCH; alg-a NO MATCH (2.4240924092409237 vs reference 2.3, lower is better); alg-i MATCH; rsm-ccd3-inscribed MATCH. mix-scd12, mix-xvert, fr-pb8 and fr-pb40 NR (packages unavailable). Existing five MATCH results reused without rerun. No broader parity claim.

## Against R and held-out cases

**Against R** (re-scored with one shared metric): FrF2 fractional-factorial catalogue 22/22 and Plackett–Burman 3/3 identical; all rsm central-composite and Box–Behnken references matched; a blocked response-surface case reached a higher D-criterion than AlgDesign `optBlock` (log det 18.03 vs 16.78); the skpr split-plot documentation example matched exactly. Every external run records its exact R and package versions; references that could not be run are reported as such. See [`reports/`](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/README.md).

See the [full report index and held-out reports](https://github.com/JohnnyZPiggy/TorchOED/blob/main/reports/README.md).
