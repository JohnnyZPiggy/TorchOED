# W17 published least-squares comparison (#562)

Frozen verdict: **1 / 2 MATCH**. warpbreaks MATCH; ChemReact NO MATCH. CPU float64; no timing or GPU claim.

R workflow [45045947960](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45045947960) SUCCESS; source `46948f6552af978bc8ee09672378fcf449d31e10`. One dispatch only.
Versions: R 4.5.2, rsm 2.10.6, Python 3.14.6, torch 2.14.1+cpu. Runtime and sessionInfo are downloaded verbatim.

## ChemReact ordering discrepancy

The frozen dispatch explicitly requires X columns `1, BlockB2, x1, x2, x1:x2, x1^2, x2^2`. The specified R `lm` formula exports coefficients in the order `1, BlockB2, x1, x2, x1^2, x2^2, x1:x2` (the last three exported estimates are -1.30855544512464, -0.933442160912807, 0.125000000000009). R has moved the interaction after the square terms. Its last three ANOVA SS are 11.3597120398473, 6.43148101939501, 0.0625000000000092.
The task asserts these orders match; the downloaded numbers show otherwise. This report retains the explicitly requested X order and raw R order, records the resulting positional mismatch as-is, and does not change the formula, reorder outputs, rerun R, or change library code. It does not establish an implementation defect. Architect ruling is needed for a future same-term-order comparison.

## Per-group maximum differences

One shared scorer handles both cases. Relative = abs(Torch-R)/abs(R), with 1e-300 denominator for an exact zero; p-values use absolute difference. No reference values are zero in these scored groups. Each group must be <= 1e-8. Results include the worst index and values.

| Group | warpbreaks | ChemReact |
|---|---:|---:|
| coef | 1.2297855e-15 | 8.46753729 |
| se | 1.26149404e-15 | 0.358432576 |
| t | 2.71687076e-15 | 11.1441459 |
| p_t | 2.49800181e-16 | 0.169381852 |
| residuals | 1.47881707e-14 | 2.96903257e-12 |
| leverage | 7.49400542e-16 | 1.98198756e-15 |
| studentized | 1.74525884e-14 | 2.9685921e-12 |
| cooks | 3.30595175e-14 | 5.93508765e-12 |
| anova.df | 0 | 0 |
| anova.ss | 3.15329616e-15 | 101.903696 |
| anova.f | 2.12780922e-15 | 101.903696 |
| anova.p | 3.60822483e-16 | 0.169381804 |
| anova.ss_resid | 1.3478211e-16 | 4.91368896e-15 |
| lack_of_fit.df_lof | 0 | 0 |
| lack_of_fit.ss_lof | 1.70057874e-15 | 7.97555677e-15 |
| lack_of_fit.df_pe | 0 | 0 |
| lack_of_fit.ss_pe | 0 | 3.33066907e-15 |
| lack_of_fit.f | 1.06011434e-15 | 1.38068655e-14 |
| lack_of_fit.p | 2.22044605e-16 | 3.99680289e-15 |

Checks: df (residual, lack of fit, pure error) warpbreaks = (50, 2, 48), ChemReact = (7, 3, 4), on both Torch and R: PASS. Shared scoring: PASS. Byte-identical results reproduction: PASS. Numerical verdict: 1 MATCH, 1 NO MATCH; not a failing harness.

Reproduce: `PYTHONPATH=src python reports/w17-lm-published/run.py`. Reads committed CSV/R output only. The rsm summary is informational and unscored in r.json.

Proposed P2.5 least-squares Status: ◐ `fit_least_squares` (§123–§124, #554); published R comparison 1/2 MATCH, ChemReact dispatch-order discrepancy (#562, [w17-lm-published](../reports/w17-lm-published/)).
Proposed parity Fit Model Status: ◐ `fit_least_squares` (§123–§124, #554); warpbreaks MATCH, ChemReact positional NO MATCH (R term order differs from frozen X; #562).
No matrix update in this evidence lane. No source, tests, dependency pins, website, workflow, held-out cases or search changes.
