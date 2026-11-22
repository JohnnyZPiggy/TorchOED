# W17 REML reuse check: BUILD

PyStatistics 7.0.0 (newest PyPI release at START) did not install on Windows / Python 3.14.6: `metadata-generation-failed`, Microsoft Visual C++ 14.0 or greater required (full verbatim output and `pip show` in pystatistics.txt). CPU REML fits and all PyStatistics comparisons are NOT_RUN: 0 of 81 requested quantities compared. Per the frozen rule this is BUILD, not a numerical failure claim about PyStatistics. No retry, GPU, adapter or production dependency.

Inputs: strip.csv committed first at cea37f6; NumPy default_rng(17), draw order block, block:A, block:B, residual, response rounded to four decimals. The generator is documented in run.py; reproduction reads only CSVs. One R dispatch at 58ec49e, run [45035411313](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45035411313), SUCCESS, script wall 1.7401 s. R 4.5.2, lme4 2.1-0, lmerTest 3.2-1 (current CRAN releases checked at START), nlme 3.1-168. Downloaded artifact frozen here; sessionInfo.txt includes the workflow's runtime package inventory because its final fresh-process sessionInfo overwrites the script's attached-package session. No R refits after results.

Exact reference: all CSV responses are Fraction; group means, stratum sums of squares, EMS, OLS coefficients and coefficient variance contributions are rational. SE square roots and REML log-likelihood logs are evaluated only at the final conversion to float. Oats components are E=MS_E, BV=(MS_BV−MS_E)/4, B=(MS_B−MS_BV)/12. Strip components are E=MS_E, BA=(MS_BA−MS_E)/4, BB=(MS_BB−MS_E)/3, B=(MS_B−MS_BA−MS_BB+MS_E)/12: each A strip contains four B subplots, each B strip three A subplots, and a block twelve. All estimates are positive. Stratum df: Oats 5/10/45; strip 3/6/9/18. Tiny integer 2×2×2 self-check gives exact components 24,14,4.

For coefficient variance contributions v_s in independent strata, Satterthwaite df=(sum v_s)^2/sum(v_s²/df_s). Omnibus Variety has df 10; individual treatment-coded Variety coefficients also contain subplot error and have df 30.23078024, so forcing every coefficient to 10 would not compare the requested summary(fit) table. Nitro and interactions have exact df 45. Strip A, B and A:B omnibus strata have df 6,9,18; treatment-coded A/B coefficients mix strata. REML logLik = −½[(n−p)(log(2π)+1)+log det(XᵀX)+sum df_s log(MS_s)]. Fixed-effect order matches R treatment contrasts; variance order in results is block, block:A, optional block:B, residual.

One shared relative metric abs(x−reference)/abs(reference) scores every pair/group. R vs exact max relative differences, recorded as-is:

| Case | Fixed | SE | Variance | logLik | df |
|---|---|---|---|---|---|
| Oats | 9.91e-13 | 3.62e-6 | 1.81e-5 | 1.10e-12 | 2.99e-5 |
| Strip | 5.13e-14 | 7.68e-6 | 1.90e-5 | 1.11e-11 | 3.00e-5 |

At 1e-8, 26/81 individual R quantities match (24 coefficients and two log-likelihoods); 4/10 groups match. The optimized R variances, SEs and df do not meet 1e-8 against exact EMS; no rerun or tolerance adjustment. PyStatistics-vs-exact and PyStatistics-vs-R are NOT_RUN in every group, and do not alter the frozen BUILD decision.

Reproduce: `PYTHONPATH=src python reports/w17-reml-check/run.py`. Uses stdlib only, saves results.json deterministically, never installs or refits. Required checks: exact EMS self-check and byte-identical regeneration (0.196 s on this host). Historical source, tests, pyproject, website and workflows untouched; tools/r_packages.txt only appends the two authorized pins. Nine evidence files; Python+R 126 lines (S120, below 1.5×180).

Deferred matrix-only texts (START forbids editing matrix in this PR):
- Adapters Status: ◐ PyStatistics 7.0.0: BUILD (#567, [w17-reml-check](../reports/w17-reml-check/)).
- REML Status: ◐ decision BUILD (#567); implementation W18.

Next: Architect reviews the recorded limitations and dispatches the matrix-only commit after merge; torch REML remains W18 work.
