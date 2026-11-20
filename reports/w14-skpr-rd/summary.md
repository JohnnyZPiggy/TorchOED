# Published skpr 1.9.2 power examples

One R oracle run [44697685526](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44697685526), frozen script source `3cc00b81a2da8076d31578f9ddd8c623ac3b9f29`; R 4.5.2, skpr 1.9.2. sessionInfo.txt and runtime.json are verbatim artifacts.

Source: [skpr 1.9.2 man/eval_design.Rd](https://github.com/cran/skpr/blob/1.9.2/man/eval_design.Rd), simple generation lines 134–139 and default eval_design(optdesign) lines 148–150; coffee generation and default eval_design(designcoffee) lines 160–168. Both use default alpha=.05/effectsize=2. Each gen_design has set.seed(20261120) immediately before it; generated designs are frozen in simple.csv and coffee.csv (repeated runs retained).

| Example | Runs | max absolute power delta | Verdict at frozen 1e-8 |
|---|---|---|---|
| simple | 11 | 4.5420656125116921e-10 | MATCH |
| coffee | 29 | 3.3691971434990364e-10 | MATCH |

Continuous coefficients use effect=1. Categorical factors use effects_coding with alphabetical R levels and the anticoef vector captured from skpr's output: coffee [1,1,1,-1,1,-1,1]; joint size/type powers use groups [[2,3],[4,5,6]]. Per the dispatch, categorical comparisons are joint effect powers, not individual contrast powers. All raw parameter powers (including intercept) remain verbatim in reference.json. The returned R model_matrix is the orthonormal correlation/diagnostic matrix (eval_design.R overwrites the earlier sum-coded attribute), so it is not used to reconstruct categorical power features.

10 comparisons PASS, 0 FAIL. Both examples count as published matches; published count 14 → 16. No index/matrix edits are authorised in this lane. Earlier B3 1e-10 NO MATCH evidence remains unchanged.

Reproduce: `python reports/w14-skpr-rd/run.py` (CPU float64). No searches, held-out, other reference calls, GPU or production edits.
