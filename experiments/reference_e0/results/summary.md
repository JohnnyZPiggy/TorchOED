# E0 real-wheel CPU observations

Verified original PyDOE 1.6.0 wheel; independent references are separated from source predictions.
Source-semantic differences and expected exceptions are valid findings, not blanket library failures.

| Case | Status |
|---|---|
| P0 | PASS_MATCH |
| P1 | FAIL_MISMATCH |
| P2-k1 | EXPECTED_EXCEPTION |
| P2-k2 | SOURCE_SEMANTIC_DIFFERENCE |
| P2-k3 | SOURCE_SEMANTIC_DIFFERENCE |
| P2-k4 | SOURCE_SEMANTIC_DIFFERENCE |
| P3-b3 | SOURCE_SEMANTIC_DIFFERENCE |
| P3-b5 | SOURCE_SEMANTIC_DIFFERENCE |
| P3-b8 | SOURCE_SEMANTIC_DIFFERENCE |
| P3-b30 | SOURCE_SEMANTIC_DIFFERENCE |
| P5-pow2-True | PASS_MATCH |
| P5-pow2-False | PASS_MATCH |
| P5-skip1 | EXPECTED_EXCEPTION |
| P4-L0-clenshaw_curtis | PASS_MATCH |
| P4-L0-chebyshev | PASS_MATCH |
| P4-L0-gauss_patterson | PASS_MATCH |
| P4-L1-clenshaw_curtis | PASS_MATCH |
| P4-L1-chebyshev | PASS_MATCH |
| P4-L1-gauss_patterson | PASS_MATCH |
| P4-L2-clenshaw_curtis | PASS_MATCH |
| P4-L2-chebyshev | PASS_MATCH |
| P4-L2-gauss_patterson | PASS_MATCH |
| P4-L3-clenshaw_curtis | PASS_MATCH |
| P4-L3-chebyshev | PASS_MATCH |
| P4-L3-gauss_patterson | PASS_MATCH |
| P4-L0-grid-type | SOURCE_SEMANTIC_DIFFERENCE |
| P4-L1-grid-type | SOURCE_SEMANTIC_DIFFERENCE |
| P4-L2-grid-type | SOURCE_SEMANTIC_DIFFERENCE |
| P4-L3-grid-type | SOURCE_SEMANTIC_DIFFERENCE |
| P4-dimension7 | PASS_MATCH |
| P4-dimension8 | PASS_MATCH |
| P6-easy-fedorov | PASS_MATCH |
| P6-easy-detmax | PASS_MATCH |
| P6-easy-simple_exchange_wynn_mitchell | PASS_MATCH |
| P6-easy-modified_fedorov | PASS_MATCH |
| P6-mentor22-fedorov | PASS_MATCH |
| P6-mentor22-detmax | PASS_MATCH |
| P6-mentor22-simple_exchange_wynn_mitchell | PASS_MATCH |
| P6-mentor22-modified_fedorov | PASS_MATCH |

Status totals: {"EXPECTED_EXCEPTION": 2, "FAIL_MISMATCH": 1, "PASS_MATCH": 25, "SOURCE_SEMANTIC_DIFFERENCE": 11}

P1 FAIL_MISMATCH: actual full quadratic rank 15/15, exact Gram determinant 18014398509481984.
The prior rank-12 oracle used D=ABC; the wheel uses a different cube. This corrects the hypothesis, not the package.
P2: k=1 IndexError; k=2/3/4 rows 6/10/15 and rank 2, versus documented 7/13/21 rows.
P3: all tested bit widths yield four distinct first-axis values and GF(2) rank 2.
P4: all three grid_type options give identical coordinates at levels 0–3; dimensions 7/8 both 705.
Independent weighted Smolyak is a different target: exact level-2 x²y² integral 1/9 with signed weights.
P5: unscrambled SciPy/torch and original wheel agree; skip=1 + base2 raises ValueError.
P6: eight real method/pool runs; distinct IDs, normalized scores and all legal swaps recorded.
Both small pools attain their exhaustive optimum; no general global-optimality claim.

GPU/performance: NOT_RUN. Only the listed inputs were exercised; no general parity claim.
