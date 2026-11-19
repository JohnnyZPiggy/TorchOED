# Four published examples rerun on R oracle

One dispatch: [44613389076](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44613389076), frozen input/script SHA `fd9c53f8089f4a9b2f8f62068b5f88235140a43e` (see runtime.json for exact SHA); R 4.5.2 on Ubuntu 24.04.5, mixexp 1.2.7.1, FrF2 2.3-5. No Colab. `sessionInfo.txt` is the workflow's fresh-process session; exact installed package versions are in `runtime.json`.
The four verbatim calls and parameters are frozen in read-only `../w12-published/examples.json` with source Rd URLs/lines. No calls or seeds were tuned. PB factors were mapped from low/high to -1/+1 before CSV export.

| Example | Shared metric, CPU float64 | Verdict |
|---|---|---|
| SCD(12) | Both 4095 x 12; sorted multiset max_abs 3.3306690738754696e-16; row-sum errors TorchOED 2.220446049250313e-16, R 1.9984014443252818e-15 | MATCH |
| Xvert | 5 vertices each; Hausdorff 2.9802321999117254e-08 > 1e-12 | NO MATCH |
| Xvert dimen=1 (separate) | 5 edge centroids each; Hausdorff 2.3841858043383013e-08 > 1e-12 | NO MATCH |
| pb(8) | Both 8 x 7; max_abs(X.T @ X - n I) = 0 | MATCH |
| pb(40) | Both 40 x 39; max_abs(X.T @ X - n I) = 0 | MATCH |

Published-example matches **8 -> 11** (three new MATCH, one NO MATCH, zero NR). Target >=12 is not reached by these four examples. Xvert returns coordinates at approximately single precision, as in w11-mixref; tolerance remains 1e-12.
`score.py` copies the frozen W12 ordered/hausdorff/orth scorers, applies each metric to both designs, and reports Xvert vertices and edge centroids separately. A local repeat reproduces results.json byte-for-byte. Scientific comparisons: 3 MATCH, 1 NO MATCH; the edge comparison is not counted as another published example.
Reproduce from repo root: `PYTHONPATH=src python reports/w13-published/score.py`. No src/tests/docs/website/pyproject changes. Ten report files; R/Python total 50 lines (S budget). No second dispatch, other examples, held-out cases or GPU run.
