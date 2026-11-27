# W19 residual-bootstrap R references

One dispatch: [46001372942](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/46001372942), source e20078a32267289d43a285b8563a99d00f496410. R 4.5.2, rsm 2.10.6, jsonlite 2.0.0; exact installed versions in runtime.json and sessionInfo.txt. The workflow captures sessionInfo in a fresh R process; the script uses rsm and base stats only.

B = 999, modified residuals e/sqrt(1-h), centred; percentile quantile type 7. JSON digits = NA. Indices are n x B, 1-based; replicates are B x p.

| Case | n x p | Seed | Column order | In-R max consistency error |
|---|---|---|---|---|
| stackloss | 21 x 4 | 20261127 | (Intercept), Air.Flow, Water.Temp, Acid.Conc. | 2.58149057685841e-12 PASS |
| ChemReact | 14 x 7 | 20261128 | (Intercept), BlockB2, x1, x2, I(x1^2), I(x2^2), x1:x2 | 5.6843418860808e-14 PASS |

Consistency is max over every replicate and coefficient of abs(replicates[b,] - (coef(fit) + solve(crossprod(X), crossprod(X, rt[indices[,b]])))); both <= 1e-10.
Local JSON re-read: 14 PASS, 0 FAIL (matrix/column/response-vector/index/replicate/interval shapes and consistency bounds). Indices all in [1,n]. No Python numerical re-scoring. G3 remains 0/3 pending #619; held-out data untouched.
