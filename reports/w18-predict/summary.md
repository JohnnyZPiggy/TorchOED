# W18 prediction profiler: frozen R comparison

R 4.5.2, rsm 2.10.6; reference workflow run [45936707367](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45936707367). No R re-run.

Reproduce: `python reports/w18-predict/run.py` (CPU float64; no timing claim). Models rebuilt from W17 CSVs in R column order; ChemReact uses centre and stationary point, Block B1.

Shared metric: maximum relative difference over fit, se.fit and all four interval ends, denominator |R| (floor 1e-300). Target 1e-8.

- warpbreaks: MATCH, 6 points, max relative difference 6.74019e-14.
- ChemReact: MATCH, 2 points, max relative difference 8.92471e-15.

Published examples: 2/2 MATCH. H50 is reserved for week end and was not run.
