# Week 18 desirability — G3

- Check (a): PASS; 15 frozen-R individual/overall values, max absolute 8.881784197001252e-15 (limit 1e-10).
- Check (b): PASS; default midpoint starts, best D = 0.9425094070843424 at [-0.5116969805768393, 1.6819999999999995, -0.5863993164533298].
- R documented D = 0.942509403229837; TorchOED 101³ grid maximum = 0.9400058024309151; both comparisons allow 1e-6.
- 11/125 starts within 1e-6 of best; all final values and accepted-step counts in results.json.
- Frozen conversion/activity models transcribed from `reports/w18-r-refs/r.json` (R 4.5.2, desirability 2.1, workflow 45936707367).
- CPU float64, one thread; no timing or global-optimality claim. No new R dispatch or held-out case.
- Reproduce from repository root: `PYTHONPATH=src python reports/w18-desirability/run.py`.
