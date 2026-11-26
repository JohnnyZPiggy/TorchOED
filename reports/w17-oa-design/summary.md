# W17 OA catalogue comparison (#573)

8/8 MATCH as frozen: both sides strength 2, equal run count and |delta A3| <= 1e-9. No retuning.

| Case | Levels | N (both) | A3 (both) | Verdict |
|---|---|---|---|---|
| C1 | 2^7 | 8 | 7 | MATCH |
| C2 | 2^11 | 12 | 18.333333333333332 | MATCH |
| C3 | 2^15 | 16 | 35 | MATCH |
| C4 | 3^4 | 9 | 8 | MATCH |
| C5 | 3^13 | 27 | 104 | MATCH |
| C6 | 4^5 | 16 | 30 | MATCH |
| C7 | 5^6 | 25 | 80 | MATCH |
| C8 | 4^2 2^9 | 16 | 33 | MATCH |

One [R workflow run 45853428085](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45853428085), frozen input SHA c9b4da7bcfa17e98cce43f34bf466f80591a282c: R 4.5.2, DoE.base 1.2-5 (CRAN current at START). Local Python 3.14.6, torch 2.15.0+cpu, one CPU thread. No local R, GPU or held-out H47.

The same scorer measures both arrays. For a scaled complete contrast basis, sum_c c(a)c(b) = s*[a=b]-1; expanding the product of (1+z*(s*[a=b]-1)) and averaging over row pairs gives exactly the frozen contrast-product definition at every degree. Integer polynomial coefficients avoid basis rounding; only the final division uses float64. Required hand-built A3 checks: 1, 7, 8 (3 PASS). Full A1...Ak vectors and unmodified R GWLP output (including A0) are informational only in results.json.

Reproduce: `PYTHONPATH=src python reports/w17-oa-design/run.py` (8 shared case scorings, 3 sandbox checks); results.json is byte-identical on rerun. The four downloaded artifact files are preserved byte-for-byte. No library, tests, matrix, website, workflow or dependency metadata changed; tools/r_packages.txt has only the DoE.base pin. General covering arrays and best-known-size evidence remain separate tasks; these results alone do not close the OA/CA checklist row.
