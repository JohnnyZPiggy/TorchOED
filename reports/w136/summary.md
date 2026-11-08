# W136: closed-form screen, gather4 decisions

Base c2bd9ef; measured source288771b; PR#138. Warm baseline: reports/w131/results.json, sourcef83993c. Search/_pair_tiles code and runtime are unchanged between that evidence and base; only the public scorer changed.
Protocol: unchanged tools/bench_b0.py, same Windows dev venv/PyDOE1.6 wheel, one thread, seeds0..4; D3 calls/seed, A/I1, restarts1. Head includes audit/capture overhead, not subtracted; baseline capture did not include H1 diagnostics.
Equality:120/120 successful D/A/I IDs/objectives exact;15/15 infeasible errors exact;40/40 default10-restart D reference rows reproduce. Grid2 n12 remains infeasible.

| Case | #131 A:D / I:D | Head A:D / I:D | D s base -> head |
|---|---|---|---|
| grid2 n=6 | 0.883 / 0.978 | 1.035 / 1.151 | 0.005777 -> 0.005868 |
| grid2 n=7 | 0.888 / 0.951 | 0.933 / 1.163 | 0.005563 -> 0.005683 |
| grid3 n=10 | 2.339 / 1.622 | 1.533 / 1.348 | 0.011582 -> 0.012499 |
| grid3 n=20 | 1.118 / 1.234 | 1.004 / 1.132 | 0.012061 -> 0.011121 |
| grid4 n=15 | 29.873 / 15.680 | 10.156 / 6.111 | 0.024691 -> 0.025769 |
| grid4 n=30 | 71.095 / 72.250 | 22.429 / 21.822 | 0.024825 -> 0.026047 |
| random500 n=10 | 123.538 / 185.391 | 68.322 / 101.807 | 0.115830 -> 0.118055 |
| random500 n=20 | 357.867 / 181.230 | 194.502 / 99.272 | 0.164880 -> 0.163956 |

| Case | Rescored fraction A / I | Max H1-scoped ratio A / I | Max finite all-rescored ratio A / I | Nonfinite gaps A / I |
|---|---|---|---|---|
| grid2 n=6 | 0.156 / 0.0889 | N/A / N/A | 9.81e+25 / 1.05e+26 | 0 / 0 |
| grid2 n=7 | 0 / 0 | N/A / N/A | N/A / N/A | 0 / 0 |
| grid3 n=10 | 0.174 / 0.141 | 4.56e-05 / 5.85e-06 | 3.18e+44 / 3.88e+44 | 49 / 38 |
| grid3 n=20 | 0 / 0.000376 | N/A / 5.29e-06 | N/A / 5.29e-06 | 0 / 0 |
| grid4 n=15 | 0.0295 / 0.0301 | 7.5e-06 / 7.07e-06 | 2.4e+44 / 2.51e+45 | 182 / 35 |
| grid4 n=30 | 3.01e-06 / 1.58e-06 | 4.73e-06 / 1.44e-06 | 4.73e-06 / 1.44e-06 | 0 / 0 |
| random500 n=10 | 0.0032 / 0.00341 | 4.97e-05 / 4.38e-05 | 1.31e+27 / 4.87e+26 | 0 / 0 |
| random500 n=20 | 7.33e-08 / 4.58e-08 | 2.14e-05 / 5.34e-06 | 2.14e-05 / 5.34e-06 | 0 / 0 |

H1 ratio is |closed-gather4|/(delta/2), over legal rescored pairs. Scoped max=4.9705602e-05 over126 samples. Scope excludes the always-rescored band and requires gather>tol-delta or closed>tol-delta/2. All finite maxima and nonfinite counts are reported separately; huge band gaps are outside H1's premise and are never used to exclude a pair. N/A means no sample. This is a rescored witness, not a full-scan proof of H1.
Fractions aggregate pair visits over all states/cutoffs/seeds. No benchmark state needed the unscreened workspace fallback. Random500 n20: A16/218424000 and I10/218424000 rescored.
C1 <=2x on every case remains NOT MET. Random500 n20 medians A59.005186->31.889677s; I29.881185->16.276258s. D code is structurally unchanged; measured D medians above are reported as-is.
Tests:105 passed,0 failed locally,50.71s; one optional NumPy warning. New saturated Fraction identity:18 legal gains,18 singular checks (rho0); H2 107/1387 and split17, A/I bitwise; whole tie/cutoff sequence equals full scan. Supplemental single-pair H2 also passed; a tight-workspace test matches the unchanged full scan exactly. Final CI linked in ENGINEER_HANDOFF.
Workspace: fixed=8mp+16nm+8p². available=max_elements-fixed-2q²; closed capacity=floor(available/96); rescore capacity=floor((available-8*closed_capacity)/(32p+256)); fallback if either<1. Closed temporaries die before gather4;8 row-ID slots per closed pair are retained. Original unscreened capacity unchanged.
Audit adapter is embedded verbatim in results.json. It records only defined legal gather4 gains; H1 failure aborts. No D/public-scorer/tile-generator/tolerance/gate edits, no held-out/GPU/R/Julia work.
