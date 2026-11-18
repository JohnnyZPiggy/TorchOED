# W11 mixexp lattice/centroid reference — #321

Attempt 1 (#317/#320, `364adab`): CRAN unavailable, all seven NOT_RUN.
Retry measured source `8be5b191ba69363d5a111fffa9fe04be88f419ba`; base `364adabdcfec938362351e77af773fbe880355d9`.
Before allocation, `git ls-remote origin refs/heads/evidence/321-mixexp-archive` returned `8be5b191ba69363d5a111fffa9fe04be88f419ba	refs/heads/evidence/321-mixexp-archive`.
Installed `https://cran.r-project.org/src/contrib/Archive/mixexp/mixexp_1.2.7.1.tar.gz` using the Issue's exact R expression (lattice/daewr dependencies).
R 4.6.1 (2026-06-24), mixexp 1.2.7.1; SLD args `(fac, lev)`, SCD args `(fac)`; complete versions in sessionInfo.txt.
Exact calls: `mixexp::SLD(3,2)`, `SLD(3,3)`, `SLD(4,2)`, `SLD(4,3)`, `mixexp::SCD(3)`, `SCD(4)`, `SCD(5)`. R exit 0, empty stdout/stderr; frozen CSV.

| Case | Shape (both) | Verdict | Max absolute deviation |
|---|---|---|---|
| L32 | 6 × 3 | MATCH | 0.0 |
| L33 | 10 × 3 | MATCH | 0.0 |
| L42 | 10 × 4 | MATCH | 0.0 |
| L43 | 20 × 4 | MATCH | 0.0 |
| C3 | 7 × 3 | MATCH | 0.0 |
| C4 | 15 × 4 | MATCH | 0.0 |
| C5 | 31 × 5 | MATCH | 0.0 |

Metric: unchanged run.py, shape and rounded-1e-9 lexicographic row multiset, max deviation ≤ 1e-12 and both row sums within 1e-12. `PYTHONPATH=src py -3.14 -X utf8 reports/w11-mixlat/run.py`: exit 0, seven MATCH verdicts, zero failures.
Pinned SAE `4129001fffcb336f5203e4b0d6765d7a0ea3150c`: `python3 <pinned>/colab_control.py --repo-path <linux-repo> --source-sha 8be5b191ba69363d5a111fffa9fe04be88f419ba --run-id w321-colab-retry-20261118 --out <scratch>/colab-retry --cli colab --device T4 --temp-root <scratch>/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`.
First allocation failed before R (new command TIMEOUT, ALLOCATION_FAILED, 177.2412s); authorized single retry PASS (85.6978s). Both confirmed cleanup, assignments 0→0 and credits 50.87→50.87. Preflight local missing-repo/CRLF-path failures were corrected before allocation; final and retry preflights PASS. No third attempt.
Observed Tesla T4, torch 2.14.1+cu130; R ran on CPU with CUDA hidden. Stop PASS, authenticated cleanup confirms zero sessions and unchanged credit; see cleanup.json. Source/status verified in downloaded artifact.
Downloaded r-heldout.json SHA256 fdd73d1c21acc4f451072331fb1a534952455579abb259adfca14379cf977449.
Only the archive install expression changed; existing driver branches, R cases and checker remain byte-unchanged. No matrix edit, held-out input, timing or optimum claim. Five report files; one net driver line.
