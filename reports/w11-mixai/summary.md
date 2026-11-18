# W11 AlgDesign A/I mixture reference (#296)

One CPU-R Colab run at `0c543152c69bc62544e6e336b630529746ff9453`, observed Tesla T4 host with CUDA hidden from R. R 4.6.1, AlgDesign 1.2.1.2, Ubuntu 24.04.4; exact session and installed function arguments in sessionInfo.txt. Installed AlgDesign alone from https://cloud.r-project.org. No second run.

Calls (A followed by I, resetting seed 11 before each): `AlgDesign::optFederov(~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3, data=candidates, nTrials=10, criterion=crit)`. Candidate bytes copied unchanged from reports/w11-mixref/candidates_r1.csv. No held-out inputs.

Shared CPU-float64 metrics: `A(X)=tr((X.T X)^-1)` and `I(X)=tr((X.T X)^-1 F.T F/13)`. R CSVs preserve original returned candidate IDs and coordinates; scoring uses the corresponding frozen candidate rows. Exhaustive enumeration covers all 286 ten-row subsets; both agree with the Architect's numbers within 1e-9 relative.

```text
A MATCH TorchOED 2056.070972199583 R 2056.070972199569 exhaustive 2056.070972199569 repeats False
I MATCH TorchOED 0.5760293191622156 R 0.5760293191622129 exhaustive 0.5760293191622129 repeats False

```

Both verdicts MATCH. Neither R design repeats candidates. TorchOED uses without-replacement designs (gate §48.5), 10 restarts and seed 11. No search tuning or new timing claim.

Controller transport PASS; stop PASS, cleanup_confirmed=true, authenticated assignments 0→0, credits 50.89→50.89. No remote remains. Download size and SHA-256 are in runtime.json, cleanup readback in cleanup.json. Eight report files. Source+checker 47 net lines against S40 (under 1.5x); no extras. Existing R/driver branches remain byte-unchanged; no matrix edit.
