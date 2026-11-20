# Week 13 frozen held-out H33–H35

Issue #426. Frozen calls, seed 0, default settings; no tuning or extra cases. Source base 79b673af9a8301a91f8fba7bea09f53fb2fe06c8.

| Case | TorchOED | Reference | Delta (TorchOED minus reference) | Verdict |
|---|---|---|---|---|
| H33 | 10 | 10 | 0 | MATCH |
| H34 | 22.785792408469998 | 23.57522175340001 | -0.7894293449300136 | MATCH or TorchOED better |
| H35 | 14095/2420208 | 14095/2420208 | 0 | MATCH |

H33: exact integer minimum squared rank distance; 3,628,800 permutations. TorchOED has 10 pairs at its minimum.
H35: exact Fraction CD2 squared; all 5,040 permutations. Independent direct Fraction sum on TorchOED design equals the vectorised integer enumeration formula exactly (1 PASS, 0 FAIL).
H34: shared TorchOED diagnostics on column ranks mapped to midpoints. Minimum distances: TorchOED 0.33541019662496846, R 0.2598076211353316. Verdict threshold: psi_T <= psi_R * (1 + 1e-9).
R oracle run [44676636724](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44676636724): R version 4.5.2 (2025-10-31), MaxPro 4.1-2. Exact call in heldout.R; downloaded H34-R.csv, sessionInfo.txt, runtime.json.
Environment: torch 2.15.0+cpu; NumPy 2.5.4 (dev-only enumeration).
Reproduce: `PYTHONPATH=src python reports/w13-heldout/run.py`. Final-HEAD byte-for-byte reproduction is checked in the PR handoff. Scientific NO MATCH is recorded as-is, not a test failure.
