# W174: remaining C1 profiles

Profiled source `eff5f9766e7521a85ad01d2abe288e87e70d8aaa`; original Issue source `6e20954b2392003e089a54bc5e47ba336ab34bff`: `git diff 6e20954 eff5f97 -- src tests tools` is empty. Fixed w141–w168 environment, one thread, seed0, restarts1, default1M workspace; unchanged bench_b0 pools/features.

One unprofiled same-SHA call precedes each cProfile call: **9/9 exact IDs/objectives PASS, 0 FAIL**. These are single profiles, not new five-seed C1 timing medians; C1 remains5/8.

Seconds: wall = N₂ + scores + cond + verify + floor + start + residual. Disjoint w165 cumulative caller accounting; residual is explicit, no overhead subtraction.

| Case | L | Wall | Unprofiled | N₂ | Scores | Cond | Verify | Floor | Start | Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid4 n15 | D | 0.029468 | 0.093053 | 0.005373 | 0.000000 | 0.001605 | 0.002474 | 0.006865 | 0.000808 | 0.012343 |
| grid4 n15 | A | 0.124662 | 0.136965 | 0.112522 | 0.004822 | 0.001354 | 0.000495 | 0.001658 | 0.000670 | 0.003141 |
| grid4 n15 | I | 0.089912 | 0.090331 | 0.072464 | 0.007225 | 0.001812 | 0.000730 | 0.002579 | 0.000641 | 0.004462 |
| grid4 n30 | D | 0.042456 | 0.039743 | 0.012659 | 0.000000 | 0.003147 | 0.004723 | 0.006600 | 0.000729 | 0.014599 |
| grid4 n30 | A | 0.092217 | 0.088877 | 0.068723 | 0.010465 | 0.002782 | 0.001070 | 0.001569 | 0.000609 | 0.006998 |
| grid4 n30 | I | 0.053825 | 0.051537 | 0.033920 | 0.008199 | 0.002030 | 0.000761 | 0.002723 | 0.000650 | 0.005542 |
| random500 n20 | D | 0.183240 | 0.177961 | 0.032131 | 0.000000 | 0.003557 | 0.005874 | 0.080077 | 0.000838 | 0.060763 |
| random500 n20 | A | 0.515880 | 0.496476 | 0.427564 | 0.023587 | 0.004464 | 0.001787 | 0.040696 | 0.000818 | 0.016964 |
| random500 n20 | I | 2.828463 | 2.748961 | 2.721864 | 0.021825 | 0.003977 | 0.001623 | 0.063647 | 0.000656 | 0.014871 |

| Case | L | Calls N₂/scores/cond/verify/floor/start | Floor iterations | One-swap iterations | Accepted k1/k2 |
|---|---|---:|---:|---:|---:|
| grid4 n15 | D | 4/0/23/11/1/5 | 38 | 12 | 9/2 |
| grid4 n15 | A | 5/10/19/18/1/4 | 8 | 10 | 7/2 |
| grid4 n15 | I | 5/16/31/30/1/4 | 14 | 16 | 13/2 |
| grid4 n30 | D | 5/0/41/20/1/5 | 38 | 21 | 17/3 |
| grid4 n30 | A | 3/22/43/42/1/4 | 8 | 22 | 20/1 |
| grid4 n30 | I | 1/16/31/30/1/4 | 14 | 16 | 15/0 |
| random500 n20 | D | 1/0/49/24/1/5 | 385 | 25 | 24/0 |
| random500 n20 | A | 1/28/55/54/1/4 | 194 | 28 | 27/0 |
| random500 n20 | I | 7/28/55/54/1/4 | 301 | 28 | 24/3 |

A/I N₂ = `_linear_pair_moves`; D analogue = `_two_candidates`. A/I verify = direct exchange cholesky_ex/cholesky_solve; D = `_verified_information` excluding starts. Start includes initial draw/sort and information/criterion. Calls include generator resumptions; floor iterations count direct floor cholesky_ex calls. Exact scripts, histories and caller splits are in profile.json.

OBSERVED — grid4 n15: A N₂ 0.112522s = 118.2% of wall−D wall (112.6% for N₂−D N₂), so N₂ dominates rather than scoring.
OBSERVED — grid4 n30: A N₂ 0.068723s = 138.1% of wall−D wall (112.7% for N₂−D N₂); I N₂ 0.033920s = 298.4% of wall−D wall (187.0% for N₂−D N₂), so N₂ dominates rather than scoring.
OBSERVED — random500 n20: A N₂ 0.427564s = 128.5% of wall−D wall (118.9% for N₂−D N₂); I N₂ 2.721864s = 102.9% of wall−D wall (101.7% for N₂−D N₂), so N₂ dominates rather than scoring.

Shares may exceed100% because D has floor/residual costs offsetting the A/I N₂ excess. No reruns to change shares.

Command: `../.sae-w95-venv/Scripts/python.exe scratch/capture.py` then `scratch/analyze.py`; scripts embedded in profile.json. No new pytest required; no src/tests/tools changes, other cases/seeds, speed-up proposal, PyDOE run, GPU, held-out, R/Julia or remote sessions.
