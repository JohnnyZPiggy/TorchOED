# W165: random500 non-N₂ accounting

Source `6b4854f757bf1a55bd825cec4defd1f504dde854`; seed0/restarts1/default1M, fixed one-thread environment. Each of six rows is cProfile on unchanged optimal_design after an exact same-SHA unprofiled call. **6/6 IDs/objectives PASS,0 mismatch.** No src/tests/tools changes.

Seconds: wall = N₂ + scores + cond + verification + floor + start + residual. Components are disjoint cumulative times; residual is explicit and includes remaining native/Python work, profiler/clock and wrapper overhead.

| n | L | Wall | N₂ | Scores | Cond | Verify | Floor | Start | Residual |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | D | 0.134182 | 0.002785 | 0.000000 | 0.002355 | 0.003760 | 0.084631 | 0.000886 | 0.039765 |
| 10 | A | 0.296449 | 0.053783 | 0.018618 | 0.003109 | 0.001186 | 0.209552 | 0.000735 | 0.009467 |
| 10 | I | 0.277950 | 0.039404 | 0.013965 | 0.002096 | 0.000867 | 0.213990 | 0.000651 | 0.006977 |
| 20 | D | 0.176845 | 0.028859 | 0.000000 | 0.003458 | 0.005686 | 0.076420 | 0.000777 | 0.061645 |
| 20 | A | 0.658480 | 0.400203 | 0.020985 | 0.003335 | 0.001305 | 0.218102 | 0.000613 | 0.013939 |
| 20 | I | 2.972762 | 2.718060 | 0.022393 | 0.003611 | 0.001425 | 0.212382 | 0.000652 | 0.014240 |

| n | L | Calls scores/cond/verify | Calls N₂/floor/start | Floor iterations | One-swap iterations | Accepted k1/k2 |
|---:|---|---:|---:|---:|---:|---:|
| 10 | D | 0/33/16 | 1/1/5 | 385 | 17 | 16/0 |
| 10 | A | 27/53/52 | 3/1/4 | 1000 | 27 | 25/1 |
| 10 | I | 22/43/42 | 5/1/4 | 1000 | 22 | 19/2 |
| 20 | D | 0/49/24 | 1/1/5 | 385 | 25 | 24/0 |
| 20 | A | 28/55/54 | 1/1/4 | 1000 | 28 | 27/0 |
| 20 | I | 28/55/54 | 7/1/4 | 1000 | 28 | 24/3 |

A/I scores = _linear_scores; cond includes iteration and candidate condition estimates. A/I verify is the direct _linear_exchange cholesky_ex + cholesky_solve kernels (individual times/counts in JSON); trial products/trace/item remain in residual. D verify = _verified_information from k2_exchange, excluding starts. A/I N₂ = _linear_pair_moves; D N₂ analogue = _two_candidates (_linear_pair_moves is0 for D). Generator call counts include resumptions, not just new scans.

Start = randperm/sort plus directly called initial feasibility/information/criterion functions; start calls count those operations (draws are1 in every row). Dual iterations count cholesky_ex calls directly from _dual_floor, one per iteration. Accepted histories are captured from unchanged returns. A/I one-swap iterations count score states; D uses accepted swaps plus terminal scan; score-pass/tie-rescan counts are separately retained.

**Observation:** the floor dominates A/I non-N₂ time (83.4%–89.7%), taking0.210–0.218s versus0.076–0.085s for D. Actual Torch D floors terminate at385 iterations; all four A/I floors hit1000. Scores take0.014–0.022s; condition estimates and verification kernels are much smaller. This identifies a cost, not a speed-up proposal.

Command: `../.sae-w95-venv/Scripts/python.exe ../.sae-w165-control-20261109/capture_nonn2.py`, then `analyze_nonn2.py`. Exact scripts, caller splits, per-kernel counts/times, iteration/history data and accounting identities: [profile.json](profile.json). No overhead subtraction; rounded table sums may differ in the last digit.

No new pytest required (evidence only), no other cases/seeds/budgets, PyDOE run, GPU, held-out cases, R/Julia or remote sessions. C1 remains4/8; no queued #168 work.
