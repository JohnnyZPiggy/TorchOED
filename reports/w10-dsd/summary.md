# Corrected daewr DSD reference (#309)

Prior attempts: [#266 / PR #274](https://github.com/JohnnyZPiggy/TorchOED/pull/274), ALLOCATION_FAILED; [#301 / PR #304](https://github.com/JohnnyZPiggy/TorchOED/pull/304), KERNEL_PROBE_FAILED; [#305 / PR #308](https://github.com/JohnnyZPiggy/TorchOED/pull/308), DOWNLOAD_FAILED (unpushed source and driver precedence defects).

Measured SHA `51e02839f9be4ad57ca90a99c6d543646d377092` was pushed before allocation. Exact `git ls-remote origin refs/heads/bench/309-dsd-corrected` readback: `51e02839f9be4ad57ca90a99c6d543646d377092 refs/heads/bench/309-dsd-corrected`.

Pinned SAE controller `4129001fffcb336f5203e4b0d6765d7a0ea3150c`: one attempt, `--task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180 --device T4`. Preflight PASS. Observed Tesla T4 (15360 MiB, driver 580.82.07), torch 2.14.1+cu130, CUDA 13.0; R used CPU with CUDA hidden. Duration 41.509961469 seconds; no retry.

Observed R 4.6.1 (2026-06-24), Ubuntu 24.04.4 LTS, daewr 1.2-11 (packageVersion 1.2.11). Exact sessionInfo is frozen in sessionInfo.txt. Installed help: `DefScreen(m, c=0, center=0, randomize=FALSE)`.
Exact calls: `daewr::DefScreen(m = 6L, c = 0, center = 0, randomize = FALSE)` and `daewr::DefScreen(m = 7L, c = 0, center = 0, randomize = FALSE)`.

| k | daewr shape | TorchOED shape | Shared signed-column-permutation / row-multiset verdict |
|---|---|---|---|
| 6 | 13 × 6 | 13 × 6 | MATCH: zero-based π=(0,1,2,3,4,5), s=(-1,-1,1,1,1,1) |
| 7 | 15 × 7 | 17 × 7 | NO MATCH (shape); both designs are printed by run.py |

Required local command: `PYTHONPATH=src py -3.14 -X utf8 reports/w10-dsd/run.py`, exit 0. Both verdicts above reproduced; no scientific overall MATCH claim. Frozen CSV contains the reference; the unchanged constructor reproduces the TorchOED design. Pure driver block move and unchanged run.py/R checks: 3 PASS, 0 FAIL. No timing comparison, matrix edit or held-out input.

Controller status remains **FAIL: ARTIFACT_SOURCE_OR_STATUS_FAILED**: download succeeded, source_sha matches, but the unchanged K7 branch does not emit the required top-level status=PASS. Its downloaded artifact has errors={} and contains the complete CSV and sessionInfo. These were extracted locally and checked independently; controller FAIL is not relabelled PASS. No retry or additional driver edit is allowed. External r-heldout.json: SHA256 fc3ba683fc9ba6df401e6bb714eca9e7507514d394a178e2fae42ba0d99614de, 1888 bytes.

Cleanup: stop PASS, cleanup_confirmed=true; assignments 0→0, credits 50.87→50.87, confirmed by authenticated usage before/after in cleanup.json. No remote remains. Seven report files total; tools diff is a pure block move (16 insertions/16 deletions), run.py and tools/r_heldout.R unchanged.
