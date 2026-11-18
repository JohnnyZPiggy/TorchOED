# W10 daewr DSD single retry (#301)

The earlier #266 attempt is preserved at [PR #274](https://github.com/JohnnyZPiggy/TorchOED/pull/274) (`13a444e3ccd77d8b18efc557df854eb8757c9cb0`): ALLOCATION_FAILED before R.

This run used source `5ee662d3b6c69c16d24455f5f818475f9b6e50d2` and pinned SAE controller `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. The controller allocated a T4-requested session, but its kernel-smoke exec timed out after the 120-second kernel timeout (125-second process cap), returning `KERNEL_PROBE_FAILED`. Device identity, R/daewr versions, installed help usage and calls were NOT_RUN/unobserved. The scientific driver never started. No CSV or sessionInfo.txt was produced.

| k | daewr shape | TorchOED shape | Verdict |
|---|---|---|---|
| 6 | unobserved | 13 × 6 (existing constructor) | NOT_RUN: KERNEL_PROBE_FAILED |
| 7 | unobserved | 17 × 7 (existing constructor) | NOT_RUN: KERNEL_PROBE_FAILED |

Planned, not executed: `daewr::DefScreen(m=6, c=0, center=0, randomize=FALSE)` and m=7 if installed help supports `randomize`; otherwise the existing K7 branch omits that argument. No MATCH/NO MATCH claim or synthetic substitute. Required local checker exits with `NOT_RUN: no frozen daewr CSV (see runtime.json)`.

Only one controller allocation was attempted. No retry: #301 permits one only on ALLOCATION_FAILED before R, whereas this was KERNEL_PROBE_FAILED. Three local setup errors before allocation were corrected: CRLF in the scratch SHA file; absent scratch temp directory; reused preflight output directory. Final preflight PASS. No scientific reruns.

Cleanup: stop PASS, cleanup_confirmed=true; controller and fresh authenticated usage both confirm assignments 0→0, credits 50.89→50.89. No remote remains. See runtime.json and cleanup.json. Five report files, one net driver line; existing K7 body and local checker unchanged. No matrix or other directories edited.
