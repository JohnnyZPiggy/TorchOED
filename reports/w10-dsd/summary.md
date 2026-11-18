# W10 daewr DSD final attempt (#305)

Previous attempts: [#266 / PR #274](https://github.com/JohnnyZPiggy/TorchOED/pull/274), ALLOCATION_FAILED before R.
Previous retry: [#301 / PR #304](https://github.com/JohnnyZPiggy/TorchOED/pull/304), KERNEL_PROBE_FAILED before R.

Measured source: `31f63ea26f6776608aa31cef859008522ba82978`; SAE controller pin `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. One controller attempt, `--device T4 --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`; derived kernel process cap 305 seconds. Preflight PASS. Observed Tesla T4, 15360 MiB, driver 580.82.07; torch 2.14.1+cu130, CUDA 13.0. R driver is CPU-only with CUDA hidden.

The kernel smoke and work exec returned PASS/exit 0, but artifact download returned FAIL/exit 1. The exact controller error is `DOWNLOAD_FAILED`. Total controller duration 56.126675252999995 seconds. No r-heldout.json, CSV or sessionInfo.txt was retrieved. R/daewr versions, installed help usage, exact executed calls and both scientific verdicts are unobserved, NOT_RUN. An exec exit code is not scientific evidence.

| k | daewr shape | TorchOED shape | Verdict |
|---|---|---|---|
| 6 | unobserved | 13 × 6 (existing constructor) | NOT_RUN: DOWNLOAD_FAILED |
| 7 | unobserved | 17 × 7 (existing constructor) | NOT_RUN: DOWNLOAD_FAILED |

Planned, not verified executed: `daewr::DefScreen(m=6, c=0, center=0, randomize=FALSE)` and m=7 if installed help supports `randomize`; otherwise the frozen K7 branch omits it. No MATCH/NO MATCH claim. Required local checker reports `NOT_RUN: no frozen daewr CSV (see runtime.json)` (exit 1).

The measured commit had not been pushed before allocation: the remote driver clones GitHub and cannot check out that source. The controller exposes no work stderr, so its observed terminal error remains DOWNLOAD_FAILED. A further existing limitation is that the frozen driver tests the w11-mixai directory before the DSD retry marker; tools are outside this task's allowed scope. Neither limitation was changed or retried. Local preallocation setup corrections: Windows worktree could not be cloned from WSL (used a Git bundle); non-login shell lacked colab on PATH (used login shell). Neither allocated a session.

No retry: only ALLOCATION_FAILED before R permits one. This is the final attempt; the published DSD match needs the Scientist defer decision described in #305.

Cleanup: stop PASS, cleanup_confirmed=true; controller and fresh authenticated usage confirm assignments 0→0, credits 50.89→50.89. No remote remains. See runtime.json and cleanup.json. Five report files total; run.py and both tools byte-unchanged, no matrix edit.
