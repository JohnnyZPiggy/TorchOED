# Week 9 held-out H19–H20

| Case | TorchOED φ | Reference φ | Δ (reference − TorchOED) | Oracle / layout / floor |
|---|---:|---:|---:|---|
| H19 | 0.57673697151068 (PASS) | NOT_RUN | — | PASS / PASS / PASS |
| H20 | 1.89787865844977 (PASS) | NOT_RUN | — | PASS / PASS / PASS |

Frozen [§14](../../docs/P2_GATE.md#14-held-out-additions-w9-frozen-2026-11-16-never-used-for-tests-tuning-or-dispatch-choices-run-once-at-week-end-reported-as-is), measurement source 82bf52fb3f9a1a4b1b96a408a42cf085747f485b. Python 3.14.6, torch 2.14.1+cpu (W8 used 2.15.0+cpu), CPU float64, one thread, seed 20261117. H19 took 0.643039 s (0.064304 s/restart); H20 took 0.490604 s (0.049060 s/restart). H20 passed the frozen integer labels as X[:, 0].long().

The local output records grouped IDs, dense scores, stop reasons and floors. See [results](results.json), [R versions and optBlock documentation](sessionInfo.txt), [runtime](runtime.json) and [cleanup read-back](cleanup.json). Colab: w239-colab-20261117 PASS; authenticated assignments read back zero after the attempt. R calls are recorded verbatim in [the driver script](../../tools/r_heldout.R). Historical H1–H18 records stand in the [report index](../README.md).

H19 AlgDesign `optBlock` documents no I criterion. H20 skpr returned `undefined columns selected` on the frozen split-plot call. Both deltas are unavailable.

measurement only; no tuning followed
