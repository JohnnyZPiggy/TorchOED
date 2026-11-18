# W11 held-out H24–H26 — blocked single attempt

Refs #329. Local numerical source `main@4aad71e39cc1ac2f37c924f763db229dc360a6dd`; committed controller source `149a5f40a6f76f19aaa09275eaababdb67127f82`. Python 3.14.6, torch 2.15.0+cpu, float64, seed 0. No timing or tuning claim.

| Case | TorchOED value | Reference value | Δ |
|---|---|---|---|
| H24 | 6 vertices; exact rational exhaustive vertex-set oracle PASS | NOT_RUN: allocation failed | N/A |
| H25 | 13 candidates frozen; D unavailable (first search output lost before saving) | NOT_RUN: allocation failed | N/A |
| H26 | NOT_RUN: frozen float64 `hard=C[:,0]` rejected | NOT_RUN: allocation failed | N/A |

H24 rational coordinates are reconstructed from float64 with `Fraction.limit_denominator(10**6)` and equal exhaustive rational bound-pair enumeration; every rational row is feasible and sums exactly to one. Raw returned coordinates and exact rows are in results.json. H25 search ran once, but the prescribed all-fixed certificate call raised `ValueError: fixed_ids must be distinct candidate IDs with length < n` before its outputs were saved. H26 raised `ValueError: hard must be a length-m integer tensor on F's device` before search. H25/H26 metric oracles are NOT_RUN; neither search was repeated with changed arguments. Architect ruling is required before additional measurements.

One pinned SAE controller attempt, T4 allocation requested solely for CPU R, CUDA hidden in the driver: preflight PASS, allocation `new` TIMEOUT / ALLOCATION_FAILED, no observed device or R execution. `sessionInfo()`, mixexp, AlgDesign and skpr versions: NOT_RUN. Reference columns are all NOT_RUN. Controller stop PASS; fresh authenticated usage confirms zero assignments and unchanged balance 50.87 → 50.87 compute units (cleanup.json).

Command: pinned `colab_control.py --repo-path <linux-repo> --source-sha 149a5f40a6f76f19aaa09275eaababdb67127f82 --run-id w329-colab-20261118 --out <scratch>/colab --cli colab --device T4 --temp-root <scratch>/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180`.

An initial CRLF shell setup failed before any controller invocation; corrected LF and login PATH, then made the single controller attempt above. Local H25/H26 dispatch conflicts are recorded on #329. H1–H23 were not rerun: prior evidence remains in reports/w3-heldout, reports/w6-heldout, reports/w7-heldout, reports/w8-heldout, reports/w9-heldout and reports/w10-heldout. Seven evidence files; no production/test/doc/website edits.

measurement only; no tuning followed
