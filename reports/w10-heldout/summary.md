# Week 10 held-out H21–H23

Frozen cases from [WEEK_GOAL.md](../../docs/WEEK_GOAL.md), Refs #271.
Measurement source `7ec0e24f6ec7cc37ac8c93c4c0b9106b5ef72d2a`. One local CPU float64 run with Python 3.14.6, torch 2.14.1+cpu and one torch thread.

| Case | TorchOED | R reference | Δ (TorchOED − R) |
|---|---|---|---|
| H21 fractional factorial | A3_A7 = (0, 7, 0, 0, 0); resolution = 4 | NOT_RUN | — |
| H22 blocked CCD | D_B = 0.5614129678567744; max_centered_gap = 0; alpha = 1.632993161855452 | D_B = 0.5614129678567735; max_centered_gap = 1.77635683940025e-15; alpha = 1.63299316185545 | D_B = 8.881784197001252e-16; max_centered_gap = -1.77635683940025e-15; alpha = 1.998401444325282e-15 |
| H23 Box–Behnken | D = 0.2521634059405792; I = 31.5 | D = 0.2521634059405792; I = 31.5 | D = 0; I = 0 |

TorchOED H21 generators (zero-based): `[[0, 1, 2], [0, 1, 3], [0, 2, 3]]`.
In-run oracles (TorchOED/R): H21 PASS/NOT_RUN, H22 PASS/PASS, H23 PASS/PASS. See [results.json](results.json).

Colab controller returned `FAIL` with `ARTIFACT_SOURCE_OR_STATUS_FAILED`. The download and pinned-source check passed. The controller rejected the envelope because the driver omitted its status field. The final driver supplies that field, and the downloaded CSVs are preserved byte for byte.
References measured: 2/3. H22 and H23 used rsm's stored `rsdes$primary` coded columns and H22 used the `Block` labels. R 4.6.1, FrF2 2.3.5, DoE.base 1.2.5 and rsm 2.10.6 are recorded in [sessionInfo.txt](sessionInfo.txt). H21 stopped at the generator lookup with the verbatim R error in [results.json](results.json). The final R branch uses the exporting DoE.base namespace. The R calls were recorded as run.
Authenticated cleanup confirmed 0 → 0 assignments and 51.17 → 51.1 compute units. See [cleanup.json](cleanup.json).

Earlier held-outs: [W4](../w4-heldout/), [W5](../w5-heldout/), [W6](../w6-heldout/), [W7](../w7-heldout/), [W8](../w8-heldout/), [W9](../w9-heldout/).

measurement only; no tuning followed
