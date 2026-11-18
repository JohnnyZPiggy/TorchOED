# W11 covariates external probe — BLOCKED (#339)
- Source: `25856bda2583125b75334555cbf9b391470fd547`; local CPU float64, Python 3.14.6, torch 2.15.0+cpu, one thread, seed 0, default 10 restarts.
- TorchOED det M_Z = 81.91999999999997: MATCH exact 2048/25; returned covariates equal input.
- Fraction oracle: 7,776 sequences, maximum 2048/25, 24 maximizers PASS; dense log-det vs objective PASS (2 PASS, 0 FAIL).
- Shared metric in run.py uses returned run order, Q = I - [1 Z]([1 Z]ᵀ[1 Z])⁻¹[1 Z]ᵀ, and det(FᵀQF).
- skpr NOT_RUN: pinned-controller kernel probe timed out before the R driver. No R/skpr versions or design obtained; DoD remains blocked. No repaired call or second session.
- Controller: `python3 /mnt/d/sae-scratch/.sae-271-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-339-20261118/linux-repo --source-sha 25856bda2583125b75334555cbf9b391470fd547 --run-id w339-colab-20261118 --out /mnt/d/sae-scratch/.sae-339-20261118/colab --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-339-20261118/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180`.
- No GPU computation was performed; observed device unavailable because the probe failed. CPU R intent only.
- Cleanup confirmed: assignments 0 → 0; balance 50.51 → 50.50; final usage 0.00/hr, zero active assignments.
- Re-score saved rows only: `PYTHONPATH=src python reports/w11-covext/run.py --score`. Never repeat the search/oracle without a new dispatch.
