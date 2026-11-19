# W11 alias external probe

Refs #342. TorchOED MATCH; skpr NOT_RUN (the specified argument is rejected).

| Design | det M | alias trace | D-efficiency | Psi | On front | Verdict |
|---|---:|---:|---:|---:|---|---|
| TorchOED | 768 | 0.3333333333333333 | 0.9306048591020996 | -0.2974787026078036 | yes | MATCH |
| skpr 1.9.2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

- Exact Fraction oracle: all 1,716 multisets, 19 distinct nonsingular pairs; Pareto front `(1024, 3/2)` x 12 and `(768, 1/3)` x 4. PASS.
- Dense shared metric reproduces TorchOED returned objective -0.29747870260780374 to 1e-10 relative. PASS. Two required checks: 2 PASS, 0 FAIL.
- One local search and enumeration at source `1b67717f5665bf85b29e5c20847bf8190fa0ee32`; Python 3.14.6 / torch 2.15.0+cpu, CPU float64, one thread, seed 0, default 10 restarts. Subsequent source only combined two import lines and formatted NOT_RUN output. No search or enumeration repeated.
- One Colab CPU-R call at source `818807ec9e22d16dc9d90a606ae9b431b6248952`; R 4.6.1 / skpr 1.9.2, full dependencies and gen_design signature in sessionInfo.txt. Error verbatim: `unused argument (aliasmodel = ~(x1 + x2 + x3)^2) ` (trailing space). No CSV; no repaired call or second session.
- Controller transport PASS, 777.827 seconds. Observed Tesla T4, driver 580.82.07; remote torch 2.14.1+cu130 / CUDA 13.0. R used CPU with CUDA hidden; no GPU science or timing claim.
- Cleanup read back: assignments 0 -> 0, balance 50.28 -> 50.15 compute units, final rate 0.00/hr and zero active assignments.
- Controller: pinned SAE 4129001, `python3 /mnt/d/sae-scratch/.sae-271-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-342-20261119/linux-repo --source-sha 818807ec9e22d16dc9d90a606ae9b431b6248952 --run-id w342-colab-20261119 --out /mnt/d/sae-scratch/.sae-342-20261119/colab --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-342-20261119/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`; preflight PASS before allocation.
- Downloaded r-heldout.json: SHA-256 c0068cbb4c23a51bf05443404f841c7f6a68ba53f1b00b49e680c92ac11dc883, 13,505,074 bytes. Runtime and cleanup are frozen separately; installation log excluded.
- Re-score saved rows only: `PYTHONPATH=src python reports/w11-aliasext/run.py --score`. No new tests, B0, held-out work or tuning required.
