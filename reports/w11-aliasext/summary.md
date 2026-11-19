# W11 alias external probe

Refs #349. Corrected skpr Alias call: MATCH; TorchOED MATCH. Shared metric tolerance: 1e-10 relative.

| Design | det M | alias trace | D-efficiency | Psi | On front | Verdict |
|---|---:|---:|---:|---:|---|---|
| skpr | 768 | 0.33333333333333331 | 0.93060485910209956 | -0.29747870260780362 | yes | MATCH |
| TorchOED | 768 | 0.33333333333333331 | 0.93060485910209956 | -0.29747870260780362 | yes | MATCH |

- One Colab CPU-R call at source `818807ec9e22d16dc9d90a606ae9b431b6248952`; merged in #347. Verbatim error: `unused argument (aliasmodel = ~(x1 + x2 + x3)^2) ` (trailing space).
- Corrected call: `gen_design(cand, model=~x1+x2+x3, trials=6, optimality="Alias", aliaspower=2, minDopt=0.8, repeats=20)`, seed 20261119; R 4.6.1 / skpr 1.9.2. Full dependency versions and signature: sessionInfo.txt. CSV rows are frozen in skpr.csv.
- Source: `2ca8cd57da1a24aff69bc0608d423210ac0319f2`. One pinned-controller Colab CPU-R run, unchanged driver; no further repaired call or second session.
- Reused the TorchOED search and exact Fraction enumeration from #347 unchanged: source `1b67717f5665bf85b29e5c20847bf8190fa0ee32`, Python 3.14.6 / torch 2.15.0+cpu, CPU float64, one thread, seed 0, 10 restarts. No search or enumeration rerun. Exact front: `(1024, 3/2)` x 12, `(768, 1/3)` x 4; 1,716 multisets, 19 nonsingular pairs.
- `PYTHONPATH=src python reports/w11-aliasext/run.py --score`: 2 PASS, 0 FAIL (saved exhaustive-oracle and dense-rescore assertions); both six-row designs re-scored with the shared dense metric.
- Controller transport PASS, 859.642 seconds. Observed Tesla T4 / driver 580.82.07 / torch 2.14.1+cu130 / CUDA 13.0; R used CPU with CUDA hidden, no GPU science or timing claim.
- Cleanup read back: assignments 0 -> 0, balance 50.06 -> 49.90, final 0.00/hr and zero active assignments.
- Downloaded r-heldout.json: SHA-256 f88b8d0ebc6cb603f437889e8ca793f2359e4ce96537fbff9089f5e6da44b362, 13,505,240 bytes. Runtime and cleanup are frozen separately.
- Controller command: `python3 /mnt/d/sae-scratch/.sae-271-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-349-20261119/linux-repo --source-sha 2ca8cd57da1a24aff69bc0608d423210ac0319f2 --run-id w349-colab-20261119 --out /mnt/d/sae-scratch/.sae-349-20261119/colab --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-349-20261119/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`; pinned SAE 4129001, preflight PASS. Paths and source variables are defined in the task-owned run script; runtime.json records the resolved source.
