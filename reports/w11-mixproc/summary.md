# W11 mixture-process AlgDesign reference — #313

Measured source `e5d47ac423c31a20521967f45a163b00d057661a`; main base `ce408ae03e12e6aa20203e51755d55af1f49e186`.
Before allocation, `git ls-remote origin refs/heads/evidence/313-mixture-process` returned `e5d47ac423c31a20521967f45a163b00d057661a\trefs/heads/evidence/313-mixture-process` (push confirmed).
Frozen inputs: 13 R1 rows from w11-mixai crossed in `id=i*4+j` order with `ff2n(2)`; quadratic Scheffe × [1,z1,z2], 18 columns, 52 candidates, n=24. No held-out inputs.
R 4.6.1, AlgDesign 1.2.1.2; verbatim sessionInfo, packageVersion and args in sessionInfo.txt.
Exact calls: for crit in c("D","A","I"), `set.seed(11); AlgDesign::optFederov(~ -1 + (x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3) + (x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3):(z1 + z2), data=candidates, nTrials=24, criterion=crit)`.
Local command: `PYTHONPATH=src py -3.14 -X utf8 reports/w11-mixproc/run.py`; TorchOED restarts=10, seed=11, CPU float64; shared F-based logdet / trace inverse / candidate-average I, relative tolerance 1e-10.

```
D MATCH TorchOED -39.71321948302598 R -39.71321948302635 R repeats False
A MATCH TorchOED 2133.8603436557964 R 2133.8603436558774 R repeats False
I TorchOED better TorchOED 0.7748268660712017 R 0.7809536418493048 R repeats False
```

Crossed order assertion PASS; three metrics computed; no exhaustive optimum or timing claim. D and A MATCH; I TorchOED better, reported as-is. R repeats no candidate in any criterion.
Sole pinned SAE 4129001 controller: `python3 <pinned>/colab_control.py --repo-path <linux-repo> --source-sha e5d47ac423c31a20521967f45a163b00d057661a --run-id w313-colab-20261118 --out <scratch>/colab --cli colab --device T4 --temp-root <scratch>/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`; preflight PASS.
Controller PASS, status/source_sha correct, call returncode=0, stdout/stderr empty; 52.4446s total. No retry. Observed Tesla T4, torch 2.14.1+cu130; R CPU with CUDA hidden.
Downloaded r-heldout.json SHA256 044d1e072e196f36fb6abeba06fd4d678322cb1ba8806d05509ebdb629b0b2b2 (external artifact).
Cleanup stop PASS, confirmed=true; authenticated assignments 0→0 and credits 50.87→50.87; no remote session remains. See runtime.json and cleanup.json.
Existing R cases and driver branches remain byte-unchanged; source/test/doc/matrix untouched. Nine evidence files. Matrix amendment deferred to the Architect's docs lane.
