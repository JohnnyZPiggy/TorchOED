# W11 mixexp lattice/centroid reference — #317

Measured source `e7d3a21296552813b663437f2b7a9ffdede95940`; base `854298ac84a8231a2571124bca7476083dfe4249`.
Before allocation, `git ls-remote origin refs/heads/evidence/317-mixture-lattice` returned `e7d3a21296552813b663437f2b7a9ffdede95940\trefs/heads/evidence/317-mixture-lattice`.
Exact planned calls: `mixexp::SLD(3,2)`, `SLD(3,3)`, `SLD(4,2)`, `SLD(4,3)`, `mixexp::SCD(3)`, `SCD(4)`, `SCD(5)`.
Sole controller run PASS (transport/source/status); scientific reference NOT_RUN: CRAN installation failed before constructor calls. No retry or substitute source. No mixexp.csv produced; sessionInfo.txt contains the verbatim installation error. R and mixexp versions NOT_RECORDED because setup exited before version capture; no match claim.

| Case | Verdict | Reason |
|---|---|---|
| L32 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| L33 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| L42 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| L43 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| C3 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| C4 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |
| C5 | NOT_RUN | mixexp unavailable for this R version; requireNamespace failed (verbatim error in sessionInfo.txt) |

Command: `PYTHONPATH=src py -3.14 -X utf8 reports/w11-mixlat/run.py`; seven NOT_RUN verdict lines, exit 0. The checker uses lexicographic rounded-1e-9 row matching, max absolute deviation tolerance 1e-12 and row-sum tolerance 1e-12 when outputs exist.
Pinned SAE `4129001fffcb336f5203e4b0d6765d7a0ea3150c`: `python3 <pinned>/colab_control.py --repo-path <linux-repo> --source-sha e7d3a21296552813b663437f2b7a9ffdede95940 --run-id w317-colab-20261118 --out <scratch>/colab --cli colab --device T4 --temp-root <scratch>/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 300 --work-timeout 1500 --cleanup-seconds 180`.
Preflight: first CLI_UNAVAILABLE from non-login PATH; next LOCAL_OPERATION_FAILED because output directory existed; login-shell/fresh-output preflight PASS. No sessions allocated by these local failures.
Controller 70.7471s PASS; observed Tesla T4, torch 2.14.1+cu130. R CPU with CUDA hidden. Cleanup stop PASS, confirmed=true; authenticated assignments 0→0, credits 50.87→50.87; no remote session remains. See cleanup.json.
Downloaded r-heldout.json SHA256 ffec9a23b0d091e911fcfc940757ca962253c089ba90e2425f104bbd21a2dcbc.
Final checker differs from measured checker only by explicit UTF-8 and single-line error representation. Existing R cases and driver branches byte-unchanged (2 PASS). Five report files, 47 code lines total; matrix deferred to Architect. No held-out input, timing or optimum claim.
