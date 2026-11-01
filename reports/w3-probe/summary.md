# W3a R AlgDesign probe — PASS

Source `3fc9ad8186e6de1afffe4ae468bbe9f6ee0c6c92`; base `19e68fe`; Issue #86; seed20261101; one probe run.
Runtime: preinstalled Rscript in the Colab Python runtime (no apt install); R4.6.1 (2026-06-24), AlgDesign1.2.1.2 installed from CRAN. Exact sessionInfo() and packageVersion() output: sessionInfo.txt; R --version: runtime.json.
Observed device: Tesla T4,15360MiB,driver580.82.07. accelerator allocated only because the pinned controller has no CPU runtime; unused; no GPU claim
Command: `Rscript tools/r_probe.R` via <=15-line r_probe_driver.py and pinned SAE controller5fb26304e4c85183aa215e6ad4db9ced928cab12, --device T4 --task-seconds300. R child CUDA_VISIBLE_DEVICES is empty.
Frozen input: cand_3x3.csv,27 rows in gen.factorial(3,3) order (A varies fastest); quadratic model p10,14 trials,5 repeats.
Chosen rows (1-based): 1, 2, 3, 6, 7, 9, 12, 13, 17, 19, 21, 23, 25, 27.
AlgDesign reported D=0.4626846245365015, A=3.806250000000003, I=null (default D call does not return I); Ge=0.83, Dea=0.815. No additional A/I optimization was run.
TorchOED shared d_optimality on CPU float64: logdet=18.683477161182246; exp((logdet-10*log(14))/10)=0.4626846245365015; relative difference=0 <=1e-9: PASS.
Required local check: `PYTHONPATH=src python -B tools/r_probe_rescore.py` —1/1 oracle assertion PASS; no new pytest,src,tests or dependency changes.
Cleanup: controller stop PASS; authenticated assignments0 before/0 after; cleanup_confirmed=true. Downloaded external artifact SHA256/size are in runtime.json; no tracked files hashed.
Initial CPU request was BLOCKED with UNSUPPORTED_DEVICE before allocation; Architect then authorized this single T4-host-CPU run. No fallback,new spending,pin upgrade or controller patch.
Julia NOT_RUN (optional; not probed). Held-out H1–H4 not run or used for tuning. Frozen inputs/outputs and exact versions retained for reproduction.
