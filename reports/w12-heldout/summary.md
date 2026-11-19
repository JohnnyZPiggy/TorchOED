# W12 frozen held-out H30–H32 (#368)

CPU float64, Python 3.14.6, torch 2.15.0+cpu, one thread; seed 0, default 10 restarts. Searches ran once. `run.py --score` re-scores saved designs only; `--references` attaches downloaded references without repeating searches. Copy mappings: H30 `8*i+k`, H32 `3*i+k`. Inputs and R pins committed at `949bae4ae9070e6c663c178151d076b17dd5c4b0` before dispatch. No production numerical source changed.

| Case / shared metric | TorchOED | Reference | Δ (TorchOED − reference) | Verdict |
|---|---|---|---|---|
| H30 exact A trace | 1797/2 = 898.5; floor 0.6162551440329218 | Fraction exhaustive 1797/2; same floor | 0 | MATCH |
| H31 PB28 | XᵀX = 28 I; absolute J₃ histogram {4: 2574, 12: 351}, 2925 triples | same exact orthogonality and J3 histogram | 0 | MATCH |
| H32 blocked D, det(M_B)^(1/6) | 3.5728152202037835; logdet 7.640123172695363; floor 0.9140932828990237 | 3.5725288795676673; logdet 7.639642287858012; floor 0.9140053724066399 | +0.0002863406361162113 | NO MATCH |

H30 visited all 3003 multisets, 2996 singular, 7 nonsingular; unique optimum multiset (0,1,2,3,4,5,6,6). All optimal multisets and selected copy IDs are in `results.json`. H30 rank-7 / float64-vs-Fraction trace, H31 int64 einsum-vs-Python triple-loop, H32 blocked-vs-augmented determinant: **5 PASS, 0 FAIL** including both R-side metric checks.

Initial [R workflow run 44518822770](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44518822770), source `949bae4ae9070e6c663c178151d076b17dd5c4b0`, failed at package installation immediately after `* DONE (FrF2)`: `Error: as.character(packageVersion(pins$package[i])) == pins$version[i] is not TRUE` (exit 1). Reference/runtime step skipped. Artifact upload: `No files were found with the provided path: /home/runner/work/_temp/r-oracle/. No artifacts will be uploaded.` Download: `no valid artifacts found to download`. Neither R call ran. The Architect authorised one re-dispatch after the separate #382 workflow fix merged.

After merging main `9a36ad163556f0c0c8bb448e63e5cea08214d5fc` into the task branch, the single authorised [re-dispatch 44536661878](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44536661878) at `4b5324bdfa43c2630c00cd4993e71bd73759be61` succeeded. Observed: R 4.5.2, FrF2 2.3-5, DoE.base 1.2-5, AlgDesign 1.2.1.2 on Ubuntu 24.04.5; R script wall time 0.5854 s. The artifact's H31.csv, H32.csv, sessionInfo.txt and runtime.json are committed byte-identically. Shared metrics re-scored the CSVs; all saved TorchOED rows and H30 enumeration remain unchanged. No search or exhaustive rerun, pin/input change, tuning or further dispatch. No Colab session; cleanup not applicable. H31 uses the same histogram shown in the table; H32 uses all grid points at most three times, so its fixed-design floor is recorded.

measurement only; no tuning followed
