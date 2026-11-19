# W12 frozen held-out H30–H32 (#368)

CPU float64, Python 3.14.6, torch 2.15.0+cpu, one thread; seed 0, default 10 restarts. Searches ran once. `run.py --score` re-scores saved designs only; `--references` attaches downloaded references without repeating searches. Copy mappings: H30 `8*i+k`, H32 `3*i+k`. Inputs and R pins committed at `949bae4ae9070e6c663c178151d076b17dd5c4b0` before dispatch. No production numerical source changed.

| Case / shared metric | TorchOED | Reference | Δ (TorchOED − reference) | Verdict |
|---|---|---|---|---|
| H30 exact A trace | 1797/2 = 898.5; floor 0.6162551440329218 | Fraction exhaustive 1797/2; same floor | 0 | MATCH |
| H31 PB28 | XᵀX = 28 I; absolute J₃ histogram {4: 2574, 12: 351}, 2925 triples | NOT_RUN | N/A | NOT_RUN |
| H32 blocked D, det(M_B)^(1/6) | 3.5728152202037835; logdet 7.640123172695363; floor 0.9140932828990237 | NOT_RUN | N/A | NOT_RUN |

H30 visited all 3003 multisets, 2996 singular, 7 nonsingular; unique optimum multiset (0,1,2,3,4,5,6,6). All optimal multisets and selected copy IDs are in `results.json`. H30 rank-7 / float64-vs-Fraction trace, H31 int64 einsum-vs-Python triple-loop, H32 blocked-vs-augmented determinant: **3 PASS, 0 FAIL**. The two R-side metric checks are **NOT_RUN**.

Exactly one [R workflow run 44518822770](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44518822770), source `949bae4ae9070e6c663c178151d076b17dd5c4b0`, failed at package installation immediately after `* DONE (FrF2)`: `Error: as.character(packageVersion(pins$package[i])) == pins$version[i] is not TRUE` (exit 1). Reference/runtime step skipped. Artifact upload: `No files were found with the provided path: /home/runner/work/_temp/r-oracle/. No artifacts will be uploaded.` Download: `no valid artifacts found to download`. Workflow-level **BLOCKED**; neither R call ran. No retry or workflow repair.

Requested pins: FrF2 2.3-5, AlgDesign 1.2.1.2; workflow requests R 4.5.2. **Observed interpreter/package versions (including DoE.base): NOT_RUN** — no `sessionInfo.txt` or `runtime.json` was produced. No CSV artifact exists to freeze. These missing files and values are not fabricated. No Colab session or cleanup was needed.

measurement only; no tuning followed
