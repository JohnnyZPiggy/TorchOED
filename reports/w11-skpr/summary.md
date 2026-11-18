# W11 skpr nested-block probe (#293)

- **Reference NOT_RUN.** One pinned-controller Colab CPU-R run at `c855f36ce13f6eb951b084ab3a22ef14daa8bd49`; non-held-out K11 inputs frozen in `tools/r_heldout.R`, seed 20261120. No second run or repaired call.
- R 4.6.1 / skpr 1.9.2 on Ubuntu 24.04.4 LTS; exact dependency versions, installed `args(gen_design)` and `formals(gen_design)$varianceratio` in `sessionInfo.txt`. The default variance ratio is 1. skpr installation from https://cloud.r-project.org succeeded; only skpr was passed to `install.packages` (no base packages).
- Observed host Tesla T4; all R work CPU with CUDA hidden. Controller transport PASS in 800.0143 seconds (includes setup/run/cleanup; no speed claim). Authenticated assignments 0→0, credits 51.06→50.89; stop PASS and cleanup_confirmed true. No remote remains. Download SHA-256 and size in runtime.json; controller readback in cleanup.json.
- Both requested calls were attempted separately, each inside its own tryCatch. The htc call failed; the full call was then attempted with htc=NULL and failed too. The second error therefore does not independently establish whether a valid nested whole-plot design would be supported. This is the exact dispatched probe's result, not a general unsupported-feature claim.
- No skpr.csv or rownames exist, so levels, distinct whole plots, blocks and runs per whole plot are NOT_RUN. Frozen requested inputs: W={−1,0,1}, X1/X2={−1,1}, 6 hard-to-change runs/blocksizes=3, 24 full runs/blocksizes=4. No structure inferred from unsuccessful calls.
- **TorchOED re-scoring: NOT_RUN (pending the D2 gate's combined information).** No score or tolerance chosen. Local `run.py` prints NOT_RUN because the reference design does not exist. Final-head CI is recorded in the PR handoff.
- Six report files; 36 net tools lines (S50). Existing M11/K4/K5 and all other branches unchanged. No src/tests/docs/website/matrix edits, held-out inputs, AlgDesign, mixexp or GPU computation.

## Verbatim call warnings and errors

```text
htc warning: first element used of 'length.out' argument 
htc error: argument must be coercible to non-negative integer 
full warning: first element used of 'length.out' argument 
full error: argument must be coercible to non-negative integer 
```
