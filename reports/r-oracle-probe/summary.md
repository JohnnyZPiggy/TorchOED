# R oracle workflow probe

Refs #372 stage 2. [Run 44500473486](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44500473486) **SUCCESS**.
Source: `00ab71db3c2d85d2740a10b4e33cde433fd94686` (main, stage 1 merged).
Command: `gh workflow run r-oracle.yml --repo JohnnyZPiggy/TorchOED --ref main -f script=reports/r-oracle-probe/probe.R`.
Downloaded with `gh run download 44500473486 --repo JohnnyZPiggy/TorchOED`.

R 4.5.2 (2025-10-31), Ubuntu 24.04.5 LTS; rsm 2.10.6, estimability 1.5.1,
jsonlite 2.0.0, remotes 2.5.0. Exact installed versions: [runtime.json](runtime.json);
[sessionInfo.txt](sessionInfo.txt) records the separate metadata process.
Reference-script wall time: 0.166 s (excludes runner setup and package installation).

| Regenerated artifact vs committed `reports/w10-rref` | Result |
|---|---|
| `rsm_ccd.csv` (CCD A/B/C) | byte-identical PASS |
| `rsm_bbd.csv` (BBD k = 3, 5) | byte-identical PASS |
| `rsm_bbd67.csv` (BBD k = 6, 7) | byte-identical PASS |

The [probe](probe.R) asserts raw-byte equality on the runner; the downloaded CSVs
also matched `git show HEAD:reports/w10-rref/<name>` locally (3 PASS, 0 FAIL).
The R version differs from the original 4.6.1; all CSV bytes match, so no pin change
or metric-only fallback was needed. Duplicate CSVs remain in the run artifact.
The stage-1 dispatch 404 is resolved by dispatching from main. No workflow or
package-list fix, FrF2 call, held-out case, GPU or Colab session was needed.
