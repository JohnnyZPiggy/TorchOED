# R oracle workflow probe

Refs #372. The rsm-only [probe](probe.R) copies the CCD/BBD calls and combine helper
from `tools/r_heldout.R` K6, leaving FrF2 and all held-out cases uncalled.
R is pinned to 4.5.2; `tools/r_packages.txt` freezes the installed oracle packages.
Outputs go to `R_ORACLE_OUT`; the probe asserts byte-identity for all three CSVs
against `reports/w10-rref` (CCD A/B/C and BBD k = 3, 5, 6, 7).

**BLOCKED (2026-11-19):**
`gh workflow run r-oracle.yml --repo JohnnyZPiggy/TorchOED --ref infra/372-r-oracle -f script=reports/r-oracle-probe/probe.R`
returned `HTTP 404: workflow r-oracle.yml not found on the default branch`.
[GitHub requires workflow_dispatch workflows on the default branch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
The workflow is new in this PR. No probe run, artifact, sessionInfo or CSV
byte-identity result exists yet; all are **NOT_RUN**. Architect ruling is needed
on staging this infrastructure before its first dispatch. The required green
probe and byte-identity DoD are not met; ordinary PR CI cannot substitute for them.
