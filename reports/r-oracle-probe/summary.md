# R oracle workflow probe

Refs #372. The rsm-only [probe](probe.R) copies the CCD/BBD calls and combine helper
from `tools/r_heldout.R` K6, leaving FrF2 and all held-out cases uncalled.
R is pinned to 4.5.2; `tools/r_packages.txt` freezes the installed oracle packages.
Outputs go to `R_ORACLE_OUT`; the probe asserts byte-identity for all three CSVs
against `reports/w10-rref` (CCD A/B/C and BBD k = 3, 5, 6, 7).

Probe dispatch and downloaded sessionInfo: pending.
