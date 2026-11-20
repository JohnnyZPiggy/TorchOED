# Week 14 paper references — NR

Access checked 2026-11-20. B1 and B2 are **NR**; published MATCH-or-better count stays **16**, with no additions. No scientific call, design transcription or shared-metric comparison was performed. Missing scores and deltas are null, not zero.

| Goal | Chosen source | Outcome |
|---|---|---|
| B1 Bayesian D | DuMouchel, W. and Jones, B. (1994), *Technometrics* 36(1), 37–47, [DOI](https://doi.org/10.1080/00401706.1994.10485399) | NR: abstract accessible, full design table and complete settings not verified verbatim |
| B2 model-robust D | Li, W. and Nachtsheim, C. J. (2000), *Technometrics* 42(4), 345–352, [DOI](https://doi.org/10.1080/00401706.2000.10485707) | NR: abstract accessible, full design table and model set not verified verbatim |

The web tool returned `Internal Error` / `URL ... is not accessible via this tool.` for both publisher full-text endpoints:

- https://www.tandfonline.com/doi/epdf/10.1080/00401706.1994.10485399?needAccess=true
- https://www.tandfonline.com/doi/epdf/10.1080/00401706.2000.10485707?needAccess=true

Table/figure number and specific table page: **NOT_VERIFIED** for both goals (the page ranges above are bibliographic only). Search snippets and third-party summaries were not used to reconstruct design rows or model settings. The Issue explicitly permits NR when tables cannot be accessed verbatim. No alternate example was selected.

`PYTHONUTF8=1 python reports/w14-paper-ext/run.py` reproduces `results.json` and prints the recorded NR outcomes; it does not retry network access or claim a numerical re-score. No input CSV is supplied because no published design was transcribed. If a supported example becomes available, freeze its exact inputs before scoring with the Issue's predeclared tolerance, 1e-9 × max(1, |published|). No R/Julia/Colab or held-out work.
