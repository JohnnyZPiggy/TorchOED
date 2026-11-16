# W8-K4 published skpr split-plot comparison

**PASS: TorchOED = skpr (1 published example compared).** Both dense GLS log determinants were 13.513636250491656 (Δ = 0). The TorchOED objective matched its dense score at the required 1e-9 relative tolerance. Both designs had full model rank, distinct candidate IDs and a single Altitude setting per plot.

R 4.6.1 and skpr 1.9.2 ([sessionInfo](sessionInfo.txt)) ran the [published example, lines 217–230](https://github.com/cran/skpr/blob/1.9.2/man/gen_design.Rd#L217-L230), matching installed Rd2ex lines 88–101, with seed 20261118 at source SHA 25fe385b233dee5c73d4ec8d5a753926a39bd784. There were 33 runs in 11 plots of size 3, with η = 1. The five model columns were intercept, Altitude, Range(Close), Range(Far) and Power, with Medium as the effects-coding reference.

The single TorchOED search used seed 0, CPU float64 and one thread (Python 3.14.6, torch 2.14.1+cpu). The [candidate CSV](candidates.csv) was replicated 11 times. [Results](results.json) record both scores and selected IDs. T4 host, CUDA hidden from R; GPU unused; no GPU claim. [Architect ruling](https://github.com/JohnnyZPiggy/TorchOED/issues/217#issuecomment-6920592918).

The initial T4 allocation timed out (2026-11-16 15:57:13 UTC, ALLOCATION_FAILED, created false). The [authorized retry](https://github.com/JohnnyZPiggy/TorchOED/issues/217#issuecomment-6922142785) started at 17:59:48 UTC, created the sole session and passed. Both stops passed with authenticated assignment counts 0 → 0. [Cleanup read-back](cleanup.json).
