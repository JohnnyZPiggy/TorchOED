# W13 F1/F2 R references (#404)

One R oracle run [44641039986](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44641039986), source `69e888a5ce6d8a4dee44da0cb8ffca29961f8891`, after base `e85e8ea33529339f39e77a4d9043c4cb987f85c2`.
R 4.5.2; lhs 1.3.0, MaxPro CRAN 4.1-2 (`packageVersion`: 4.1.2), DiceDesign 1.10. Full runtime/session and raw outputs: `sessionInfo.txt` (JSON first line, session text thereafter).
Four CSVs frozen once at the resumed base; n=8, k=3, seed=0, default settings. No settings/seed tuning or second dispatch.

- Diagnostics: 16/16 comparisons MATCH. DiceDesign DisC2 = CD2 and MaxProMeasure = psi (both power mappings 1); no squaring adjustment. Maximum CD2 absolute difference across the four CSVs: 1.37e-15; tolerance 1e-12.
- Quality maximin: TorchOED phi15 2.0732525070396077 vs rank-midpoint R 3.277905052016956: TorchOED better.
- Quality MaxPro: TorchOED psi 12.246017233073282 vs rank-midpoint R 11.119922570622206: NO MATCH, delta +1.1260946624510755. Recorded as-is; F1 Ext remains partial.
- Rd examples: lhs maximinLHS.Rd line 74 (`Rd2ex` line 9), 4 x 3, TorchOED better (phi15 1.6452337368527556 vs 2.310300628135955).
- Rd examples: MaxPro MaxProLHD.Rd line 65 (`Rd2ex` line 10), 10 x 4, TorchOED better (psi 13.227729408335211 vs 13.43815350925327).
- Rd examples: DiceDesign discrepancyCriteria.Rd line 68 (`Rd2ex` line 11), 40 x 2, MATCH (CD2 0.1479244295131235 vs 0.14792442951311552, delta 7.99e-15).
- Published-example count: 11 -> 14 (three new matches / better). The separate 8 x 3 MaxPro quality NO MATCH is not a published Rd example.

Original Rd source lines count LF lines in the pinned CRAN source archives; full installed-source `Rd2ex` text and executed calls are frozen in the raw JSON. Each first call ran verbatim under set.seed(20261119); DiceDesign includes its preceding design construction. No held-out size was run.
Reproduce: `PYTHONPATH=src python reports/w13-spacefill-ext/score.py`; every numeric comparison uses TorchOED's shared `space_filling_diagnostics`. R quality columns are ranked then mapped to midpoints.
Scope: ten report files, three dev-only R package pins, one reports index row, two matrix Ext cells. No production dependencies or source/test/website edits; no Colab, GPU, held-out or B0 run.
