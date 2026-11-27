# W18 held-out H49–H51

All three frozen cases MATCH; each TorchOED case executed once at default settings. Later runs rescore saved values in runtime.json.

R oracle [45985746679](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45985746679), reference SHA 06c48423ca39fc21b236f252d509d41ec7edae4b; library 94361f75319744ccaf2ebc716d08feeb2e53b9b7. Versions: {'R': 'R version 4.5.2 (2025-10-31)', 'lme4': '2.1-0', 'lmerTest': '3.2-1', 'Python': '3.14.6', 'torch': '2.14.1+cpu'}.

- H49 MATCH: max relative vs exact 2.96221381e-13 (limit 1e-10). All variance components positive. R comparisons and worst quantities are preserved in results.json; reference limits 1e-8 fixed/logLik, 1e-4 SE/variance/df.
- H50 MATCH: max relative 2.11179165e-15 (limit 1e-8), worst ci.upr.
- H51 MATCH: D=0.77781745930520252, point=[1.0, 0.49999999994923644]; absolute D error 2.22044605e-16 (limit 1e-8), point max error 5.076356e-11 (limit 1e-5).

H49 column order check PASS: reconstruct treatment contrasts with recipe first, temperature second, interactions with recipe fastest; every matrix element equals R model.matrix. Cake replicate/recipe/temp/angle mapped to Block/Variety/nitro/yield for the unchanged imported exact scorer. Oats exact column regression PASS at 1e-12.

Required command: `PYTHONPATH=src python reports/w18-heldout/run.py`. Byte-identical results.json reproduction PASS (2.812 s including Python startup); first scoring/calls 2.942 s. CPU float64, one thread. No tuning, source/test edits or timing claim.
