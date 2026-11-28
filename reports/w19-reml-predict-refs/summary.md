# W19 REML prediction: frozen R references (#639)

One successful [r-oracle.yml run 46132567842](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/46132567842) at source `0ad7acd3f76d1d9067e0c99b297ef51cee266ca8`; script `reml_predict.R` (20 lines). Frozen input: `reports/w17-reml-check/Oats.csv`; no refetch.

R version 4.5.2 (2025-10-31); lme4 2.1-0, lmerTest 3.2-1, nlme 3.1-168, jsonlite 2.0.0. Exact session and full installed package versions are in the unchanged `sessionInfo.txt` and `runtime.json`. Runtime 1.7216 seconds.

L.csv uses R model.matrix coefficient order (header retained). Rows 1–12: variety Golden Rain, Marvellous, Victory (outer), nitro 0, 0.2, 0.4, 0.6 (inner); 13–15 variety means in that order; 16–19 nitro means in that order; 20 grand mean. Row labels also occur in r.json. Fixed effects, VarCorr variances, all 20 contest estimates/SE/df/95% CI ends, and 56 qt values (36 frozen grid + 20 contest df) are retained as emitted with digits=NA.

Cross-check against the rounded Architect values of §146 item 3; shared relative metric |R − reference| / max(1, |reference|). Reported as measured, without tuning.

| Rows | R SE range | SE relative difference range | R df range | df relative difference range |
|---|---|---|---|---|
| cells | 9.10701031487–9.10701031487 | 3.61644367121e-06–3.61644367238e-06 | 16.0815709085–16.0815709303 | 2.98512733357e-05–2.98526292386e-05 |
| variety means | 7.79757998492–7.79757998492 | 5.20740161112e-06–5.20740161237e-06 | 8.86875458695–8.86875459324 | 2.552793435e-05–2.55286432927e-05 |
| nitro means | 7.17475408246–7.17475408246 | 6.12045103837e-06–6.12045103837e-06 | 6.79188614911–6.79188615123 | 2.42708374775e-05–2.42711508718e-05 |
| grand mean | 6.64068031599–6.64068031599 | 7.26978823546e-06–7.26978823546e-06 | 4.99989944016–4.99989944016 | 2.01119683121e-05–2.01119683121e-05 |

The R numerical-Hessian differences are retained; no fit control or inputs changed. TorchOED code and tests do not read these files in this PR. No tests/ check is required for this evidence-only dispatch.

Artifact files L.csv, r.json, runtime.json and sessionInfo.txt are copied byte-for-byte from the downloaded artifact. CI on the evidence commit is reported in the PR handoff; this reference is consumed only by the later Studio follow-up.
