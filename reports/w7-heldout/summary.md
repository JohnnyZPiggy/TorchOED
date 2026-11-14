# Week 7 held-out H14–H16

Issue #209. Measurement source `9e0346866eea9df3a44d07670a4e6d2fe1a85724`. One local `python -B reports/w7-heldout/run.py` invocation on Windows CPU, Python 3.14.6, torch 2.15.0+cpu, float64, one thread, seed 20261114. Search wall time covers all ten restarts. H1–H13 retain the [W4](../w4-heldout/), [W5](../w5-heldout/) and [W6](../w6-heldout/) records.

| Case | Frozen result and required check |
|---|---|
| H14 | **PASS.** TorchOED logdet 18.98246610187344, dense error 0, plot A settings (−1, −1, 1, −1, 0, 1), max setting repeats 2, `LOCAL_SPLIT_PLOT_EXCHANGE` stop and certificate, 0.404167 s/restart. skpr **NOT_RUN**. The prescribed [two-stage script](../../tools/r_heldout.R) failed in `reduceRunMatrix` with “undefined columns selected.” |
| H15 | **PASS.** Objective and dense logdet −27.72588722239781, 3.55e−15 above the exact bound within tolerance. Selected set equals the {5, 2} lattice. `LOCAL_2_NUMERICAL` stop and certificate, 0.021810 s/restart. |
| H16 | **PASS.** All ten powers and noncentralities in [results](results.json). Dense inverse agreed with Cholesky. Equal-λ power maximum difference 4.44e−16 and SciPy 1.18.1 maximum absolute difference 6.66e−16. |

One Colab CPU `Rscript tools/r_heldout.R H14` invocation with [frozen B,C rows](H14.csv), R 4.6.1 and skpr 1.9.2. See [sessionInfo](sessionInfo.txt), [runtime](runtime.json) and [cleanup read-back](cleanup.json). The pinned controller used a T4 host with CUDA hidden from R. Stop passed and authenticated usage showed zero assignments.

measurement only; no tuning followed
