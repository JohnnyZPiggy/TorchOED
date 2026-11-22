# P2+ signature on the published AlgDesign example (#541)

Source: AlgDesign 1.2.1.2 `Federov.Rd` lines 127–131; R 4.6.1 reference designs frozen in [w12-published](../w12-published/). Candidate IDs are zero-based rows of `inputs/alg-d.csv`; quadratic p=10, m=27, n=14.

| Criterion / design | Shared value | Efficiency floor | certificate | stop_reason |
|---|---:|---:|---|---|
| D / published | 18.683477161182246 | 0.975053309925363 | NOT_CERTIFIED | NO_K2_NEIGHBORHOOD |
| D / TorchOED | 18.683477161182246 | 0.975053309925363 | LOCAL_2_NUMERICAL | LOCAL_2_NUMERICAL |
| A / published | 2.3 | 0.9289365080953955 | LOCAL_2_NUMERICAL | LOCAL_2_NUMERICAL |
| A / TorchOED | 2.4240924092409237 | 0.8813830530860193 | LOCAL_2_NUMERICAL | LOCAL_2_NUMERICAL |
| I / published | 0.7440667990372216 | 0.9252602210990462 | LOCAL_2_NUMERICAL | LOCAL_2_NUMERICAL |
| I / TorchOED | 0.7440667990372217 | 0.925260221099046 | LOCAL_2_NUMERICAL | LOCAL_2_NUMERICAL |

D is logdet(XᵀX), higher is better; A is tr((XᵀX)⁻¹), I is tr((XᵀX)⁻¹ FᵀF/27), lower is better. One shared dense function scores all six designs. D matches; A favors the published design; I differs by 1.1e-16. No calls or settings were adjusted after observing results.

All 14 published IDs were fixed, restarts=1, seed=0; TorchOED used seed=0 and all other defaults (10 restarts). A fixed-design numerical local label is recorded as returned; it does not establish optimality over unrestricted exchanges. The D fixed call reports NOT_CERTIFIED because no free k=2 neighborhood exists, while still reporting a dual efficiency floor. The dual floors quantify global efficiency, independently of the local label.

Independent dense p/max d bounds: D 0.830449826989619, A 0.892857142857143, I 0.613547439052458 (all within 1e-12); library D floor lies between its bound and 1. Certified ID sets equal reference ID sets for D/A/I. Required checks plus saved-metric recomputation: **13 PASS, 0 FAIL**.

Run: `PYTHONPATH=src python reports/w16-signature/run.py`; check saved evidence: append `--check`. Torch 2.15.0+cpu, Windows, AMD Ryzen 7 4800H with Radeon Graphics, float64, one thread. No R/Colab run or library changes. Results are frozen in `results.json`.
