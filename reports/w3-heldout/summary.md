# W3 held-out: one run, reported as-is

{"cpu": "AMD Ryzen 7 4800H with Radeon Graphics", "os": "Windows-11-10.0.26200-SP0", "python": "3.14.6", "torch": "2.15.0+cpu", "pydoe": "1.6.0", "numpy": "2.5.4", "scipy": "1.18.1", "threads": 1}
D = logdet(M); A/I = trace(M^-1 B). Cells: shared value / efficiency vs best found / median seconds (R quality only). D efficiency = exp((value-best)/p); A/I = best/value. PyDOE D-selected designs are also rescored for A/I.
AlgDesign: RUN (quality only); Julia NOT_RUN. R timing is never compared. Frozen-input oracles: {'H3': 'PASS m=100, identical coordinates', 'H4': 'PASS torch.equal float64 round-trip'}

| Case / criterion | TorchOED value / eff / s | certified floor / stop | sequential | fedorov | modified_fedorov | detmax | AlgDesign value / eff |
|---|---|---|---|---|---|---|---|
| H1 / D | 19.3041177 / 1 / 0.346439 | 0.775434243 / LOCAL_2_NUMERICAL | 16.6687356 / 0.768328267 / 0.0779451 | 19.3041177 / 1 / 0.619012 | 19.3041177 / 1 / 0.280371 | 19.3041177 / 1 / 0.26857 | 19.3041177 / 1 |
| H1 / A | 2.13055556 / 1 / 0.141804 | 0.652951192 / LOCAL_1_SWAP_NUMERICAL | 3.86492679 / 0.551253793 / 0.0779451 | 2.54553333 / 0.836978062 / 0.619012 | 2.54553333 / 0.836978062 / 0.280371 | 2.54553333 / 0.836978062 / 0.26857 | 2.13055556 / 1 |
| H1 / I | 0.528472222 / 1 / 0.152137 | 0.741444404 / LOCAL_1_SWAP_NUMERICAL | 0.856887229 / 0.616734856 / 0.0779451 | 0.589924921 / 0.895829628 / 0.619012 | 0.589924921 / 0.895829628 / 0.280371 | 0.589924921 / 0.895829628 / 0.26857 | 0.528472222 / 1 |
| H2 / D | 24.1792118 / 1 / 0.376422 | 0.673501239 / LOCAL_2_NUMERICAL | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | 24.1792118 / 1 |
| H2 / I | 0.804109229 / 1 / 0.10884 | 0.469630317 / LOCAL_1_SWAP_NUMERICAL | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | 0.804262869 / 0.999808969 |
| H3 / D | 7.53446872 / 1 / 0.28597 | 0.645408559 / LOCAL_2_NUMERICAL | 7.00485957 / 0.91551531 / 0.0327921 | 7.53446872 / 1 / 0.146322 | 7.53446872 / 1 / 0.0884675 | 7.53446872 / 1 / 0.0799371 | 7.53446872 / 1 |
| H3 / A | 2.49731693 / 1 / 0.0818794 | 0.541544938 / LOCAL_1_SWAP_NUMERICAL | 3.7126326 / 0.672653934 / 0.0327921 | 2.85256355 / 0.875464079 / 0.146322 | 2.85256355 / 0.875464079 / 0.0884675 | 2.85256355 / 0.875464079 / 0.0799371 | 2.49731693 / 1 |
| H3 / I | 0.418999965 / 1 / 0.0948132 | 0.616313509 / LOCAL_1_SWAP_NUMERICAL | 0.555341079 / 0.754491215 / 0.0327921 | 0.470887682 / 0.889808719 / 0.146322 | 0.470887682 / 0.889808719 / 0.0884675 | 0.470887682 / 0.889808719 / 0.0799371 | 0.419367394 / 0.999123849 |
| H4 / D | 38.9152331 / 1 / 2.13754 | 0.814802336 / LOCAL_2_NUMERICAL | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | 38.9152331 / 1 |

R D/A conversions (1e-9 relative), I ratio (conjecture only): [{"case": "H1", "criterion": "D", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}, {"case": "H1", "criterion": "A", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}, {"case": "H1", "criterion": "I", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": 0.9999999999999992}, {"case": "H2", "criterion": "D", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}, {"case": "H2", "criterion": "I", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": 0.9999999999999976}, {"case": "H3", "criterion": "D", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}, {"case": "H3", "criterion": "A", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}, {"case": "H3", "criterion": "I", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": 0.9999999999999998}, {"case": "H4", "criterion": "D", "DA": {"D": "PASS", "A": "PASS"}, "I_ratio": null}]
Colab cleanup PASS: owned session stopped; assignments before=0, after=0. Tesla T4 observed but unused for R; no GPU timings.

Rscript in Colab: R 4.6.1, AlgDesign 1.2.1.2, Ubuntu 24.04.4 LTS; exact sessionInfo in sessionInfo.txt. Frozen source 468d094; input files and both harnesses committed before either run. Quality is the third timed local call, not a best-of-three selection.
TorchOED ties AlgDesign on all D rows and H1/H3 A, and has slightly lower I loss on H2/H3. H1 D ties all three exchange PyDOE methods; H3 D does too, but takes 3.57744x the equally good detmax time (0.285970s vs 0.0799371s), missing the P1 3x time target there. No cross-environment R timing comparison or global-optimality claim.
