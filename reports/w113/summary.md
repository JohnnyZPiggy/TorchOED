# W113 dual-floor measurement

Timed source: `a6ac83bc4b8debb6109350b3a85aa63db16a3c5f`; base `17840f7a036e209fce3b11440d8a7aa107e84e83`. Same-session Colab CPU bench_b0 pair; CUDA disabled, one thread, three calls/config, seeds 0..4.

PyDOE reference rows failed with `TypeError: inv() got an unexpected keyword argument assume_a`. PyDOE rows are NOT_RUN with that exact error. Architect ruling #113 comment 6669815005 fixes the baseline case set at eight cases (all except infeasible grid2 n=12); no new reference run is required.

| case | base s | head s | ratio | returned D floor |
|---|---|---|---|---|
| grid2 n=6 | 0.040229409 | 0.044724926 | 1.11174703 | 0.884845463 |
| grid2 n=7 | 0.037983687 | 0.038336503 | 1.00928862 | 0.945349757 |
| grid3 n=10 | 0.099431398 | 0.094312492 | 0.948518214 | 0.863045692 |
| grid3 n=20 | 0.134690504 | 0.102286997 | 0.759422483 | 0.97587468 |
| grid4 n=15 | 0.250288873 | 0.249585728 | 0.997190666 | 0.869899868 |
| grid4 n=30 | 0.327890959 | 0.319148214 | 0.973336426 | 0.986508409 |
| random500 n=10 | 0.631927512 | 0.68836941 | 1.08931704 | 0.917886785 |
| random500 n=20 | 1.51870643 | 1.57501526 | 1.03707684 | 0.981160728 |

D IDs/logdet/certificates/stop unchanged: 40/40.

{'python': '3.13.16', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39', 'cpu': 'x86_64', 'torch': '2.11.0+cu130', 'numpy': '2.4.6', 'scipy': '1.16.3', 'pydoe': '1.6.0', 'threads': 1}

grid3 n=20:
A: floor 0.588018530727 -> 0.939454560924; time 0.167396138 -> 0.254735071; ratio 1.52174999.
I: floor 0.51119350922 -> 0.95787637625; time 0.17342454 -> 0.271202669; ratio 1.56380792.

Overhead hypothesis <=10% fails on grid2 n=6 and A/I. No tuning or rerun. Controller PASS; cleanup confirmed, active assignments 0 -> 0. Architect ruling accepts measured overhead as-is; A/I overhead hypothesis FAIL. No tuning or rerun.

D floor medians across seeds 0..4, then across the eight fixed cases: 0.931618270859 (target >=0.90: PASS).
A/I floors >=0.75: PASS. Evidence source unchanged; no held-out inputs were used.
