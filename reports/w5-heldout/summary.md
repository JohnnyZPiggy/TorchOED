# W5 held-out H9/H10: four rows, as-is
Source `e4d3d2f84448ccc43ed2a8c3d6011767f8217b32`; environment {"python": "3.14.6", "torch": "2.15.0+cpu", "os": "Windows-11-10.0.26200-SP0", "cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD", "threads": 1}; CPU float64, one call/row, defaults (10 restarts), seed 20261110, no warmup; 10800s subprocess cap/row (includes imports), 0 TIMEOUT. A/I API objective is negative trace; dense value below is positive trace. Certificates concern only the linearized problem at the nominal parameters.
H1–H8 reused unchanged: [W4 record](../w4-heldout/summary.md). Command: `python scratch/run_once.py` (four subprocesses of `python reports/w5-heldout/run.py CASE CRITERION`, each timeout=10800s); final-SHA oracle-only check: `python reports/w5-heldout/run.py --check`. 12/12 oracle assertions PASS, 0 FAIL; no src/tests changes, no tuning or external compute.
| Row | IDs / x | API objective / dense value | returned floor / KW floor | stop | wall s / per-restart s |
|---|---|---|---|---|---|
| H9/D | [43, 41, 42, 400, 0, 1, 399, 2, 398] / [0.43, 0.41000000000000003, 0.42, 4.0, 0.0, 0.01, 3.99, 0.02, 3.98] | 2.46480684383 / 2.46480684383 | 0.959205983145 / 0.8752185617157626 | LOCAL_2_NUMERICAL | 0.789174 / 0.0789174 |
| H9/I | [0, 50, 51, 52, 400, 399, 397, 53, 398] / [0.0, 0.5, 0.51, 0.52, 4.0, 3.99, 3.97, 0.53, 3.98] | -0.211531608985 / 0.211531608985 | 0.951113610116 / N/A | LOCAL_2_NUMERICAL | 6.0376291 / 0.60376291 |
| H10/D | [66, 68, 67, 0, 1, 600, 599, 598] / [0.66, 0.68, 0.67, 0.0, 0.01, 6.0, 5.99, 5.98] | 0.0458816452915 / 0.0458816452915 | 0.972794369485 / 0.7279444999798731 | LOCAL_2_NUMERICAL | 2.9065776 / 0.29065776 |
| H10/A | [66, 599, 1, 67, 0, 600, 65, 598] / [0.66, 5.99, 0.01, 0.67, 0.0, 6.0, 0.65, 5.98] | -10.2243465386 / 10.2243465386 | 0.679109775479 / N/A | LOCAL_2_NUMERICAL | 55.4794754 / 5.54794754 |
measurement only; no tuning followed
