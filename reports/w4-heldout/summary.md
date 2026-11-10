# W4 held-out record: one run, no tuning

Source `98e57442b4f2bda69bbf0fb6e3b510f0ab392880`; Issue #176. Frozen inputs vs `8b74cda`: PASS; src/tests/tools unchanged: PASS; reused reference rows unchanged: PASS.
{"cpu": "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD", "os": "Windows-11-10.0.26200-SP0", "python": "3.14.6", "torch": "2.15.0+cpu", "pydoe": "1.6.0", "numpy": "2.5.4", "scipy": "1.18.1", "threads": 1}
One call per row, restarts=1, seed=0, no warmup, 10800s subprocess wall cap including imports. Historical W3 driver used default restarts=10 and median of 3 calls; historical timings retained unchanged, current/W3 seconds ratio is descriptive only.
D = logdet M, efficiency exp((value-best)/p); A/I = trace(M^-1 B), efficiency best/value. Best includes current/W3 TorchOED and unchanged matching W3 references; quality vs W3 uses the same efficiency formula with W3 TorchOED as reference (>= 1-1e-9). Time PASS means <=3x fastest comparable PyDOE fedorov/modified_fedorov/detmax, excluding sequential. W3 per-restart seconds are its reported median divided by 10; raw and normalized ratios are both shown. Historical references are reused, not fresh same-session timings.
No PyDOE/R/Julia/Colab/GPU or bench_b0 run; no remote sessions. C1 remains closed, not met, 5/8.

| Row | value / best efficiency / seconds | floor / stop | quality vs W3 | current/W3 seconds (raw / per restart) | vs PyDOE / time | IDs vs W3 (ordered / set) |
|---|---|---|---|---|---|---|
| H1/D | 19.30411765 / 1 / 0.0439942 | 0.968347262 / LOCAL_2_NUMERICAL | PASS (1) | 0.151468x / 1.51468x | 0.167085x / PASS | DIFF / DIFF |
| H1/A | 2.200737449 / 0.9681098292 / 0.164502 | 0.906130501 / LOCAL_2_NUMERICAL | FAIL (0.9681098292) | 0.0114464x / 0.114464x | 0.624757x / PASS | DIFF / DIFF |
| H1/I | 0.5397847834 / 0.97904246 / 0.0773498 | 0.934063556 / LOCAL_2_NUMERICAL | FAIL (0.97904246) | 0.00418343x / 0.0418343x | 0.293765x / PASS | DIFF / DIFF |
| H2/D | 23.85622968 / 0.9734438125 / 0.0434865 | 0.930685482 / LOCAL_2_NUMERICAL | FAIL (0.9734438125) | 0.129812x / 1.29812x | N/A | DIFF / DIFF |
| H2/I | 0.8058332053 / 0.997584977 / 0.187334 | 0.902562234 / LOCAL_2_NUMERICAL | FAIL (0.997584977) | 0.0140812x / 0.140812x | N/A | DIFF / DIFF |
| H3/D | 7.534468718 / 1 / 0.103561 | 0.939630359 / LOCAL_2_NUMERICAL | PASS (1) | 0.347229x / 3.47229x | 1.29393x / PASS | DIFF / same |
| H3/A | 2.497316925 / 1 / 0.0598347 | 0.935548108 / LOCAL_2_NUMERICAL | PASS (1) | 0.0153459x / 0.153459x | 0.747593x / PASS | DIFF / same |
| H3/I | 0.4195703727 / 0.9986404965 / 0.0781841 | 0.949659832 / LOCAL_2_NUMERICAL | FAIL (0.9986404965) | 0.0194097x / 0.194097x | 0.976856x / PASS | DIFF / DIFF |
| H4/D | 38.91523306 / 1 / 1.08873 | 0.986015902 / LOCAL_2_NUMERICAL | PASS (1) | 0.543569x / 5.43569x | N/A | DIFF / same |
| H5/D | 40.00864016 / 0.9958151697 / 1.41125 | 0.982396755 / LOCAL_2_NUMERICAL | FAIL (0.9958151697) | 0.164248x / 1.64248x | 0.0532997x / PASS | DIFF / DIFF |
| H6/D | 32.94345199 / 0.9982106567 / 0.0334619 | 0.920052563 / LOCAL_2_NUMERICAL | FAIL (0.9982106567) | 0.264939x / 2.64939x | N/A | DIFF / DIFF |
| H6/I | 0.8976870452 / 1 / 0.222339 | 0.785473814 / LOCAL_2_NUMERICAL | PASS (1) | 0.04164x / 0.4164x | N/A | DIFF / DIFF |
| H7/I | 0.5926611735 / 1 / 0.140213 | 0.933970854 / LOCAL_2_NUMERICAL | PASS (1) | 0.015146x / 0.15146x | N/A | DIFF / DIFF |
| H7/D | 15.12734589 / 0.9986641946 / 0.214694 | 0.970208612 / LOCAL_2_NUMERICAL | FAIL (0.9986641946) | 0.729704x / 7.29704x | N/A | DIFF / DIFF |
| H8/D | 17.74625278 / 0.9999292595 / 1.04504 | 0.967819589 / LOCAL_2_NUMERICAL | FAIL (0.9999292595) | 0.157521x / 1.57521x | 0.224837x / PASS | DIFF / DIFF |
| H8/A | 2.452011438 / 0.9969479608 / 527.73 | 0.943977875 / LOCAL_2_NUMERICAL | FAIL (0.9969479608) | 0.0799393x / 0.799393x | 113.539x / FAIL | DIFF / DIFF |
| H8/I | 0.3887281372 / 0.994480595 / 427.012 | 0.950397008 / LOCAL_2_NUMERICAL | FAIL (0.994480595) | 0.0654357x / 0.654357x | 91.8699x / FAIL | DIFF / DIFF |

Quality vs W3: 6/17 PASS, 11 FAIL, 0 TIMEOUT. Time vs reused PyDOE: 8/10 PASS (historical W3 recorded: 3/10, ten restarts/call). All changed designs are marked DIFF above; exact IDs and reused PyDOE/AlgDesign columns are in results.json. No optimization followed these measurements.
