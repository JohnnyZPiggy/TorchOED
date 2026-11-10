# W5 scale crossover

Measured code SHA: 0505d67f36aea5ccbd359a56428c2d0757b9c37c; cleanup stop PASS, assignments 0 -> 0, balance 57.98 -> 56.77.
One Colab host; float64, candidate/initialization seed 0, n=2p, one restart; 1800s subprocess wall cap.
One untimed CUDA warmup; transfers included; CUDA synchronized before timer stop. Search timing excludes shared rescoring.
floor is shared KW; torchoed_floor is the refined returned bound. No retries or tuning. 30 protocol entries: 6 PASS (18/18 oracles), 24 ERROR, 0 TIMEOUT. Errors: 15 PyDOE SciPy inv(assume_a); 4 Torch workspace; 5 k=5,m=100000 feature-output budget failures before any method runs. No finished PyDOE result; crossover comparison is inconclusive, not measured quality losses.
Environment: {"python": "3.13.16", "torch": "2.14.1+cu130", "pydoe": "1.6.0", "numpy": "2.4.6", "scipy": "1.16.3", "threads": 1, "cuda": "13.0", "gpu": "Tesla T4", "driver": "580.82.07", "cpu": "model name\t: Intel(R) Xeon(R) CPU @ 2.00GHz"}

| k | m | method | seconds/status | logdet | floor | torchoed_floor | repeats |
|---|---|---|---|---|---|---|---|
| 3 | 1000 | fedorov | ERROR |  |  |  |  |
| 3 | 1000 | modified_fedorov | ERROR |  |  |  |  |
| 3 | 1000 | detmax | ERROR |  |  |  |  |
| 3 | 1000 | cpu | 0.6709243720000586 | 20.009800061023043 | 0.7894658458649016 | 0.9722818445351031 | 0 |
| 3 | 1000 | cuda | 1.0751041999999416 | 20.009800061023043 | 0.7894658458649016 | 0.9722818445351031 | 0 |
| 5 | 1000 | fedorov | ERROR |  |  |  |  |
| 5 | 1000 | modified_fedorov | ERROR |  |  |  |  |
| 5 | 1000 | detmax | ERROR |  |  |  |  |
| 5 | 1000 | cpu | 0.7709407610000198 | 52.32556781244542 | 0.7948867354218425 | 0.9804275960797801 | 0 |
| 5 | 1000 | cuda | 1.5362649930000316 | 52.32556781244542 | 0.7948867354218425 | 0.9804275960797801 | 0 |
| 3 | 10000 | fedorov | ERROR |  |  |  |  |
| 3 | 10000 | modified_fedorov | ERROR |  |  |  |  |
| 3 | 10000 | detmax | ERROR |  |  |  |  |
| 3 | 10000 | cpu | 8.695636020999927 | 21.28435214766783 | 0.7323964957074166 | 0.9723462460858325 | 0 |
| 3 | 10000 | cuda | 11.135963982000021 | 21.28435214766783 | 0.7323964957074166 | 0.9723462460858325 | 0 |
| 5 | 10000 | fedorov | ERROR |  |  |  |  |
| 5 | 10000 | modified_fedorov | ERROR |  |  |  |  |
| 5 | 10000 | detmax | ERROR |  |  |  |  |
| 5 | 10000 | cpu | ERROR |  |  |  |  |
| 5 | 10000 | cuda | ERROR |  |  |  |  |
| 3 | 100000 | fedorov | ERROR |  |  |  |  |
| 3 | 100000 | modified_fedorov | ERROR |  |  |  |  |
| 3 | 100000 | detmax | ERROR |  |  |  |  |
| 3 | 100000 | cpu | ERROR |  |  |  |  |
| 3 | 100000 | cuda | ERROR |  |  |  |  |
| 5 | 100000 | fedorov | ERROR |  |  |  |  |
| 5 | 100000 | modified_fedorov | ERROR |  |  |  |  |
| 5 | 100000 | detmax | ERROR |  |  |  |  |
| 5 | 100000 | cpu | ERROR |  |  |  |  |
| 5 | 100000 | cuda | ERROR |  |  |  |  |
no crossover found (k=3, cpu)
no crossover found (k=3, cuda)
no crossover found (k=5, cpu)
no crossover found (k=5, cuda)
measurement only; no claim beyond these 6 cells
