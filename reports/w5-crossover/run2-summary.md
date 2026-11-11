# W5 scale crossover run 2

Measured code SHA: 410b9c2c68699068b321820afeba17ff1dbcc057
One Colab host; float64, candidate/initialization seed 0, n=2p, one restart; 1800s subprocess wall cap.
One untimed CUDA warmup; transfers included; CUDA synchronized before timer stop. Search timing excludes shared rescoring.
floor is shared KW; torchoed_floor is the refined returned bound. No retries or tuning.
Environment: {"python": "3.13.16", "torch": "2.14.1+cu130", "pydoe": "1.6.0", "numpy": "2.4.6", "scipy": "1.18.1", "threads": 1, "cuda": "13.0", "gpu": "Tesla T4", "driver": "580.82.07", "cpu": "model name\t: Intel(R) Xeon(R) CPU @ 2.00GHz"}

| k | m | method | seconds/status | logdet | floor | torchoed_floor | repeats |
|---|---|---|---|---|---|---|---|
| 3 | 1000 | fedorov | 9.004539572999988 | 19.99210978218451 | 0.6503579231444785 |  | 0 |
| 3 | 1000 | modified_fedorov | 4.506773349999975 | 19.917102261222833 | 0.6704456864466277 |  | 0 |
| 3 | 1000 | detmax | 8.18615834800002 | 19.99210978218451 | 0.6503579231444783 |  | 0 |
| 3 | 1000 | cpu | 0.6837562030000015 | 20.009800061023043 | 0.7894658458649016 | 0.9722818445351031 | 0 |
| 3 | 1000 | cuda | 1.023741249000011 | 20.009800061023043 | 0.7894658458649016 | 0.9722818445351031 | 0 |
| 5 | 1000 | fedorov | 7.331228897999978 | 49.62078438346606 | 0.5064135493240711 |  | 0 |
| 5 | 1000 | modified_fedorov | 8.744281516000001 | 49.62078438346606 | 0.5064135493240711 |  | 0 |
| 5 | 1000 | detmax | 8.536529212999994 | 49.62078438346606 | 0.5064135493240711 |  | 0 |
| 5 | 1000 | cpu | 0.8115376689999891 | 52.32556781244542 | 0.7948867354218425 | 0.9804275960797801 | 0 |
| 5 | 1000 | cuda | 1.723951468999985 | 52.32556781244542 | 0.7948867354218425 | 0.9804275960797801 | 0 |
| 3 | 10000 | fedorov | 206.757574395 | 21.27450044494405 | 0.7484572270646577 |  | 0 |
| 3 | 10000 | modified_fedorov | 32.47234751499997 | 21.181697749824256 | 0.7450974356014265 |  | 0 |
| 3 | 10000 | detmax | 155.01177775800005 | 21.274500444944053 | 0.7484572270646574 |  | 0 |
| 3 | 10000 | cpu | 7.396110738000061 | 21.28435214766783 | 0.7323964957074166 | 0.9723462460858325 | 0 |
| 3 | 10000 | cuda | 4.6114408890000504 | 21.28435214766783 | 0.7323964957074166 | 0.9723462460858325 | 0 |
| 5 | 10000 | fedorov | 1435.608085427 | 56.7516740899648 | 0.7316111148989316 |  | 0 |
| 5 | 10000 | modified_fedorov | 198.83092278000004 | 56.48425485671348 | 0.7287587429054382 |  | 0 |
| 5 | 10000 | detmax | 410.9989747550003 | 56.77632526255357 | 0.7143839978592075 |  | 0 |
| 5 | 10000 | cpu | 40.707821907999914 | 56.83456598788113 | 0.781988872741611 | 0.9759982962346844 | 0 |
| 5 | 10000 | cuda | 9.194156619000296 | 56.83456598788113 | 0.781988872741611 | 0.9759982962346844 | 0 |
| 3 | 100000 | fedorov | 1542.0696224429998 | 21.766119302854367 | 0.7891020938860334 |  | 0 |
| 3 | 100000 | modified_fedorov | 311.0065048759998 | 21.60563541823972 | 0.6431834059213318 |  | 0 |
| 3 | 100000 | detmax | 789.0902181669999 | 21.766119302854364 | 0.7891020938860328 |  | 0 |
| 3 | 100000 | cpu | 685.1843481810001 | 21.74694934757779 | 0.7800648221624676 | 0.9733020143258417 | 0 |
| 3 | 100000 | cuda | 55.065778520000094 | 21.74694934757779 | 0.7800648221624676 | 0.9733020143258417 | 0 |
| 5 | 100000 | fedorov | TIMEOUT |  |  |  |  |
| 5 | 100000 | modified_fedorov | TIMEOUT |  |  |  |  |
| 5 | 100000 | detmax | TIMEOUT |  |  |  |  |
| 5 | 100000 | cpu | TIMEOUT |  |  |  |  |
| 5 | 100000 | cuda | 187.72138826599985 | 59.295534521578304 | 0.7782886703306097 | 0.976334351583762 | 0 |
k=3, cpu: smallest crossover m=1000
k=3, cpu: inconclusive m=[]
k=3, cuda: smallest crossover m=1000
k=3, cuda: inconclusive m=[]
k=5, cpu: smallest crossover m=1000
k=5, cpu: inconclusive m=[]
k=5, cuda: smallest crossover m=1000
k=5, cuda: inconclusive m=[]
measurement only; no claim beyond these 6 cells

Attempts 1–2 ended by controller/session or kernel-probe failure before 30/30 rows; discarded; this table is attempt 3 in full.
Attempt 1 cleanup read-back: no allocation observable (session absent, assignments 0, usage 0.00/hr).

C1 gate: PASS, PyDOE fedorov k=3 m=200, 20/20 exact ID mapping.
Cleanup: stop PASS; assignments 0 -> 0; balance 56.51 -> 52.24.
Row statuses: {"PASS": 26, "TIMEOUT": 4}; all finished-row oracles PASS.
All 6/6 cells have recorded TorchOED and PyDOE outcomes under the issue's timeout rule. At k=5, m=100000, CUDA finished in 187.72 s and all PyDOE methods reached the cap, so PyDOE quality is unavailable in that cell.
