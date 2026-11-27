# W17 covering-array sizes vs best-known

R run 45913977821, frozen source `e66c8f430e109dc8e3738a1064beb47aece7d375`.
Versions: {"R": "R version 4.5.2 (2025-10-31)", "CAs": "0.24-3", "caverify": "0.3.0", "Python": "3.14.6", "torch": "2.15.0+cpu"}.

Target: Colbourn November 2024 snapshot via pinned CAs 0.24-3; reference lookup NOT_RUN.
Python coverage: 99/99 PASS; R covered: 0 TRUE, 99 NOT_RUN; ratios measured: 0/99; count <= 1.3: NOT_RUN.

| t,v | 4 | 5 | 6 | 8 | 10 | 13 | 16 | 20 | 30 | 50 | 100 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2,3 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 2,4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 2,5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 2,6 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3,2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3,3 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3,4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3,5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3,6 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Errors (verbatim): ["Error in CAs::eCAN(3, 10, 3) : object 'colbournBigFrame' not found"]
Index-one consistency failures: []
R workflow FAILED at the first str(eCAN(3, 10, 3)) probe; both pins installed. The CSVs are host-generated NOT_RUN placeholders, not R outputs. No second dispatch permitted; matrix pending text retained.
Scope: CPU evidence only; no constructor changes, tuning, held-out cases or timing claims.
