# E0-CORRECTION-01 results

Status: **PASS; NO_MERGE; Architect exact-head review pending.**

Tested source: `778db21ad78e770b4cd1ae0e703a6cb8a4ec0e2b`.
Historical reviewed head: `8a4f2e8cc51653594387196c9bac95279f1bbe3d`.
Current main at recovery: `501b485a9eb61436fd5bcf263ced7d718b1029aa`.
Original PR base: `e6c0052adb2bff559c625a8319f89b061c971e9f`.

| Gate | Result |
|---|---|
| Same original wheel/version/import | PASS; same SHA-256, 76 installed-source byte matches |
| Original archive | 11/11 files unchanged; original SHA256SUMS 10/10 PASS |
| Historical records | Original 39 and all statuses preserved |
| P1-corrected | One fresh actual wheel call; X 20x4, F 20x15, rank15, exact det 18014398509481984 |
| Different D=ABC control | Independent math only: rank12, det0; not an upstream failure |
| P4 | Four explicitly derived interpretations, each referencing three original direct-call IDs |
| P6 | Eight stored selections rechecked for legal IDs/score parity; optimality diagnostics kept separate |
| Focused command | Exit0, finite180s limit |
| Original tests | Fresh 3 passed, exit0, finite60s limit |
| Supplement tests | Fresh 4 passed, exit0, finite60s limit |
| GPU / performance / E1 / E2 / M0 | NOT_RUN / unchanged |
| Merge | NO_MERGE; PR remains Draft |

P4 mapping (no additional PyDOE invocation for these rows):

| Derived comparison | Three actual historical source IDs |
|---|---|
| P4-L0-grid-type | P4-L0-clenshaw_curtis; P4-L0-chebyshev; P4-L0-gauss_patterson |
| P4-L1-grid-type | P4-L1-clenshaw_curtis; P4-L1-chebyshev; P4-L1-gauss_patterson |
| P4-L2-grid-type | P4-L2-clenshaw_curtis; P4-L2-chebyshev; P4-L2-gauss_patterson |
| P4-L3-grid-type | P4-L3-clenshaw_curtis; P4-L3-chebyshev; P4-L3-gauss_patterson |

Historical 39 records comprise 35 direct/import-provenance records and four derived comparisons. This focused supplement adds **one** separately counted corrected-P1 upstream function invocation. Its negative control and P4/P6 interpretations are not additional package calls.

The legacy full comparator is unchanged and was not rerun; original expected exit1/P1 FAIL remain in their original immutable artifacts. Both test suites and the focused actual-wheel command ran freshly. See supplement_01_commands.json and new logs for actual exits and hashes. Original and supplemental evidence are intentionally separate.

