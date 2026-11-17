# Week 10 goal — 2026-11-17 → 2026-11-24 (VISION.md v3 phase P2 completion, part 2: classical constructors and aliasing)

**Auto-advance (Architect, 2026-11-17).** Week 9 goals B1–B4 were met on 2026-11-17 (#228, #232, #234, #236, #238). Held-out H19–H20 were recorded as-is by #241 (`ff26b52`): both TorchOED rows PASS, and both references are NOT_RUN (AlgDesign `optBlock` documents no I criterion; skpr 1.9.2 raised `undefined columns selected`). `blocks + split-plot together` is not in the W9 gate and moves to week 11, per its matrix row. Baseline: `main` at the START of this goal's commit task.

**Why this week:** JMP's Screening and Response Surface platforms, and R FrF2/rsm, are the first tools a practitioner reaches for. TorchOED has no fractional factorial, Plackett–Burman, central composite or Box–Behnken constructor and no alias matrix, so these four matrix rows are ❌. Week 10 fills them, plus the open Ext cell of definitive screening.

Every Engineer dispatch names the goal (C1–C3) it serves and updates its matrix row in the same PR. New public functions and keywords are decided in a gate first, with an independent fresh-context review, and are exported from `torchoed` with an API entry, a guide snippet and a CHANGELOG line. Constructors return coded `torch.float64` tensors (CPU by default, `device=` accepted), consistent with `ff2n` and `definitive_screening`.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **C1 Fractional factorials, fold-over, alias matrix** | Gate first (independent review): regular 2^(k−p) designs from generators, and by `(k, runs)` from one cited published table of maximum-resolution / minimum-aberration generators for 8, 16, 32 and 64 runs; full and single-factor fold-over; alias matrix `A = (F₁ᵀF₁)⁻¹F₁ᵀF₂` exposed through `design_diagnostics` (keyword decided in the gate). Oracles: (a) the word-length pattern from the J-characteristics `A_j = N⁻² Σ_{|S|=j} (Σᵢ Πc∈S xᵢc)²` (exact `Fraction`) equals the one implied by the defining relation, and the resolution is its smallest nonzero `j`; (b) for regular designs, every alias entry is in {0, ±1} and matches the defining-relation words, checked exhaustively on ≥ 3 designs; (c) a full fold-over of a resolution-III design has resolution ≥ IV (gate PROVES it), checked on ≥ 2 designs; (d) the dense alias formula matches at 1e-10. Measured number: the table entries reproduced exactly (generators and WLP), target **every entry in the cited table for k ≤ 11**. External: R FrF2 on Colab for ≥ 3 `(runs, k)` cases, re-scored by the shared WLP, or NR. The matrix row has every cell ✅ / N/A / NR | ❌ row; 0 entries |
| **C2 Plackett–Burman, central composite (incl. blocked), Box–Behnken** | Gate first (independent review): Plackett–Burman for N = 12, 20 and 24 (and every N ≡ 0 mod 4 ≤ 48 that the gate's construction covers; any other N raises); CCD with `alpha` rotatable / orthogonal / face-centred / numeric, center points per block, and orthogonal blocking into cube and axial blocks (Box–Hunter condition stated and PROVED in the gate); Box–Behnken for k = 3…7 from the published Box–Behnken (1960) incomplete-block table. Oracles: (a) PB main-effect columns exactly orthogonal (`Fraction`); (b) CCD rotatability moment condition and orthogonal-blocking condition `max |M_c − M_B| = 0` (block-centered vs overall-centered information of the full quadratic) exact on ≥ 2 k; (c) BBD runs equal the published table row-for-row (as sets) for k = 3, 4, 5. Measured number: constructor cases matched exactly to a published table or rsm/FrF2 output, target **≥ 10** across the three families. External: R rsm `ccd`/`bbd` and FrF2 `pb` on Colab, re-scored by the shared metric, or NR | ❌ rows; 0 cases |
| **C3 Definitive screening beyond Paley prime orders** | Gate first (independent review): conference matrices for every order the gate's construction covers (at least Paley for prime powers q, both q ≡ 1 and q ≡ 3 mod 4, via GF(q) arithmetic), with `definitive_screening(k)` for every k ≤ 30 whose order (k, or k + 1 for odd k, as now) is covered; orders with no conference matrix raise with the reason. Oracles: `CᵀC = (n − 1)I` exact; main effects orthogonal to each other and to all two-factor interactions and quadratic columns (`Fraction`); existing Paley results unchanged. Measured number: k ≤ 30 supported, now prime-order Paley only, target **every k ≤ 30 whose order is covered by the gate, others documented**. External: one published DSD (Jones and Nachtsheim 2011, or Xiao, Lin and Bai 2012) matched up to row/column permutation and sign, or NR | Paley q prime only |

**Held-out additions (frozen here, before any C1 or C2 code merges; never used for tests, tuning or dispatch choices; run once at week end, reported as-is):**
- **H21 (fractional factorial):** the TorchOED 16-run design for k = 7 chosen by the C1 `(k, runs)` path, with no generators given. Record the generators, the WLP `(A₃, …, A₇)` and the resolution. Reference: R FrF2 `FrF2(nruns=16, nfactors=7, randomize=FALSE)`, with `generators()` and `sessionInfo()` recorded and re-scored by the same shared WLP function.
- **H22 (blocked CCD):** the TorchOED k = 3 central composite with orthogonal-blocking `alpha`, a cube block with 4 center points and an axial block with 2 center points (20 runs). Shared metric with `F = polynomial_features(rows, 2, include_bias=False)`: `D_B = (det M_B / n⁹)^(1/9)` and `max |M_c − M_B|`; also record `alpha`. Reference: R rsm `ccd(3, n0=c(4,2), alpha="orthogonal", randomize=FALSE)` in coded units, re-scored by the same function.
- **H23 (Box–Behnken):** the TorchOED k = 4 Box–Behnken with 3 center points (27 runs), unblocked. Shared metric with `F = polynomial_features(rows, 2)`: `D = (det(FᵀF)/n¹⁵)^(1/15)` and `I = tr(W M⁻¹)` with `M = FᵀF / n` and `W` the moment matrix of the 3⁴ grid. Reference: R rsm `bbd(4, n0=3, block=FALSE, randomize=FALSE)` in coded units, re-scored by the same function.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise. Constructors are deterministic and timed only if a goal asks.

**Safeguards:**
- Every gate passes an independent fresh-context review before code.
- Published tables are cited by source and edition, and frozen as repo data or in the gate. Do not transcribe from memory.
- No claim of JMP parity beyond reproduced references.
- External references follow the VISION.md oracle rules.
- No tuning on held-out cases.
- The matrix row of each goal is updated with evidence links in the same PR.
- At most 3 feature goals.

**Out of scope this week:**
- Minimum-aberration search beyond the cited table, and blocked fractional factorials (week 15).
- `blocks + split-plot together`, mixtures, covariates, and alias-optimal and compound criteria (week 11).
- Supersaturated designs, orthogonal and covering arrays (P2+).
- GPU work, P2+ and P3.
- Real PyPI and Issue #13.
