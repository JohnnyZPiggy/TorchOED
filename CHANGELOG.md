# Changelog

## Unreleased

- Add `plackett_burman` for balanced two-level screening designs with 4 to 48 runs in multiples of four (#255).

- Extend `definitive_screening` to 3–20 and 23–30 factors using Paley and doubled-Paley conference matrices (#248).
- `fractional_factorial` constructs regular two-level designs from unsigned generators.
- `fold_over` reverses all factors or a single factor in the added runs.
- `word_length_pattern` computes the generalized word-length pattern for two-level designs, and `design_diagnostics(..., alias=…)` reports omitted-term aliasing for unstructured designs.

## [0.7.0] — 2026-11-16

### Added

- `jacobian_features` builds model Jacobians for design (#182, [API](website/api.md)).
- `next_batch` and a simulated closed-loop example (#184, #186, [guide](website/closed_loop.md)).
- `optimal_design(..., blocks=…)` supports fixed block sizes (#195, [guide](website/industrial.md)).
- `definitive_screening` constructs definitive screening designs (#198, [API](website/api.md)).
- `design_diagnostics` reports design correlations, VIFs and prediction variance (#200, [guide](website/industrial.md)).
- Split-plot D design with `whole_plots=`, `hard=` and `eta=` (#205, [guide](website/industrial.md)).
- `simplex_lattice` constructs mixture candidate sets (#208, [API](website/api.md)).
- Per-coefficient power in `design_diagnostics(..., effect=, alpha=)` (#210, [guide](website/industrial.md)).
- Industrial DOE guide with runnable examples (#216, [guide](website/industrial.md)).
- [Published skpr split-plot comparison report](reports/w8-skpr/summary.md) (#220).

### Changed

- A/I two-swap screening and pruning (#134–#172).
- Engine consolidation preserved outputs and reduced `src/` from 1914 to 1898 lines, missing the ≤ 1874 target (#218).

### Known limitations

- CPU float64 only for numerical acceptance. The API is experimental.
- Split-plot and blocks support D only.
- Split-plot vs skpr: one published skpr 1.9.2 example compared (`gen_design` split-plot, 33 runs in 11 plots of 3, η = 1): equal dense GLS log-determinant 13.5136, Δ = 0 (#220, [report](reports/w8-skpr/summary.md)); no wider skpr or JMP parity is claimed.
- Real PyPI publication requires the Scientist.

## [0.6.0] — 2026-11-06

### Added
- D/A/I one-/two-swap VND with criterion-specific floors (#92, #108; [API](docs/M3C_API.md)).
- Fixed-run augmentation (#98; [API](docs/M3C_API.md)) and candidate constraints/effects coding (#94; [API](docs/M1_API.md)).
- Typed public API and shipped `py.typed` (#100).
- Public documentation site with executed examples (#104; [quick start](https://johnnyzpiggy.github.io/TorchOED/)).
- macOS and minimum-torch CI (#96), sdist checks and TestPyPI publishing workflow (#106).

### Changed
- Approximate-design dual floors (#115, gate 4b): D median 0.9316 over eight dev cases; grid3 n=20 A/I 0.9395/0.9579 ([evidence](reports/w113/summary.md)).
- Trusted internal D verification (#117): outputs bit-identical, head/base time 0.81-0.96; 5/8 dev cases meet <=3x fastest PyDOE exchange ([evidence](reports/w114/summary.md)).

### Known limitations
- CPU float64 numerical certificates are not global optimality proofs; experimental API, no GPU support claim.
- P1 time target remains unmet on grid2 n=6/n=7 and grid3 n=10; ten TorchOED restarts are compared with single PyDOE calls.
- H5-H8 are frozen; week-end held-out evaluation is pending. Real PyPI publication requires the Scientist.

## [0.5.0] — 2026-10-31

### Added
- Certified `d_efficiency_lower_bound` on search results and comparable benchmark designs.
- Fixed CPU comparisons with PyDOE 1.6.0 and a consolidated [Beta report](reports/beta/summary.md).

### Changed
- `k2_exchange` ranks two-swaps using G-based scores, exact verification of survivors, prefix pruning, and closed-form survivor scores (B1–B6).
- The fixed random500 n=20 case takes 1.331 s (3.13× sequential; the ≤3× target is not met); G2 scale results are recorded in the Beta report.

### Known limitations
- CPU float64 only for M3 search; numerical local certificates and efficiency floors do not establish global optimality. No GPU or PyDOE parity claim.
- Experimental API; compatibility is not guaranteed.

## [0.1.0] — 2026-10-31

### Added

#### M1 — constructors and features
- `full_factorial`: enumerate factorial level indices ([API](docs/M1_API.md)).
- `ff2n`: construct signed two-level factorial corners ([API](docs/M1_API.md)).
- `lhs`: generate Latin hypercube samples ([API](docs/M1_API.md)).
- `sobol`: generate Sobol sequence samples ([API](docs/M1_API.md)).
- `polynomial_features`: expand factors into a polynomial model matrix ([API](docs/M1_API.md)).

#### M2 — information and criteria
- `exact_information`: form the information matrix for exact observation IDs ([API](docs/M2_API.md)).
- `weighted_information`: form probability-weighted information ([API](docs/M2_API.md)).
- `d_optimality`: score numerical SPD information by log determinant ([API](docs/M2_API.md)).
- `a_optimality`: score numerical SPD information by negative inverse trace ([API](docs/M2_API.md)).
- `e_optimality`: score numerical PSD information by its minimum eigenvalue ([API](docs/M2_API.md)).
- `c_optimality`: score a contrast by negative variance on SPD information ([API](docs/M2B_API.md)).

#### M3 — search and scoring
- `fedorov_exchange`: improve provided IDs with deterministic one-swaps ([API](docs/M3_API.md)).
- `score_two_swaps`: score and verify every legal simultaneous two-swap ([API](docs/M3B1_API.md)).
- `k2_exchange`: run deterministic one-/two-swap VND from provided IDs ([API](docs/M3B2_API.md)).
- `d_optimal_design`: select the best seeded feasible-start VND restart ([API](docs/M3C_API.md)).

### Known limitations
- M3 search and scoring require CPU float64 inputs.
- Local-optimality certificates are numerical, not exact or global.
- No GPU support claim.
- No PyDOE parity claim.
- Experimental API; compatibility is not guaranteed.
