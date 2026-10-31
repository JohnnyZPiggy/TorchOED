# Changelog

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
