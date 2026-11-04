# Criteria and efficiency bounds

Let F contain all candidate feature rows, S the chosen n distinct IDs, and M = F[S]ᵀF[S]. The information matrix must be numerically positive definite for D/A/I search.

| criterion | Maximized objective | B |
|---|---|---|
| D | logdet M | not used |
| A | −tr(M⁻¹) | identity Iₚ |
| I | −tr(M⁻¹B) | uniform candidate moment FᵀF/m |

A depends on feature scaling and coding. I measures average prediction variance over the supplied candidates. Pass `criterion="A"` or `criterion="I"` to `optimal_design`; its `objective` retains the maximization sign.

For D, G = FM⁻¹Fᵀ and the returned bound is p/(n maxⱼ Gⱼⱼ), equivalently exp(−gap/p). For A/I, Qⱼⱼ = fⱼᵀM⁻¹BM⁻¹fⱼ and the bound is tr(BM⁻¹)/(n maxⱼ Qⱼⱼ). `efficiency_lower_bound` is criterion-specific; D also retains `d_efficiency_lower_bound`.

These are conservative lower bounds against the unrestricted approximate optimum, and therefore also against the best exact or fixed-run augmented optimum. They are not achieved efficiencies. The theorems hold in exact arithmetic; float64 evaluations carry numerical error and are not interval certificates.

D/A/I use 1-/2-swap search. A local numerical certificate describes only the checked neighbourhood, never global optimality. Read the frozen [W1 definitions and proofs](https://github.com/JohnnyZPiggy/TorchOED/blob/6c9d63d3b48b5afe6a3189364e19614d9897854d/docs/W1_GENERIC_CRITERIA_GATE.md).
