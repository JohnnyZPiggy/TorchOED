# M0 numerical scope note

This short Issue #1 deliverable links to the canonical [M1/M2 numerical contract](M1_M2_NUMERICAL_CONTRACT.md), including §9, and [Mentor §25](SCIENTIFIC_MENTOR_RESEARCH.md#25-e0-actual-pydoe-160-wheel-evidence-corrected-full-quadratic-oracle-and-architect-review-2026-10-29). It does not freeze a public M0 numerical API or expand scope.

- Physical candidates C:(m,k), features F:(m,p), selected features X:(n,p); future explicit batch B gives (B,m,p)/(B,n,p). Information would be (p,p)/(B,p,p); candidate scores (m,)/(B,m); swaps (n,m)/(B,n,m) or tiles.
- Exact information defaults to raw X.T@X; normalization /n must be explicit. Approximate weights are nonnegative with sum1; weighted GEMM, no dense diag. These future functions are **not implemented** here.
- Canonical future maximization signs: D=logdet(M), A=-tr(M^-1), E=lambda_min(M); C needs c, I needs H, G/V need evaluation rows. S/T remain deferred. The old plan's A minimization is statistically equivalent to the contract's negative maximization; never compare signs blindly.
- Float64 CPU is the initial reference; float32, other devices, batching and autograd require separate tests. Future random generators must state device/version/seed semantics; no cross-backend bitwise promise.
- No silent ridge or normalization. Future rank-deficient/nonfinite/invalid shapes require explicit status/exception policies. Dense covariance restricts **before** inversion; correlated repeats need observation-event identities and are deferred.
- Candidate ID differs from coordinate equality; without-replacement IDs and replicated events need distinct policies. No fuzzy isclose masks for identity.
- Frozen PyDOE defaults differ: information normalized=True; D returns determinant; A is negative inverse trace with possible implicit regularization; Jacobian search adds a small ridge and permits repeats. Equality oracles must align definitions before comparing values.
- Corrected E0 Small Composite has rank15/15; the separate D=ABC control has rank12. Historical source-only predictions are not production feature evidence.
- M0 implements only private _gram(F)=F.mT@F for a small CPU fixture. This internal path has no promised public input-validation, normalization, repeated-ID or solver contract. None of the 91 upstream public exports is implemented.
