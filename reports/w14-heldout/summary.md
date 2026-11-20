# Week 14 frozen held-out H36–H39

Frozen in docs/WEEK_GOAL.md by #434, before W14 feature merges; no tuning, extra seeds or sizes.
H36–H38 MATCH exhaustive exact references (details in results.json); H39 TorchOED recorded, R NOT_RUN.
H36: exact integer Bareiss across all 296,010 subsets; independent Fraction elimination on TorchOED's determinant.
H37: Y = 3F has integer entries; G = YᵀY + 9K, metric tr(G⁻¹YᵀY)/27. Exact primary Gram determinant excludes singular subsets; float64 prefilter, then Fraction for every subset within relative 1e-8 of the float minimum.
H38: exact integer determinants across all 12,870 subsets; order by det(M1)^7 det(M2)^5; independent Fraction elimination on both TorchOED determinants.
H39: S.csv has 17 significant digits and reads back bit-equal to the default feasible Sobol sample; original-coordinate TorchOED ψ and minimum distance are recorded.
ONE R workflow run 44714545359, source 8de053521994022cdaade5b17eeca193506257c0, R 4.5.2, support 0.1.7: FAILED before support::sp.
Engineer sequencing error: initial sample generation failed on float32 constraints against float64 S; shell continued and dispatched before S.csv was committed. No second dispatch; no reference design exists, verdict NOT_RUN, verbatim error in results.json.
The final driver sets float64 defaults before constructing the frozen calls (otherwise the literal constraints have float32 dtype). Frozen calls/seeds/settings are unchanged.
Raw workflow runtime.json and sessionInfo.txt are preserved. H39-R.csv is absent because the R reference failed; seven report files total.
Reproduce: PYTHONPATH=src python reports/w14-heldout/run.py (NumPy is a dev-only exact-reference/prefilter helper).
Required reproduction reruns the identical frozen calls, not a tuning run. Counts/values and maximum/minimum multiplicities are in results.json.
Published examples stay at 16. HO matrix cells remain untouched, for the week-15 goal commit.
