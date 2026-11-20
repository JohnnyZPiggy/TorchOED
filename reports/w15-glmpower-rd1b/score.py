import json
from pathlib import Path
import torch
from torchoed import glm_power

root = Path(__file__).resolve().parent
raw = json.loads((root / "skpr.json").read_text(encoding="utf-8"))
runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
result = dict(case="RD1b", run_id=runtime["run_id"], source_sha=runtime["source_sha"],
              R_version=runtime["R_version"], skpr_version=runtime["packages"]["skpr"],
              mbest_version=runtime["packages"].get("mbest"), expression=raw["expression"])
summary = ["# RD1b", f"Run {result['run_id']}; {result['R_version']}; skpr {result['skpr_version']}; mbest {result['mbest_version']}."]
if raw["status"] == "NOT_RUN":
    result.update(verdict="NOT_RUN", error=raw["error"], powers=None, combined_se=None)
    summary += ["NOT_RUN: child Rscript failed; stderr is recorded verbatim in results.json and skpr.json."]
else:
    attrs = raw["attributes"]
    method = attrs["parameter_analysis_method_string"]
    result.update(parameter_analysis_method_string=method, nsim_skpr=raw["nsim"])
    if "firth" in str(method).lower() or "penali" in str(method).lower():
        result.update(verdict="NOT_COMPARABLE (Firth)", powers=raw["power"])
        summary += [f"NOT_COMPARABLE (Firth): {method}"]
    else:
        X = torch.tensor(attrs["model_matrix"], dtype=torch.float64)
        beta = torch.tensor(attrs["anticoef"], dtype=torch.float64)
        out = glm_power(X, beta, "binomial", alpha=attrs["alpha"], nsim=10000,
                        generator=torch.Generator().manual_seed(0))
        rows = [r for r in raw["power"] if r["type"].startswith("parameter.power")]
        assert len(rows) == len(beta)
        comparisons = []
        for row, pt in zip(rows, out["power"].tolist()):
            ps = row["power"]
            se = (ps*(1-ps)/raw["nsim"] + pt*(1-pt)/10000)**0.5
            delta = abs(ps-pt)
            z = delta/se if se else (0.0 if delta == 0 else float("inf"))
            comparisons.append(dict(parameter=row["parameter"], skpr=ps, torchoed=pt,
                                    combined_se=se, z=z, match=delta <= 4*se))
        result.update(verdict="MATCH" if all(r["match"] for r in comparisons) else "NO MATCH",
                      comparisons=comparisons, worst_z=max(r["z"] for r in comparisons),
                      nsim_torchoed=10000, nonconverged=out["nonconverged"], separated=out["separated"])
        summary += [f"{result['verdict']}: {sum(r['match'] for r in comparisons)}/{len(comparisons)} coefficients within 4 combined SE; worst z={result['worst_z']:.6g}."]
summary += ["Single authorized retry; no second dispatch.",
            "Published examples: 18 -> 19." if result["verdict"] == "MATCH" else "Published examples: 18 unchanged."]
(root / "results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
(root / "summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8", newline="\n")
