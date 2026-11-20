"""Reproduce the final RD1 NOT_RUN record from the single authorized R run."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = json.loads((root / "skpr.json").read_text(encoding="utf-8"))
runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
assert raw["status"] == "NOT_RUN"
result = dict(case="RD1", verdict="NOT_RUN", run_id=runtime["run_id"],
              source_sha=runtime["source_sha"], R_version=runtime["R_version"],
              skpr_version=runtime["packages"]["skpr"],
              expression=raw["expression"], error=raw["error"],
              powers=None, combined_se=None,
              reason="Child Rscript failed: missing mbest; no evaluated design or powers were exported. No second dispatch.")
(root / "results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
summary = ["# RD1: NOT_RUN",
           f"Run {result['run_id']}; {result['R_version']}; skpr {result['skpr_version']}.",
           "Expression: eval_design_mc(designbinom, ~a + b, alpha=0.2, nsim=100, effectsize=c(0.7, 0.9), glmfamily=\"binomial\") inside run_documentation().",
           "Child Rscript failed because future could not attach the missing mbest package.",
           "Per-coefficient powers and combined SE: NOT_RUN; no model matrix or anticoef was exported.",
           "Verbatim expression and stderr: results.json and skpr.json. No second dispatch; published example count stays 18."]
(root / "summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8", newline="\n")
