import csv, json
from pathlib import Path
import torch
from torchoed import lhs, space_filling_diagnostics
p = Path(__file__).parent
torch.set_num_threads(1)
def tensor(rows): return torch.tensor(rows, dtype=torch.float64)
def diagnostics(x): return {k: v.item() for k, v in space_filling_diagnostics(x).items()}
def midpoint(x): return (x.argsort(0).argsort(0).to(torch.float64) + .5) / len(x)
def close(a, b): return abs(a-b) <= 1e-12 * abs(b)
raw = json.loads((p / "sessionInfo.txt").read_text().splitlines()[0])
frozen = {m: tensor([[float(v) for v in r] for r in csv.reader((p / (m+".csv")).open())]) for m in ("random","maximin","maxpro","uniform")}
ours = {m: diagnostics(x) for m, x in frozen.items()}
# Infer one convention on the first frozen design, then apply it unchanged.
r = raw["diagnostics"]["random"]; t = ours["random"]
cd_power = 1 if close(t["cd2"], r["cd2"]) else 2 if close(t["cd2"]**2,r["cd2"]) else None
mp_power = 1 if close(t["maxpro"],r["maxpro"]) else 3 if close(t["maxpro"]**3,r["maxpro"]) else None
rows = {}
for m, t in ours.items():
    r = raw["diagnostics"][m]
    checks = {k: close(t[k],r[k]) for k in ("min_distance","phi_p")}
    checks["cd2"] = cd_power is not None and close(t["cd2"]**cd_power,r["cd2"])
    checks["maxpro"] = mp_power is not None and close(t["maxpro"]**mp_power,r["maxpro"])
    rows[m] = dict(torchoed=t,reference=r,checks=checks,verdict="MATCH" if all(checks.values()) else "NO MATCH")
def quality(x, ref, metric):
    a, b = diagnostics(x), diagnostics(midpoint(ref))
    return dict(torchoed=a,reference=b,delta=a[metric]-b[metric],verdict="MATCH or TorchOED better" if a[metric]<=b[metric]*(1+1e-9) else "NO MATCH")
quality_rows = {m: quality(frozen[m],tensor(x),"phi_p" if m=="maximin" else "maxpro") for m,x in raw["quality"].items()}
examples = {}
for pkg, e in raw["examples"].items():
    if e.get("status") == "skipped: held-out size":
        examples[pkg] = e; continue
    ref = tensor(e["design"]); n,k = ref.shape
    if (n,k) in ((10,2),(20,3),(7,2)): raise ValueError("held-out example must be skipped before execution")
    if pkg == "DiceDesign":
        t = diagnostics(ref); r = e["metrics"]["cd2"]
        row = dict(torchoed=t,reference_cd2=r,verdict="MATCH" if cd_power is not None and close(t["cd2"]**cd_power,r) else "NO MATCH")
    else:
        m = "maximin" if pkg=="lhs" else "maxpro"
        x = lhs(n,k,mode=m,generator=torch.Generator().manual_seed(0))
        row = quality(x,ref,"phi_p" if m=="maximin" else "maxpro")
    row.update(rd_file=e["rd_file"],rd2ex_line=e["rd2ex_line"],call=e["call"],shape=[n,k]); examples[pkg]=row
result = dict(run_id=raw["run_id"],source_sha=raw["source_sha"],R_version=raw["R_version"],versions=raw["versions"],cd2_power_mapping=cd_power,maxpro_power_mapping=mp_power,diagnostics=rows,quality=quality_rows,examples=examples,before=11,after=11+sum(e.get("verdict", "NO MATCH")!="NO MATCH" for e in examples.values()))
(p / "results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
print(json.dumps(result,indent=2))
