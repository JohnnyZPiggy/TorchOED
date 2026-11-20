import csv, json, platform, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from torchoed import design_diagnostics, effects_coding

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
reference = json.loads((HERE/"reference.json").read_text())
runtime = json.loads((HERE/"runtime.json").read_text())
results = []
for name, ref in reference.items():
    rows = list(csv.DictReader((HERE/f"{name}.csv").open()))
    if name == "simple":
        F = torch.tensor([[1., *[float(r[k]) for k in ("A","B","C")]] for r in rows], dtype=torch.float64)
        groups, terms = [[1],[2],[3]], ["A","B","C"]
    else:
        cost = torch.tensor([2*float(r["cost"])-3 for r in rows], dtype=torch.float64)
        coded = [effects_coding(torch.tensor([levels.index(r[k]) for r in rows]),len(levels))
                 for k,levels in [("size",["Grande","Short","Venti"]),
                                  ("type",["Colombian","Ethiopian","Kona","Sumatra"])]]
        F = torch.cat((torch.ones((len(rows),1)),cost[:,None],*coded),1)
        groups, terms = [[1],[2,3],[4,5,6]], ["cost","size","type"]
    if name == "simple":
        torch.testing.assert_close(F,torch.tensor(ref["model_matrix"],dtype=F.dtype),rtol=0,atol=0)
    effect = torch.tensor(ref["anticoef"],dtype=F.dtype)
    diag = design_diagnostics(F,torch.arange(len(F)),effect=effect,groups=groups)
    comparisons = []
    for j,col in enumerate(ref["columns"]):
        if j == 0 or (name == "coffee" and col != "cost"): continue
        r = next(r["power"] for r in ref["power"] if r["type"]=="parameter.power" and r["parameter"]==col)
        comparisons.append(dict(term=col,kind="parameter",torch_power=diag["power"][j].item(),R_power=r,delta=diag["power"][j].item()-r))
    for j,term in enumerate(terms):
        r = next(r["power"] for r in ref["power"] if r["type"]=="effect.power" and r["parameter"]==term)
        comparisons.append(dict(term=term,kind="effect",torch_power=diag["group_power"][j].item(),R_power=r,delta=diag["group_power"][j].item()-r))
    delta = max(abs(r["delta"]) for r in comparisons)
    results.append(dict(example=name,n=len(F),comparisons=comparisons,max_abs_delta=delta,verdict="MATCH" if delta<=1e-8 else "NO MATCH"))
output = dict(run_id=runtime["run_id"],source_sha=runtime["source_sha"],R_version=runtime["R_version"],skpr=runtime["packages"]["skpr"],python=platform.python_version(),torch=torch.__version__,tolerance=1e-8,examples=results,published_count_before=14,published_count_after=14+sum(r["verdict"]=="MATCH" for r in results))
(HERE/"results.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
print(json.dumps(output,indent=2))
print(f"Power comparisons: {sum(abs(v['delta'])<=1e-8 for r in results for v in r['comparisons'])} PASS, {sum(abs(v['delta'])>1e-8 for r in results for v in r['comparisons'])} FAIL")
