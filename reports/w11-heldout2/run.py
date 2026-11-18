"""Frozen #336 CPU float64 cases: run once, never tune or rerun searches."""
import itertools, json, math, platform, subprocess, sys, time
from pathlib import Path
import torch
from torchoed import full_factorial, mixture_process_features, optimal_design, scheffe_features, simplex_centroid
torch.set_num_threads(1)
path = Path(__file__).with_name('results.json')
result = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), python=platform.python_version(), torch=torch.__version__, device='CPU', dtype='float64', threads=1, seed=0, restarts=10, chunk=2048, tie_rtol=1e-10, tie_atol=1e-12)
if '--score' in sys.argv: result = json.loads(path.read_text())
def score(F, ids, case, Z=None, G=None):
    s = F[ids]; M = s.mT@s
    if Z is not None:
        X = torch.cat((torch.ones_like(Z), Z), 1); Q = torch.eye(len(Z), dtype=F.dtype)-X@torch.linalg.solve(X.mT@X, X.mT); M = s.mT@Q@s
    L, info = torch.linalg.cholesky_ex(M); good = info == 0; L = L[good]
    ld = torch.linalg.slogdet(M[good]).logabsdet; penalty = torch.zeros_like(ld)
    if G is not None: penalty = torch.cholesky_solve(s[good].mT@G[ids[good]], L).square().sum((-2,-1))
    objective = -torch.cholesky_inverse(L).diagonal(dim1=-2,dim2=-1).sum(-1) if case == 'H28' else (ld/4-math.log(ids.shape[1])-0.5*penalty if G is not None else ld)
    metric = (ld/6-math.log(ids.shape[1])).exp() if case == 'H27' else (-objective if case == 'H28' else objective)
    return good, objective, metric, ld.exp(), penalty
for case, n, count in [('H27',9,293930), ('H28',6,60480), ('H29',8,2220075)]:
    start = time.perf_counter(); Z = G = None
    if case == 'H27': F = mixture_process_features(scheffe_features(simplex_centroid(3), 'linear'), torch.tensor([[1.,-1.],[1.,0.],[1.,1.]], dtype=torch.float64))
    else:
        X = full_factorial([3]*(2 if case == 'H28' else 3)).double()-1; x = X.unbind(1)
        F = torch.stack((x[0],x[1],x[0]*x[1]),1) if case == 'H28' else torch.cat((torch.ones_like(X[:,:1]),X),1)
        if case == 'H28': Z = torch.tensor([[1.],[-1.]]*3, dtype=torch.float64)
        else: G = torch.stack((x[0]*x[1],x[0]*x[2],x[1]*x[2]),1)
    options = dict(criterion='A', covariates=Z) if Z is not None else (dict(alias=G,alias_weight=0.5) if G is not None else dict(criterion='D'))
    if '--score' in sys.argv:
        row = result[case]; _, objective, _, _, _ = score(F,torch.tensor([row['ids']]),case,Z,G)
        assert row['visited']==count and row['reference_count']=='PASS'
        assert row['dense_rescore']=='PASS' and math.isclose(objective.item(),row['objective'],rel_tol=1e-10,abs_tol=1e-12)
        print(case, '2 PASS, 0 FAIL (saved counts; dense re-score only)'); continue
    r = optimal_design(F,n,**options,generator=torch.Generator().manual_seed(0)); ids = r['ids']; floor = optimal_design(F,n,**options,fixed_ids=ids,restarts=1)['efficiency_lower_bound']
    _, obj, metric, det, penalty = score(F,ids[None],case,Z,G)
    row = dict(ids=ids.tolist(),objective=r['objective'],metric=metric.item(),floor=floor,det_M=det.item(),alias_trace=penalty.item(),dense_rescore='PASS' if math.isclose(obj.item(),r['objective'],rel_tol=1e-10,abs_tol=1e-12) else 'FAIL')
    selections = iter((itertools.permutations if Z is not None else itertools.combinations)(range(len(F)),n)); visited = singular = ties = 0; best = -math.inf; first = None
    while batch := list(itertools.islice(selections,2048)):
        b = torch.tensor(batch); good, values, _, _, _ = score(F,b,case,Z,G); visited += len(b); singular += (~good).sum().item()
        if not len(values): continue
        top = values.max().item()
        if top > best and not math.isclose(top,best,rel_tol=1e-10,abs_tol=1e-12): best = top; ties = 0; first = None
        tied = torch.isclose(values,torch.tensor(best,dtype=F.dtype),rtol=1e-10,atol=1e-12); ties += tied.sum().item()
        if first is None and tied.any(): first = b[good][tied][0].tolist()
    _, _, ref, refdet, refpenalty = score(F,torch.tensor([first]),case,Z,G); delta = ref.item()-metric.item()
    row.update(reference=dict(ids=first,metric=ref.item(),det_M=refdet.item(),alias_trace=refpenalty.item(),tied_optima=ties),delta=delta,match='MATCH' if math.isclose(ref.item(),metric.item(),rel_tol=1e-10,abs_tol=0) else 'NO MATCH',visited=visited,singular=singular,reference_count='PASS' if visited==count else 'FAIL',wall_seconds=time.perf_counter()-start)
    result[case] = row; path.write_text(json.dumps(result,indent=2)+'\n'); print(case, json.dumps(row), flush=True)
