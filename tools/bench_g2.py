"""B4 fixed G2 measurement: python tools/bench_g2.py in the existing dev venv."""
import importlib
import json
import subprocess
import sys
import time
import bench_b0 as b

torch = b.torch
torch.set_num_threads(1)
design = importlib.import_module('torchoed.d_optimal_design')
original = design.k2_exchange
def capture(*args, **kwargs):
    result = original(*args, **kwargs)
    counts.append(dict(scanned_pairs=result['scanned_pairs'], history_length=len(result['swap_history']),
                       accepted_by_phase={str(k): sum(m['k']==k for m in result['swap_history']) for k in (1,2)}))
    return result
design.k2_exchange = capture  # Retain existing return diagnostics without extra search calls.
C = 2*torch.rand(2000,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0))-1
F = b.polynomial_features(C, degree=2)
out = b.ROOT/'reports'/(sys.argv[1] if len(sys.argv)>1 else 'b4')
result_path = out/('g2.json' if out.name=='beta' else 'results.json')
out.mkdir(parents=True, exist_ok=True)
data = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),
            environment=dict(python=b.platform.python_version(),torch=torch.__version__,threads=1),
            m=2000,p=10,n=20,restarts=10,candidate_seed=0,rows=[])
diagnostics = {}
for seed in (0,1,2):
    counts = []
    begin = time.perf_counter()
    result = design.d_optimal_design(F,20,restarts=10,generator=torch.Generator().manual_seed(seed))
    row = dict(seed=seed,wall_seconds=time.perf_counter()-begin,ids=result['ids'].tolist(),objective=result['objective'],
               stop_reason=result['stop_reason'],certificate=result['certificate'],d_efficiency_lower_bound=result['d_efficiency_lower_bound'])
    metric = dict(ids=row['ids'],repeats=len(row['ids'])-len(set(row['ids'])))
    b.score_design(F,metric,None)
    row['independent_bound'] = metric['d_efficiency_lower_bound']
    row['bound_absolute_difference'] = abs(row['d_efficiency_lower_bound']-row['independent_bound'])
    row['bound_check_passed'] = row['bound_absolute_difference']<=1e-9 and 0<row['d_efficiency_lower_bound']<=1 and 0<row['independent_bound']<=1
    data['rows'].append(row)
    diagnostics[seed] = counts
    result_path.write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8')
    print(row,flush=True)
    assert row['bound_check_passed']
data['G2_met'] = all(r['wall_seconds']<=120 and r['bound_check_passed'] for r in data['rows'])
if not data['G2_met']:
    slowest = max(data['rows'],key=lambda r:r['wall_seconds'])['seed']
    data['slowest_seed_diagnostics'] = dict(seed=slowest,restarts=diagnostics[slowest])
result_path.write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8')
lines = ['# B4: G2 scale measurement','', '**G2 '+('MET' if data['G2_met'] else 'NOT MET')+'**: three fixed seeds; target <=120 s each with a checked certified lower bound.', '', 'Command: `python tools/bench_g2.py` in the existing dev venv. random2000 seed 0, quadratic p=10, n=20, restarts=10; all other parameters default, one thread, one timed call per seed. Timing includes copying existing restart diagnostics; independent bench_b0.score_design rescoring is excluded. No PyDOE search or GPU run. Bounds are floors, not achieved efficiencies.', '', '| Seed | Wall s | Objective | Stop / certificate | Returned bound | Independent bound | Absolute difference | Check |','|---|---|---|---|---|---|---|---|']
lines += [f"| {r['seed']} | {r['wall_seconds']:.6g} | {r['objective']:.12g} | {r['stop_reason']} / {r['certificate']} | {r['d_efficiency_lower_bound']:.12g} | {r['independent_bound']:.12g} | {r['bound_absolute_difference']:.3g} | {r['bound_check_passed']} |" for r in data['rows']]
if not data['G2_met']: lines += ['',f"Slowest seed: {slowest}; existing per-restart scanned_pairs, history lengths and accepted moves per phase are in results.json."]
if out.name!='beta': (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
