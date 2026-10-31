"""Issue #40 fixed CPU comparison: python tools/bench_b0.py PYDOE_1_6_WHEEL."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import statistics as stats
import subprocess
import sys
import time

import numpy as np
import pydoe
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from torchoed import d_optimal_design, d_optimality, full_factorial, polynomial_features
from oracle_exchange import determinant


def score_design(F, row, exact_F):
    X = F[row["ids"]]
    M = X.mT@X
    try:
        row["logdet"] = d_optimality(M).item()
    except ValueError as exc:
        row.update(logdet=None, metric_error=str(exc), raw_slogdet=str(torch.linalg.slogdet(M).logabsdet.item()))
        if exact_F is not None: row["exact_determinant"] = str(determinant(exact_F,row["ids"]))


def main():
    torch.set_num_threads(1)
    wheel = Path(sys.argv[1]).read_bytes()
    digest = hashlib.sha256(wheel).hexdigest()
    assert version('pydoe') == '1.6.0'
    assert digest == '5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec'
    out = ROOT/'reports'/(sys.argv[2] if len(sys.argv)>2 else 'b0')
    out.mkdir(parents=True, exist_ok=True)
    data = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        environment=dict(python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
                         torch=torch.__version__, numpy=np.__version__, scipy=version('scipy'), pydoe=version('pydoe'), threads=1),
        wheel=dict(sha256=digest, bytes=len(wheel)), candidate_seed=0, cases=[])
    def save():
        (out/'results.json').write_text(json.dumps(data, allow_nan=False)+'\n', encoding='utf-8')
    pools = [(f'grid{k}', full_factorial([3]*k).double()-1) for k in (2,3,4)]
    pools.append(('random500', 2*torch.rand(500,3,dtype=torch.float64,generator=torch.Generator().manual_seed(0))-1))
    for name, C in pools:
        F = polynomial_features(C, degree=2)
        p = F.shape[1]
        lookup = {tuple(row): i for i,row in enumerate(C.tolist())}
        sizes = [p, 2*p] + ([7] if name=='grid2' else [])
        for n in sizes:
            case = dict(pool=name, n=n, p=p, candidates=C.tolist(), rows=[])
            data['cases'].append(case)
            exact_F = [[1,*x,*(v*v for v in x),*(a*b for a,b in combinations(x,2))] for x in C.tolist()] if name.startswith('grid') else None
            if name=='grid2' and n in (6,7):
                optimum = max(determinant(exact_F, s) for s in combinations(range(9),n))
                case['exact_optimum_det'] = str(optimum)
            for method, seed in [('TorchOED',s) for s in range(5)] + [(m,None) for m in ('sequential','fedorov','modified_fedorov','detmax')]:
                row = dict(method=method, seed=seed, times_seconds=[])
                case['rows'].append(row)
                try:
                    for repeat in range(3):
                        begin = time.perf_counter()
                        result = (d_optimal_design(F,n,restarts=10,generator=torch.Generator().manual_seed(seed))
                                  if seed is not None else pydoe.optimal_design(C.numpy(),n,degree=2,criterion='D',method=method))
                        row['times_seconds'].append(time.perf_counter()-begin)
                        if repeat==0:
                            ids = result['ids'].tolist() if seed is not None else [lookup[tuple(x)] for x in result[0].tolist()]
                            row.update(ids=ids, actual_n=len(ids), repeats=len(ids)-len(set(ids)))
                            score_design(F,row,exact_F)
                            if seed is not None:
                                row.update(stop_reason=result['stop_reason'], certificate=result['certificate'])
                            if 'exact_optimum_det' in case:
                                row['attains_exact_optimum'] = len(ids)==n and not row['repeats'] and determinant(exact_F,ids)==optimum
                    row['median_seconds'] = stats.median(row['times_seconds'])
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)
                print(name,n,method,seed,row.get('logdet'),row.get('median_seconds'),row.get('error',row.get('metric_error','')),flush=True)
                save()
            scores = [r['logdet'] for r in case['rows'] if r.get('logdet') is not None and 'error' not in r]
            for row in case['rows']:
                if row.get('logdet') is not None and 'error' not in row:
                    row['D_efficiency'] = math.exp((row['logdet']-max(scores))/p)
                elif scores and row.get('exact_determinant')=='0': row['D_efficiency'] = 0.
            save()
    summary(data, out)


def summary(data, out):
    before = json.loads((ROOT/'reports/b0/results.json').read_text(encoding='utf-8')) if out.name=='b1' else None
    lines = ['# '+out.name.upper()+': fixed CPU comparison', '',
        'Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>` in the PyDOE 1.6.0 dev environment; one thread, three calls/configuration, seeds 0..4 and 10 TorchOED restarts.',
        'Grids use {-1,0,1}; random500 uses torch float64 seed 0. Shared logdet uses TorchOED quadratic features and numerical-SPD guard; rejected integer designs are checked with Fraction, not a floating determinant residual.',
        'TorchOED logdet/efficiency cells are min/median/max; time is median of per-seed three-call medians. Feature generation/shared rescoring excluded; PyDOE internal feature work included. Efficiency is relative to best found per n, including flagged repeats; n/a means no finite reference or incomplete call.',
        'Comparisons are TorchOED versus each PyDOE row (quality tolerance 1e-9); repeated IDs are not comparable. Certificates, exact attainment, IDs, errors and timings are in results.json. No new parameters or cases were tuned.']
    if before:
        lines += ['', 'B1 command appends `b1` to the command above; only the output directory/report changes. B0 median seconds and B0/current time ratios are shown beside the rerun timings for both libraries. grid2 n=12 remains infeasible (12 distinct IDs from 9 candidates); it is retained unchanged.']
    for pool, label in (('grid2','grid2: m=9, p=6; n=12 exceeds candidate count'),('grid3','grid3: m=27, p=10'),('grid4','grid4: m=81, p=15'),('random500','random500: m=500, p=10')):
        exact = [f"n={c['n']} det={c['exact_optimum_det']}" for c in data['cases'] if c['pool']==pool and 'exact_optimum_det' in c]
        lines += ['', '## '+label+('; exact optima: '+', '.join(exact) if exact else ''), '', '| n | Method | logdet | D-eff | Median s | B0 median s (B0/current) | Status / repeats / exact optimum | TorchOED quality / time versus row |', '|---|---|---|---|---|---|---|---|']
        for case in (c for c in data['cases'] if c['pool']==pool):
            tor = [r for r in case['rows'] if r['method']=='TorchOED' and 'error' not in r and r.get('logdet') is not None]
            for method in ('TorchOED','sequential','fedorov','modified_fedorov','detmax'):
                rows = [r for r in case['rows'] if r['method']==method]
                ok = [r for r in rows if 'error' not in r]
                def values(key):
                    numbers = [r.get(key) for r in ok]
                    if not numbers or None in numbers: return '-inf (exact singular)' if key=='logdet' and ok and ok[0].get('exact_determinant')=='0' else 'n/a'
                    return '/'.join(f'{x:.6g}' for x in (min(numbers),stats.median(numbers),max(numbers))) if method=='TorchOED' else f'{numbers[0]:.6g}'
                status = '; '.join(sorted({r.get('error', r.get('metric_error',r.get('stop_reason','OK'))+' / '+r.get('certificate','')+' repeats='+str(r.get('repeats',0))) for r in rows}))
                if 'exact_optimum_det' in case: status += ' / exact='+str(sum(bool(r.get('attains_exact_optimum')) for r in rows))+'/'+str(len(rows))
                comparison = 'reference' if method=='TorchOED' else 'not comparable (error, repeats, or unavailable TorchOED)'
                if method!='TorchOED' and tor and ok and not ok[0]['repeats'] and ok[0]['actual_n']==case['n']:
                    r = ok[0]; delta = stats.median(t['logdet'] for t in tor)-r['logdet'] if r.get('logdet') is not None else None
                    quality = ('better: finite vs -inf' if r.get('exact_determinant')=='0' else 'not comparable: rejected metric') if delta is None else ('equal' if abs(delta)<=1e-9 else 'better' if delta>0 else 'worse')+f' (delta {delta:.6g})'
                    seconds = stats.median(t['median_seconds'] for t in tor)
                    comparison = quality+' / '+('better' if seconds<r['median_seconds'] else 'worse' if seconds>r['median_seconds'] else 'equal')+' CPU time'
                baseline = 'n/a'
                if before:
                    old = next(c for c in before['cases'] if c['pool']==pool and c['n']==case['n'])
                    times = [r['median_seconds'] for r in old['rows'] if r['method']==method and 'median_seconds' in r]
                    if times and ok:
                        previous = stats.median(times)
                        baseline = f"{previous:.6g} ({previous/stats.median(r['median_seconds'] for r in ok):.3g}x)"
                lines.append(f"| {case['n']} | {method} | {values('logdet')} | {values('D_efficiency')} | {format(stats.median(r['median_seconds'] for r in ok),'.6g') if ok else 'n/a'} | {baseline} | {status} | {comparison} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
