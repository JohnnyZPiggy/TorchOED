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
from torchoed import d_optimal_design, full_factorial, polynomial_features
from oracle_exchange import determinant


def main():
    torch.set_num_threads(1)
    wheel = Path(sys.argv[1]).read_bytes()
    digest = hashlib.sha256(wheel).hexdigest()
    assert version('pydoe') == '1.6.0'
    assert digest == '5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec'
    out = ROOT/'reports/b0'
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
            if name=='grid2' and n in (6,7):
                exact_F = [[1,*x,*(v*v for v in x),x[0]*x[1]] for x in C.tolist()]
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
                            X = F[ids]
                            sign, score = torch.linalg.slogdet(X.mT@X)
                            row.update(ids=ids, actual_n=len(ids), repeats=len(ids)-len(set(ids)),
                                logdet=score.item() if sign.item()>0 and math.isfinite(score.item()) else None)
                            if seed is not None:
                                row.update(stop_reason=result['stop_reason'], certificate=result['certificate'])
                            if 'exact_optimum_det' in case:
                                row['attains_exact_optimum'] = len(ids)==n and not row['repeats'] and determinant(exact_F,ids)==optimum
                    row['median_seconds'] = stats.median(row['times_seconds'])
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)
                print(name,n,method,seed,row.get('logdet'),row.get('median_seconds'),row.get('error',''),flush=True)
                save()
            scores = [r['logdet'] for r in case['rows'] if r.get('logdet') is not None and 'error' not in r]
            for row in case['rows']:
                if row.get('logdet') is not None and 'error' not in row:
                    row['D_efficiency'] = math.exp((row['logdet']-max(scores))/p)
            save()
    summary(data, out)


def summary(data, out):
    lines = ['# B0: fixed CPU comparison', '',
        'Command: `python tools/bench_b0.py <pydoe-1.6.0-py3-none-any.whl>`; use the dev-only PyDOE 1.6.0 environment.',
        'One thread; 3 calls per configuration; TorchOED seeds 0..4, 10 restarts. Times exclude feature generation and shared rescoring; PyDOE internal feature work is included.',
        'Grid coordinates are {-1,0,1}; random500 uses CPU torch float64 seed 0. D-efficiency uses the best found per n, including flagged replicate designs.',
        'TorchOED cells: logdet min/median/max; D-eff min/median/max; time = median of per-seed three-call medians. Errors are not counted as quality results.',
        'All returned rows use exact candidate matches; repeated IDs are scored but not comparable to no-replacement designs. See results.json for IDs, all errors, timings and certificates.']
    for pool in ('grid2','grid3','grid4','random500'):
        lines += ['', '## '+pool, '| n | Method | logdet | D-efficiency | Median seconds | Status / repeats |', '|---|---|---|---|---|---|']
        for case in (c for c in data['cases'] if c['pool']==pool):
            if 'exact_optimum_det' in case:
                lines.append(f"Exact oracle n={case['n']}: det={case['exact_optimum_det']}; attaining methods: "+', '.join(sorted({r['method'] for r in case['rows'] if r.get('attains_exact_optimum')}))+'.')
            for method in ('TorchOED','sequential','fedorov','modified_fedorov','detmax'):
                rows = [r for r in case['rows'] if r['method']==method]
                ok = [r for r in rows if 'error' not in r and r.get('logdet') is not None]
                values = lambda key: '/'.join(f'{x:.6g}' for x in (min(r[key] for r in ok),stats.median(r[key] for r in ok),max(r[key] for r in ok))) if method=='TorchOED' else f"{ok[0][key]:.6g}"
                status = '; '.join(sorted({r.get('error', r.get('stop_reason','OK')+' / repeats='+str(r.get('repeats',0))) for r in rows}))
                lines.append(f"| {case['n']} | {method} | {values('logdet') if ok else 'n/a'} | {values('D_efficiency') if ok else 'n/a'} | {stats.median(r['median_seconds'] for r in ok) if ok else 'n/a'} | {status} |")
    lines += ['', '## Better, equal, worse (quality tolerance 1e-9 logdet; measured time only)']
    for case in data['cases']:
        tor = [r for r in case['rows'] if r['method']=='TorchOED' and 'error' not in r]
        comparisons = []
        for r in case['rows']:
            if r['method']=='TorchOED' or 'error' in r or r.get('logdet') is None: continue
            if not tor or r['repeats'] or r['actual_n']!=case['n']: comparisons.append(r['method']+': not comparable'); continue
            delta = stats.median(t['logdet'] for t in tor)-r['logdet']
            quality = 'equal' if abs(delta)<=1e-9 else 'better' if delta>0 else 'worse'
            seconds = stats.median(t['median_seconds'] for t in tor)
            comparisons.append(f"{r['method']}: quality {quality} (delta {delta:.6g}); time {'better' if seconds<r['median_seconds'] else 'worse' if seconds>r['median_seconds'] else 'equal'} ({seconds:.6g} vs {r['median_seconds']:.6g}s)")
        lines.append(f"- {case['pool']} n={case['n']}: "+('; '.join(comparisons) or 'no completed comparable results')+'.')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
