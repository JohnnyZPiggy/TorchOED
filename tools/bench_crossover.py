"""W5-M4: fixed crossover calls; --smoke runs the required two CPU rows."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import importlib
import json
import math
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
import bench_b0 as b

torch = b.torch
torch.set_num_threads(1)
MAX_ELEMENTS = 2**27
METHODS = ('fedorov', 'modified_fedorov', 'detmax', 'cpu', 'cuda')


def call(k, m, method):
    C = 2*torch.rand(m, k, dtype=torch.float64, generator=torch.Generator().manual_seed(0))-1
    F = b.polynomial_features(C, 2, max_elements=MAX_ELEMENTS)
    n = 2*F.shape[1]
    if method == 'cuda':
        design = importlib.import_module('torchoed.d_optimal_design')
        original = design.k2_exchange
        def cuda_exchange(F, ids, **kwargs):
            result = original(F.cuda(), ids.cuda(), **kwargs)
            result['ids'] = result['ids'].cpu()
            return result
        design.k2_exchange = cuda_exchange
        torch.cuda.synchronize()
    begin = time.perf_counter()
    if method in ('cpu', 'cuda'):
        result = b.optimal_design(F, n, criterion='D', restarts=1, generator=torch.Generator().manual_seed(0), max_elements=MAX_ELEMENTS)
        ids = result['ids'].tolist()
    else:
        result = b.pydoe.optimal_design(C.numpy(), n, degree=2, criterion='D', method=method)
    if method == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    if method not in ('cpu', 'cuda'):
        lookup = {tuple(x): i for i, x in enumerate(C.tolist())}
        ids = [lookup[tuple(x)] for x in result[0].tolist()]
    row = dict(k=k, m=m, method=method, seconds=seconds, ids=ids, repeats=len(ids)-len(set(ids)))
    b.score_design(F, row, None)
    if method in ('cpu', 'cuda'):
        dense = torch.linalg.slogdet(F[ids].mT@F[ids]).logabsdet.item()
        row.update(objective=result['objective'], returned_floor=result['efficiency_lower_bound'])
        floor = row['d_efficiency_lower_bound']
        row['oracles'] = dict(objective=abs(result['objective']-dense) <= 1e-9,
            kw_floor=abs(math.exp(-result['efficiency_gap_bound']/F.shape[1])-floor) <= 1e-9,
            refined_floor=floor-1e-9 <= result['efficiency_lower_bound'] <= 1+1e-9)
        assert all(row['oracles'].values()), row
    else:
        row['oracles'] = dict(exact_mapping=len(ids) == n)
        assert all(row['oracles'].values()), row
    row['status'] = 'PASS'
    return row


def run(smoke):
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       **{name: version(name) for name in ('pydoe', 'numpy', 'scipy')}, threads=1)
    assert environment['pydoe'] == '1.6.0' and environment['scipy'] == '1.18.1'
    if not smoke:
        environment.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
            cpu=next(line for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')))
    data = dict(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=b.ROOT, text=True).strip(),
                environment=environment, rows=[])
    out = b.ROOT/('scratch/smoke.json' if smoke else 'reports/w5-crossover/run2-results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    def child(k, m, method):
        try:
            result = subprocess.run([sys.executable, '-B', __file__, '--call', str(k), str(m), method],
                                    capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return dict(k=k, m=m, method=method, status='TIMEOUT')
        if result.returncode:
            return dict(k=k, m=m, method=method, status='ERROR', error=result.stderr.strip().splitlines()[-1])
        return json.loads(result.stdout)
    if not smoke:
        data['pydoe_gate'] = child(3, 200, 'fedorov')
        out.write_text(json.dumps(data, allow_nan=False)+'\n', encoding='utf-8')
        assert data['pydoe_gate']['status'] == 'PASS', data['pydoe_gate']
        data['untimed_warmup'] = child(3, 1000, 'cuda')
    timeouts = {}
    for m in ((200,) if smoke else (1000, 10000, 100000)):
        for k in ((3,) if smoke else (3, 5)):
            for method in (('cpu', 'fedorov') if smoke else METHODS):
                row = (dict(k=k, m=m, method=method, status=f'SKIPPED (TIMEOUT at m={timeouts[k, method]})')
                       if (k, method) in timeouts else child(k, m, method))
                if row['status'] == 'TIMEOUT':
                    timeouts[k, method] = m
                data['rows'].append(row)
                out.write_text(json.dumps(data, allow_nan=False)+'\n', encoding='utf-8')
                print(json.dumps(row), flush=True)
    if not smoke:
        summary(data)
    if smoke:
        assert all(row['status'] == 'PASS' for row in data['rows']), data['rows']


def summary(data):
    lines = ['# W5 scale crossover run 2', '', 'Measured code SHA: '+data['source_sha'],
        'One Colab host; float64, candidate/initialization seed 0, n=2p, one restart; 1800s subprocess wall cap.',
        'One untimed CUDA warmup; transfers included; CUDA synchronized before timer stop. Search timing excludes shared rescoring.',
        'floor is shared KW; torchoed_floor is the refined returned bound. No retries or tuning.',
        'Environment: '+json.dumps(data['environment']), '',
        '| k | m | method | seconds/status | logdet | floor | torchoed_floor | repeats |',
        '|---|---|---|---|---|---|---|---|']
    for r in data['rows']:
        fields = [r['k'], r['m'], r['method'], r.get('seconds', r['status']), r.get('logdet', ''),
                  r.get('d_efficiency_lower_bound', ''), r.get('returned_floor', ''), r.get('repeats', '')]
        lines.append('| '+' | '.join(map(str, fields))+' |')
    for k in (3, 5):
        for variant in ('cpu', 'cuda'):
            winner, unavailable, inconclusive = None, None, []
            for m in (1000, 10000, 100000):
                cell = [r for r in data['rows'] if r['k']==k and r['m']==m]
                t = next(r for r in cell if r['method']==variant)
                pydoe = [r for r in cell if r['method'] not in ('cpu', 'cuda')]
                refs = [r for r in pydoe if r['status']=='PASS']
                if all(r['status']=='ERROR' for r in pydoe): inconclusive.append(m)
                if t['status']!='PASS': continue
                if not refs and any(r['status']=='TIMEOUT' or r['status'].startswith('SKIPPED') for r in pydoe):
                    if unavailable is None: unavailable = m
                elif refs and t['seconds'] < min(r['seconds'] for r in refs) and all(
                        t['logdet'] >= r['logdet']-1e-9*max(1,abs(r['logdet'])) for r in refs):
                    if winner is None: winner = m
            lines.append(f'k={k}, {variant}: smallest crossover m={winner}' if winner else
                f'k={k}, {variant}: faster at m={unavailable}; PyDOE quality unavailable (no quality win)' if unavailable else
                f'no crossover found (k={k}, {variant})')
            lines.append(f'k={k}, {variant}: inconclusive m={inconclusive}')
    lines.append('measurement only; no claim beyond these 6 cells')
    (b.ROOT/'reports/w5-crossover/run2-summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1] == '--call':
        print(json.dumps(call(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]), allow_nan=False))
    else:
        run(sys.argv[1] == '--smoke')
