"""Frozen 99-cell covering-array size and independent coverage evidence."""
import csv
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path
import torch
from torchoed import covering_array

ROOT = Path(__file__).resolve().parent
TV = [(2,v) for v in range(3,7)]+[(3,v) for v in range(2,7)]
KS = [4,5,6,8,10,13,16,20,30,50,100]

def save(name, data):
    (ROOT/name).write_text(json.dumps(data, indent=2)+'\n', newline='\n')

def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    if '--export' in sys.argv:
        arrays = {}
        for t,v in TV:
            for k in KS:
                arrays[f'{t}.{v}.{k}'] = covering_array(k,v,strength=t).tolist()
                print(t,v,k,len(arrays[f'{t}.{v}.{k}']),flush=True)
        save('arrays.json', arrays)
        return
    arrays = json.loads((ROOT/'arrays.json').read_text())
    runtime = json.loads((ROOT/'runtime.json').read_text())
    refs = {tuple(int(r[x]) for x in ('t','v','k')):r for r in csv.DictReader((ROOT/'ecan.csv').open())}
    checks = {tuple(int(r[x]) for x in ('t','v','k')):r for r in csv.DictReader((ROOT/'verify.csv').open())}
    cells = []
    for name, rows in arrays.items():
        t,v,k = map(int,name.split('.'))
        columns = list(zip(*rows))
        covered = all(len(set(zip(*(columns[j] for j in cols)))) == v**t for cols in combinations(range(k),t))
        assert covered, name
        ref, check = refs[t,v,k], checks[t,v,k]
        n = None if ref['eCAN'] in ('NA','') else int(ref['eCAN'])
        cells.append(dict(t=t,v=v,k=k,N_H=len(rows),eCAN=n,ratio=None if n is None else round(len(rows)/n,6),
            covered=check['covered']=='TRUE',python_coverage=covered,error=ref.get('error','') or check.get('error',''),
            optimal_consistency=None if n is None or len(rows)!=v**t else n==v**t))
    ratios = [c['ratio'] for c in cells if c['ratio'] is not None]
    result = dict(workflow_run_id=runtime['run_id'],source_sha=runtime['source_sha'],
        versions=dict(R=runtime['R_version'],CAs=runtime['packages'].get('CAs','NOT_RUN'),
        caverify=runtime['packages'].get('caverify','NOT_RUN'),Python=platform.python_version(),torch=torch.__version__),
        max_ratio=max(ratios,default=None),within_1_3=sum(r<=1.3 for r in ratios),cells=cells)
    save('results.json',result)
    lines = ['# W17 covering-array sizes vs best-known','',f"R run {runtime['run_id']}, frozen source `{runtime['source_sha']}`.",
        'Versions: '+json.dumps(result['versions'])+'.','',
        'Colbourn November 2024 snapshot via pinned CAs 0.24-3; ratios reported as-is.',
        f"Python coverage: {sum(c['python_coverage'] for c in cells)}/99 PASS; R covered: {sum(c['covered'] for c in cells)}/99; ratios <= 1.3: {result['within_1_3']}/99.",'',
        '| t,v | '+' | '.join(map(str,KS))+' | max |','|---|'+'---|'*12]
    for t,v in TV:
        cs = [c for c in cells if (c['t'],c['v'])==(t,v)]
        rs = [c['ratio'] for c in cs if c['ratio'] is not None]
        lines.append(f'| {t},{v} | '+' | '.join('NOT_RUN' if c['ratio'] is None else f"{c['ratio']:.6f}" for c in cs)+f" | {max(rs) if rs else 'NOT_RUN'} |")
    errors = sorted({c['error'] for c in cells if c['error']})
    lines += ['', 'Errors (verbatim): '+json.dumps(errors),
        'Index-one consistency failures: '+json.dumps([f"{c['t']}.{c['v']}.{c['k']}" for c in cells if c['optimal_consistency'] is False]),
        'Scope: CPU evidence only; no constructor changes, tuning, held-out cases or timing claims.']
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n',newline='\n')
    print(f'Coverage 99 PASS 0 FAIL; {len(ratios)} ratios; wall {time.monotonic()-start:.6f} s')

if __name__ == '__main__':
    main()
