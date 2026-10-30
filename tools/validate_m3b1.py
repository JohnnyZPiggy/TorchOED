"""Bounded CPU validation. Run with an installed editable package and pytest.

python tools/validate_m3b1.py --output <D-drive-evidence-dir> --temp <D-drive-temp>
Uses the current interpreter; never installs dependencies or starts GPU/CI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--temp', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, temp = args.output.resolve(), args.temp.resolve()
    out.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(TMP=str(temp), TEMP=str(temp), PYTHONDONTWRITEBYTECODE='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=str(out/'pip-cache'))
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=root, text=True).strip()
    def run(name, argv, cwd, timeout):
        start = time.time()
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout)
        (out/(name+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        record = dict(name=name, argv=argv, cwd=str(cwd), timeout_seconds=timeout,
                      elapsed_seconds=time.time()-start, exit_code=result.returncode)
        commands.append(record)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see log')
    commands = []
    sources = {p: hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+p],
                cwd=root)).hexdigest() for p in git('ls-files').splitlines()
               if p.startswith(('src/', 'tests/', 'tools/')) or p == 'pyproject.toml'}
    report = dict(tested_sha=git('rev-parse', 'HEAD'), source_sha256=sources,
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  torch=torch.__version__, cuda_build=torch.version.cuda,
                  device='cpu', commands=commands, scientific_performance='NOT_RUN',
                  gpu='NOT_RUN_USER_CPU_ONLY', actions='NOT_RUN_USER_PROHIBITED',
                  environment='fresh task venv with system-site-packages; not hermetic')
    try:
        run('pytest', [sys.executable, '-m', 'pytest', '-q',
                      '--junitxml='+str(out/'pytest.xml')], root, 180)
        cases = ET.parse(out/'pytest.xml').findall('.//testcase')
        report['gates'] = {gate: ('PASS' if any(c.attrib['name'].startswith('test_'+gate+'_') and c.attrib['classname']=='tests.test_m3b1'
                  and not list(c) for c in cases) else 'FAIL')
                  for family, count in [('B',12)]
                  for gate in [family+str(i) for i in range(1,count+1)]}
        report['gates'].update(B3='NOT_RUN',B5='NOT_RUN',B11='NOT_APPLICABLE')
        assert all(v == 'PASS' for k,v in report['gates'].items() if k not in ('B3','B5','B11'))
        report['pytest_testcases'] = len(cases)
        run('wheel_build', [sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                           '--no-build-isolation', '--wheel-dir', str(out/'wheel'), '.'], root, 120)
        wheel = next((out/'wheel').glob('torchoed-*.whl'))
        report['wheel_sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        wheel_env = out/'wheel-venv'
        run('wheel_venv', [sys.executable, '-m', 'venv', '--system-site-packages',
                          str(wheel_env)], out, 120)
        wheel_python = wheel_env/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run('wheel_install', [str(wheel_python), '-m', 'pip', 'install', '--no-deps',
                             str(wheel)], out, 120)
        smoke = """import importlib.util,torch,torchoed
assert importlib.util.find_spec('pydoe') is None
assert 'site-packages' in torchoed.__file__
assert len(torchoed.__all__)==12
s2=torchoed.score_two_swaps(torch.tensor([[1.,0.],[0.,1.],[.9,.9],[.9,-.9]],dtype=torch.float64),torch.tensor([0,1]))
assert s2['evaluated_pairs']==1 and s2['status'].tolist()==[0]
assert abs(s2['ratios'][0].item()-2.6244)<1e-9
r=torchoed.fedorov_exchange(torch.tensor([[1.,0.],[0.,1.],[2.,0.]],dtype=torch.float64),torch.tensor([0,1]))
assert r['ids'].tolist()==[2,1]
assert r['stop_reason']=='LOCAL_1_SWAP_NUMERICAL'
assert torchoed.exact_information(torch.tensor([[1.,-1.],[1.,1.]],dtype=torch.float64),torch.tensor([0,1])).tolist()==[[2.,0.],[0.,2.]]
M=torch.tensor([[2.,0.],[0.,8.]],dtype=torch.float64)
assert abs(torchoed.d_optimality(M).item()-2.772588722239781)<1e-9
assert abs(torchoed.a_optimality(M).item()+.625)<1e-9
assert torchoed.e_optimality(M).item()==2
assert torchoed.weighted_information(torch.eye(2,dtype=torch.float64),torch.tensor([.5,.5],dtype=torch.float64)).tolist()==[[.5,0.],[0.,.5]]
assert torchoed.full_factorial((2,3)).tolist()==[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
assert torchoed.polynomial_features(torchoed.ff2n(2)).shape==(4,6)
assert torchoed.lhs(5,3,generator=torch.Generator().manual_seed(4)).shape==(5,3)
assert torchoed.sobol(3,2).shape==(3,2)
print('PASS',torchoed.__file__,torch.__version__)
"""
        run('outside_checkout_wheel_smoke', [str(wheel_python), '-c', smoke], out, 60)
        import math
        import runpy
        from fractions import Fraction
        from itertools import combinations
        from torchoed import score_two_swaps
        oracle = runpy.run_path(str(root/'tests/oracle_two_swap.py'))
        gram,det,cases = oracle['exact_gram'],oracle['bareiss_det'],oracle['pair_cases']
        def features(rows):
            return torch.tensor([[float(Fraction(str(x))) for x in r] for r in rows],dtype=torch.float64)
        rows,selected = oracle['INTEGER11'],list(range(6))
        result = score_two_swaps(features(rows),torch.tensor(selected),tile_size=17)
        denominator = det(gram(rows,selected))
        records=[]
        for i,(positions,incoming,trial,matrix,numerator) in enumerate(cases(rows,selected)):
            ratio=float(numerator/denominator)
            actual=result['ratios'][i].item()
            assert result['status'][i].item()==0 and numerator>0
            assert math.isclose(actual,ratio,abs_tol=1e-9,rel_tol=1e-9)
            records.append(dict(outgoing_positions=positions,incoming_ids=incoming,
                exact_trial_gram=[[str(v) for v in r] for r in matrix],
                exact_determinant=str(numerator),exact_ratio=str(numerator/denominator),
                ratio=actual,ratio_abs_error=abs(actual-ratio),
                gain=result['gains'][i].item(),gain_abs_error=abs(result['gains'][i].item()-math.log(ratio)),
                status='VALID'))
        assert len(records)==150
        trap=oracle['TRAP']
        table=[dict(ids=list(s),exact_determinant=str(det(gram(trap,s)))) for s in combinations(range(4),2)]
        boundaries=[]
        for name,values in [('singular',[[1,0],[0,1],[1,0],[2,0]]),
                            ('near_singular_spd',[[1,0],[0,1],[1,0],[1,'1/10000000000']])]:
            actual=score_two_swaps(features(values),torch.tensor([0,1]))
            boundaries.append(dict(name=name,exact_trial_determinant=str(det(gram(values,[2,3]))),
                status=actual['status_names'][actual['status'][0].item()],reason=actual['reasons'][0],
                ratio=actual['ratios'][0].item(),gain_defined=bool(torch.isfinite(actual['gains'][0]))))
        diagnostic=dict(total_candidates=11,selected_count=6,incoming_count=5,legal_pairs=150,
            initial_gram=[[str(v) for v in r] for r in gram(rows,selected)],
            exact_initial_determinant=str(denominator),pairs=records,
            trap_six_subsets=table,boundaries=boundaries,
            B3='NOT_RUN',B5='NOT_RUN',B11='NOT_APPLICABLE',search='NOT_IMPLEMENTED')
        (out/'oracle-diagnostics.json').write_text(json.dumps(diagnostic,indent=2)+'\n',encoding='utf-8')
        report['floating_errors'] = dict(max_ratio_abs_error=max(x['ratio_abs_error'] for x in records),
            max_gain_abs_error=max(x['gain_abs_error'] for x in records),atol=1e-9,rtol=1e-9,dtype='float64')
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','tested_sha','gates','floating_errors')},indent=2))


if __name__ == '__main__':
    main()
