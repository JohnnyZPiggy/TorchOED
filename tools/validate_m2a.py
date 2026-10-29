"""Bounded CPU validation. Run with an installed editable package and pytest.

python tools/validate_m2a.py --output <D-drive-evidence-dir> --temp <D-drive-temp>
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
        report['gates'] = {gate: ('PASS' if any(c.attrib['name'].startswith('test_'+gate+'_')
                  and not list(c) for c in cases) else 'FAIL')
                  for family, count in [('A',12)]
                  for gate in [family+str(i) for i in range(1,count+1)]}
        assert set(report['gates'].values()) == {'PASS'}
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
assert len(torchoed.__all__)==10
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
        from torchoed import d_optimality, a_optimality, e_optimality
        report['floating_errors'] = {}
        for dtype, tol in [(torch.float64,1e-9),(torch.float32,1e-5)]:
            M = torch.tensor([[2.,1.],[1.,2.]],dtype=dtype)
            errors = {name: abs(score(M).item()-expected) for name,score,expected
                      in [('D',d_optimality,math.log(3)),('A',a_optimality,-4/3),('E',e_optimality,1.)]}
            assert max(errors.values()) <= tol
            report['floating_errors'][str(dtype)] = dict(errors=errors,atol=tol,rtol=tol)
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','tested_sha','gates','floating_errors')},indent=2))


if __name__ == '__main__':
    main()
