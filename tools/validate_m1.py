"""Bounded CPU validation. Run with an installed editable package and pytest.

python tools/validate_m1.py --output <D-drive-evidence-dir> --temp <D-drive-temp>
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
                  for family, count in [('F',4),('L',4),('S',5),('P',6)]
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
assert len(torchoed.__all__)==5
assert torchoed.full_factorial((2,3)).tolist()==[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
assert torchoed.polynomial_features(torchoed.ff2n(2)).shape==(4,6)
assert torchoed.lhs(5,3,generator=torch.Generator().manual_seed(4)).shape==(5,3)
assert torchoed.sobol(3,2).shape==(3,2)
print('PASS',torchoed.__file__,torch.__version__)
"""
        run('outside_checkout_wheel_smoke', [str(wheel_python), '-c', smoke], out, 60)
        from torchoed import polynomial_features
        c64 = torch.tensor([[.2,-.3],[.7,.9]],dtype=torch.float64)
        def expected(c):
            return torch.tensor([[1,a,b,a*a,b*b,a*b] for a,b in c.tolist()],dtype=torch.float64)
        report['floating_errors'] = {}
        for dtype, tol in [(torch.float64,1e-9),(torch.float32,1e-6)]:
            c = c64.to(dtype)
            error = (polynomial_features(c).double()-expected(c)).abs().max().item()
            assert error <= tol
            report['floating_errors'][str(dtype)] = dict(max_abs_error=error,atol=tol,rtol=tol)
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','tested_sha','gates','floating_errors')},indent=2))


if __name__ == '__main__':
    main()
