"""Bounded CPU evidence; uses installed dependencies, no downloads or GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--explore', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', M2B_ERRORS=str(out/'oracle-errors.json'))
    target = 'tests/test_m2b.py' if args.explore else 'tests'
    cmd = [sys.executable, '-B', '-m', 'pytest', target, '-q', '-p', 'no:cacheprovider',
           '--junitxml='+str(out/'junit.xml')]
    start = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=240)
        code, log = result.returncode, result.stdout+result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        log = ''.join(x.decode(errors='replace') if isinstance(x, bytes) else (x or '')
                      for x in (exc.stdout, exc.stderr))+'\nTIMEOUT\n'
    (out/'pytest.log').write_text(log, encoding='utf-8')
    import torch
    report = dict(source_sha=source, status='PASS' if code == 0 else 'FAIL',
                  phase='EXPLORE' if args.explore else 'VALIDATE', python=sys.version,
                  torch=torch.__version__, os=platform.system(), command=cmd,
                  timeout_seconds=240, exit_code=code, elapsed_seconds=time.monotonic()-start,
                  source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [repo/'src/torchoed/criteria.py', repo/'src/torchoed/__init__.py',
                              repo/'tests/test_m2b.py', repo/'tests/oracle_contrast.py']})
    (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(log)
    print(json.dumps(report, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
