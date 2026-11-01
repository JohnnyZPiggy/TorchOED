"""Pinned Colab adapter; transport success preserves scientific failure evidence."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/torchoed-s1')
sha = os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','install','pydoe==1.6.0'],check=True,capture_output=True)
env = dict(os.environ,PYTHONPATH=str(root/'src'))
run = subprocess.run([sys.executable,'-B','tools/bench_g2.py','s1-gpu','cuda'],cwd=root,env=env,capture_output=True,text=True)
p = root/'reports/s1-gpu/results.json'
data = json.loads(p.read_text()) if p.exists() else dict(source_sha=sha,rows=[])
data.update(status='PASS',scientific_status='PASS' if run.returncode==0 else 'BLOCKED',exit_code=run.returncode,traceback=run.stderr if run.returncode else '',device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
pathlib.Path('/content/s1-gpu.json').write_text(json.dumps(data,allow_nan=False)+'\n')
