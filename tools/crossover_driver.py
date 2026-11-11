"""Pinned W5 Colab adapter; transport success preserves row failures."""
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/torchoed-crossover')
sha = os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','install','pydoe==1.6.0','scipy==1.18.1'],check=True,capture_output=True)
run = subprocess.run([sys.executable,'-B','tools/bench_crossover.py','--run'],cwd=root,capture_output=True,text=True)
p = root/'reports/w5-crossover/run2-results.json'
data = json.loads(p.read_text()) if p.exists() else dict(source_sha=sha,rows=[])
data.update(status='PASS',scientific_status='PASS' if run.returncode==0 else 'ERROR',exit_code=run.returncode,
    traceback=run.stderr if run.returncode else '',device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
pathlib.Path('/content/w5-crossover.json').write_text(json.dumps(data,allow_nan=False)+'\n')
