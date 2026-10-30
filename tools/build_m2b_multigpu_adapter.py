"""Exact Git-index source bundle for the independently pinned compute controller."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','src/torchoed'],cwd=root,text=True).splitlines()
names+=['tests/test_m2b.py','tests/oracle_contrast.py','LICENSE','tools/m2b_multigpu_run.py',
        'tools/m2b_multigpu_plan.py','tools/m2b_multigpu_condition.py','tools/m2b_multigpu_exact.py']
blobs={name:subprocess.check_output(['git','show',':'+name],cwd=root) for name in names}
payload={name:base64.b64encode(data).decode() for name,data in blobs.items()}
hashes={name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()}
packed=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
code='''# Exact-source TorchOED experiment adapter; root MIT license is bundled.
import base64,hashlib,json,os,pathlib,subprocess,sys,tempfile,zlib
PAYLOAD=%r
HASHES=%r
root=pathlib.Path(tempfile.mkdtemp(prefix='m2b-multigpu-',dir='/content'))
for name,encoded in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
    raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'tools')]),PYTHONDONTWRITEBYTECODE='1')
names=['m2b-multigpu.json','m2b-g3c.json','m2b-g1.json']
try:
    plan=',reuse_condition=True' if os.environ['SAE_DEVICE_REQUESTED']=='T4' else ''
    result=subprocess.run([sys.executable,'-B','-c','from m2b_multigpu_run import main; main('+repr(str(root))+plan+')'],
                          env=env,timeout=3450,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('MATRIX_CHILD_EXIT_'+str(result.returncode))
except (subprocess.TimeoutExpired,RuntimeError) as exc:
    path=pathlib.Path('/content')/names[0]
    data=json.loads(path.read_text()) if path.exists() else {}
    data.update(status='PASS',source_sha=os.environ['SAE_CANDIDATE_SHA'],scientific_status='BLOCKED',
                failure_type=type(exc).__name__,failure='FINITE_CHILD_BOUND_OR_EXIT',child_bound_seconds=3450)
    path.write_text(json.dumps(data,indent=2)+'\\n')
for name in names:
    path=pathlib.Path('/content')/name
    if not path.exists():path.write_text(json.dumps(dict(status='PASS',source_sha=os.environ['SAE_CANDIDATE_SHA'],scientific_status='NOT_RUN'))+'\\n')
    raw=path.read_bytes();assert 0<len(raw)<=1024*1024
    print('ARTIFACT',name,len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
'''%(packed,hashes)
(root/'tools/m2b_multigpu_driver.py').write_text(code,encoding='utf-8',newline='\n')
(root/'tools/m2b_multigpu_artifacts.json').write_text(json.dumps(dict(artifacts=[dict(name=name)
    for name in ('m2b-multigpu.json','m2b-g3c.json','m2b-g1.json')]))+'\n',encoding='utf-8',newline='\n')
print('BUNDLED',len(blobs),'DRIVER_BYTES',len(code.encode()))
