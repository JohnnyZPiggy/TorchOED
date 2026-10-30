"""Bundle exact local Git source for the controller; no network/install on GPU."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

repo=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','src/torchoed'],cwd=repo,text=True).splitlines()
names += ['tests/test_m2b.py','tests/oracle_contrast.py','tools/m2b_gpu_matrix.py','LICENSE']
blobs={name:subprocess.check_output(['git','show',':'+name],cwd=repo) for name in names}
payload={name:base64.b64encode(raw).decode() for name,raw in blobs.items()}
hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in blobs.items()}
packed=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
driver='''# Self-contained exact-source adapter; original TorchOED MIT license included.
import base64, hashlib, json, os, pathlib, sys, tempfile, zlib, subprocess
PAYLOAD = %r
HASHES = %r
root=pathlib.Path(tempfile.mkdtemp(prefix='m2b-source-',dir='/content'))
for name,encoded in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
    raw=base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest()==HASHES[name]
    path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'tools')]))
path=pathlib.Path('/content/m2b-gpu.json')
try:
    result=subprocess.run([sys.executable,'-B','-c',
        'from m2b_gpu_matrix import main; main('+repr(str(root))+')'],
        env=env,timeout=410,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('MATRIX_CHILD_EXIT_'+str(result.returncode))
except (subprocess.TimeoutExpired,RuntimeError) as exc:
    data=json.loads(path.read_text()) if path.exists() else dict(source_sha=os.environ['SAE_CANDIDATE_SHA'],G1={},G2=[],G3=[],G4=[],G5=[])
    data.update(status='PASS',scientific_status='BLOCKED',failure_type=type(exc).__name__,
                failure='FINITE_MATRIX_CHILD_BOUND_OR_EXIT',child_timeout_seconds=410,
                uncompleted_configs='NOT_RUN')
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\\n')
raw=path.read_bytes()
print('ARTIFACT',path.name,len(raw),hashlib.sha256(raw).hexdigest())
''' % (packed,hashes)
(repo/'tools/m2b_gpu_driver.py').write_text(driver,encoding='utf-8',newline='\n')
tail=driver.replace("main('+repr(str(root))+')'", "main('+repr(str(root))+',start_p=100,start_B=18150)'")
assert tail!=driver
(repo/'tools/m2b_tail_driver.py').write_text(tail,encoding='utf-8',newline='\n')
(repo/'tools/m2b_artifacts.json').write_text('{"artifacts":[{"name":"m2b-gpu.json"}]}\n',encoding='utf-8')
print(json.dumps(hashes,indent=2))
