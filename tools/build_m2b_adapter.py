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
import base64, hashlib, json, os, pathlib, sys, tempfile, zlib
PAYLOAD = %r
HASHES = %r
root=pathlib.Path(tempfile.mkdtemp(prefix='m2b-source-',dir='/content'))
for name,encoded in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
    raw=base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest()==HASHES[name]
    path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
sys.path[:0]=[str(root/'src'),str(root/'tools')]
from m2b_gpu_matrix import main
path=main(root)
raw=path.read_bytes()
print('ARTIFACT',path.name,len(raw),hashlib.sha256(raw).hexdigest())
''' % (packed,hashes)
(repo/'tools/m2b_gpu_driver.py').write_text(driver,encoding='utf-8',newline='\n')
(repo/'tools/m2b_artifacts.json').write_text('{"artifacts":[{"name":"m2b-gpu.json"}]}\n',encoding='utf-8')
print(json.dumps(hashes,indent=2))
