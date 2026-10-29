from pathlib import Path
import hashlib,json,sys
import torch,pytest
root=Path(__file__).parent
f=torch.tensor([[1.,-1.],[1.,0.],[1.,1.]],dtype=torch.float64)
actual=f.mT@f
expected=torch.tensor([[3.,0.],[0.,2.]],dtype=torch.float64)
torch.testing.assert_close(actual,expected,atol=0,rtol=0)
p=(root/"scratch"/"warm-roundtrip.json").resolve()
assert str(p).startswith("D:") and not p.exists()
raw=json.dumps(actual.tolist()).encode()
p.write_bytes(raw)
assert p.read_bytes()==raw
checksum=hashlib.sha256(p.read_bytes()).hexdigest()
p.unlink()
assert not p.exists()
result={"status":"PASS","python":sys.version.split()[0],"torch":torch.__version__,"pytest":pytest.__version__,
        "device":"cpu","dtype":"float64","max_abs_error":(actual-expected).abs().max().item(),
        "roundtrip_path":str(p),"roundtrip_sha256":checksum,"cleanup_confirmed":True,"gpu":"NOT_RUN_USER_CPU_ONLY"}
(root/"evidence"/"warm-local.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
print(json.dumps(result))
