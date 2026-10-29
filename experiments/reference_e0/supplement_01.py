"""Append-only E0-CORRECTION-01; historical evidence is read-only."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

from run_reference import determinant, digest, exact_d, WHEEL_SHA

FROZEN_HEAD = "8a4f2e8cc51653594387196c9bac95279f1bbe3d"
FROZEN_TESTED_SOURCE = "f5e9fde69fb31e5e4e9691158f213fe181c38260"
EXPECTED_DETERMINANT = 18014398509481984
SCOPE = Path(__file__).resolve().parent
REPO = SCOPE.parents[1]

def git_bytes(sha, relative):
    return subprocess.check_output(["git","-C",str(REPO),"show",sha+":"+relative],timeout=10)

def archive_integrity():
    manifest_path = "experiments/reference_e0/results/SHA256SUMS"
    frozen = git_bytes(FROZEN_HEAD,manifest_path)
    assert (REPO/manifest_path).read_bytes() == frozen
    report = []
    for line in frozen.decode().splitlines():
        expected, name = line.split("  ",1)
        relative = "experiments/reference_e0/"+name
        raw = (SCOPE/name).read_bytes()
        assert raw == git_bytes(FROZEN_HEAD,relative), name
        assert digest(raw) == expected, name
        report.append({"path":name,"sha256":expected,
            "git_blob_sha1":hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()})
    assert len(report)==10
    return {"frozen_head":FROZEN_HEAD,"original_tested_source":FROZEN_TESTED_SOURCE,
            "manifest_sha256":digest(frozen),"original_manifest_entries_verified":10,
            "original_files_unchanged_including_manifest":11,"files":report}

def features(points):
    import torch
    x = torch.as_tensor(points,dtype=torch.float64)
    assert x.ndim==2 and x.shape[1]==4
    return torch.stack([torch.ones(len(x),dtype=torch.float64),*x.T,*x.square().T,
        *(x[:,i]*x[:,j] for i,j in itertools.combinations(range(4),2))],dim=1)

def exact_gram_det(f):
    assert all(float(value).is_integer() for row in f for value in row)
    rows = [[int(value) for value in row] for row in f]
    return determinant([[sum(row[i]*row[j] for row in rows)
                         for j in range(15)] for i in range(15)])

def negative_control():
    cube=[list(row)+[row[0]*row[1]*row[2]] for row in itertools.product((-1,1),repeat=3)]
    axes=[[(2*sign if j==i else 0) for j in range(4)] for i in range(4) for sign in (-1,1)]
    # Same row ordering/center multiplicity as the observed small composite.
    return cube+[[0]*4]*2+axes+[[0]*4]*2

def p6_legal_score_pass(case):
    """Only IDs/shape/rank and D-score parity; no optimality diagnostic enters."""
    values = case["actual"]["output"]
    inputs = case["input"]
    recorded = case["comparison"]
    selected = recorded["selected_ids"]
    n = inputs["n_points"]
    pool = inputs["candidates"]
    legal = (len(values)==n and len(selected)==n and len(set(selected))==n
        and all(isinstance(i,int) and 0<=i<len(pool) for i in selected)
        and all(point==pool[i] for point,i in zip(values,selected))
        and recorded["rank"]==len(pool[0])+1)
    if not legal:
        return False
    score = float(exact_d(values))
    return abs(score-recorded["upstream_normalized_D"]) <= 1e-9*max(1,abs(score))

def run(wheel, out):
    import numpy as np
    import scipy
    import torch
    assert sys.prefix != sys.base_prefix
    archive = archive_integrity()
    wheel = wheel.resolve()
    assert digest(wheel.read_bytes())==WHEEL_SHA
    import pydoe
    assert metadata.version("pydoe")=="1.6.0"
    installed = Path(pydoe.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve())
    count=0
    with zipfile.ZipFile(wheel) as contents:
        for name in contents.namelist():
            if name.startswith("pydoe/") and name.endswith(".py"):
                assert (installed.parent.parent/name).read_bytes()==contents.read(name), name
                count += 1
    assert count==76
    source_sha=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True,timeout=10).strip()
    source_hashes={}
    for name in ("supplement_01.py","test_supplement_01.py","run_reference.py","test_reference.py"):
        raw=(SCOPE/name).read_bytes()
        assert raw==git_bytes(source_sha,"experiments/reference_e0/"+name), name
        source_hashes[name]={"sha256":digest(raw),
            "git_blob_sha1":hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()}
    original=json.loads((SCOPE/"results/cases.json").read_bytes())
    original_counts=dict(Counter(row["comparison"]["status"] for row in original))
    assert len(original)==39 and original_counts=={
        "PASS_MATCH":25,"FAIL_MISMATCH":1,"EXPECTED_EXCEPTION":2,"SOURCE_SEMANTIC_DIFFERENCE":11}
    # Exactly one new original-wheel function invocation in this focused run.
    actual=pydoe.small_composite_design(4,center=(2,2))
    f=features(actual)
    rank=int(torch.linalg.matrix_rank(f,atol=1e-9,rtol=1e-9))
    gram_det=exact_gram_det(f)
    assert actual.shape==(20,4) and tuple(f.shape)==(20,15)
    assert rank==15 and gram_det==EXPECTED_DETERMINANT
    corrected={"case_id":"P1-corrected","status":"PASS_MATCH",
        "evidence_type":"FRESH_ORIGINAL_WHEEL_CALL","called_original_pydoe":True,
        "function":"pydoe.small_composite_design","input":{"n":4,"center":[2,2]},
        "expected":{"shape":[20,4],"feature_shape":[20,15],"rank":15,"exact_gram_determinant":str(EXPECTED_DETERMINANT)},
        "actual":{"shape":list(actual.shape),"feature_shape":list(f.shape),"rank":rank,
            "exact_gram_determinant":str(gram_det),"output":actual.tolist()},
        "oracle":"Independent quadratic feature map, float64 SVD rank and exact rational determinant of integer Gram",
        "dtype":"float64","device":"cpu","atol":1e-9,"rtol":1e-9}
    control=features(negative_control())
    control_rank=int(torch.linalg.matrix_rank(control,atol=1e-9,rtol=1e-9))
    assert control_rank==12 and exact_gram_det(control)==0
    negative={"case_id":"P1-different-D-equals-ABC","status":"PASS_MATCH",
        "evidence_type":"INDEPENDENT_MATH_CONTROL","called_original_pydoe":False,
        "expected_rank":12,"actual_rank":control_rank,"shape":[20,4],"feature_shape":[20,15],
        "exact_gram_determinant":"0","output":negative_control(),
        "interpretation":"Different fractional cube with identical axial/center pattern; not a PyDOE failure."}
    by_id={row["case_id"]:row for row in original}
    p4=[]
    for level in range(4):
        ids=[f"P4-L{level}-{grid}" for grid in ("clenshaw_curtis","chebyshev","gauss_patterson")]
        source=[by_id[cid] for cid in ids]
        assert all(row["actual"]["called_original_pydoe"] for row in source)
        arrays=[row["actual"]["output"] for row in source]
        assert arrays[0]==arrays[1]==arrays[2]
        p4.append({"case_id":f"P4-L{level}-grid-type","evidence_type":"DERIVED_FROM_ACTUAL_CASES",
            "called_original_pydoe":False,"derived_from":ids,"coordinates_identical":True,
            "status":"SOURCE_SEMANTIC_DIFFERENCE",
            "supersedes_interpretation_only":"Original ambiguous called_original_pydoe flag; original row is unchanged."})
    p6=[]
    for row in original:
        if row["case_id"].startswith("P6-"):
            assert p6_legal_score_pass(row)
            diag=row["comparison"]
            p6.append({"case_id":row["case_id"],"evidence_type":"DERIVED_FROM_ACTUAL_CASES",
                "called_original_pydoe":False,"derived_from":[row["case_id"]],
                "PASS_MATCH_meaning":"Candidate-ID legality and normalized D-score agreement only",
                "legal_score_pass":True,
                "separate_diagnostics":{key:diag[key] for key in
                    ("one_swap_local_exact","global_gap","exhaustive_global_D")},
                "general_optimality_claim":False})
    assert len(p6)==8
    document={"supplement":"E0-CORRECTION-01","status":"PASS","tested_source_sha":source_sha,
        "source_hashes":source_hashes,"original_archive":archive,
        "historical_evidence":{"records":39,"direct_function_or_import_provenance_rows":35,
            "derived_p4_comparisons":4,"unchanged_status_counts":original_counts},
        "new_function_invocations":1,"new_import_provenance":"Fresh import, version and installed-wheel byte verification",
        "wheel":{"filename":wheel.name,"sha256":WHEEL_SHA,"bytes":wheel.stat().st_size,
            "installed_source_files_verified":count},
        "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
            "torch":torch.__version__,"pydoe":metadata.version("pydoe"),"pydoe_file":str(installed),
            "platform":platform.platform(),"cpu":platform.processor(),"seed":20261029,"device":"cpu"},
        "corrected_p1":corrected,"different_negative_control":negative,
        "p4_provenance_corrections":p4,"p6_pass_semantics":p6,
        "legacy_comparator":"Unchanged; documented exit 1 from original P1 is not an all-pass result.",
        "legacy_comparator_rerun":"NOT_RUN: no legacy runner changes; focused actual-wheel regression plus both test suites are fresh.",
        "gpu":"NOT_RUN","performance":"NOT_RUN","merge":"NO_MERGE"}
    # Public evidence uses a task-root token, never private workspace paths.
    safe=json.dumps(document,indent=2,allow_nan=False).replace(str(wheel.parent.parent),"<E0_TASK_ROOT>")
    assert not out.exists(), "Append-only: choose a new supplement output file"
    out.write_bytes((safe+"\n").encode())
    print(json.dumps({"supplement":"E0-CORRECTION-01","status":"PASS",
        "tested_source_sha":source_sha,"fresh_original_wheel_calls":1,
        "P1_rank":rank,"P1_exact_determinant":str(gram_det),"different_control_rank":control_rank,
        "P4_derived_rows":len(p4),"P6_separate_diagnostics":len(p6),
        "original_manifest":"10/10 PASS; all original 11 files unchanged","output_sha256":digest(out.read_bytes())}))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.wheel,args.out))

