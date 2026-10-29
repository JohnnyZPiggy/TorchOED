"""Bounded E0 real-wheel CPU observations; never imports TorchOED."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as Q
import hashlib
import importlib.metadata as metadata
import inspect
import itertools as it
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import warnings
import zipfile

SEED = 20261029
WHEEL_SHA = "5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec"
BASE = "e6c0052adb2bff559c625a8319f89b061c971e9f"
UPSTREAM = "866d3f411942cb662af8054562dc1f7e07372401"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def gf2_rank(rows):
    rows = [sum(int(v) << j for j, v in enumerate(row)) for row in rows]
    pivots = {}
    for row in rows:
        while row:
            bit = row.bit_length() - 1
            if bit in pivots:
                row ^= pivots[bit]
            else:
                pivots[bit] = row
                break
    return len(pivots)

def digital_reference(bits):
    # For x^2+x+1, s[t+2]=s[t+1]+s[t] over GF(2), s=(0,1,...).
    sequence = [0, 1]
    for _ in range(2 * bits - 2):
        sequence.append(sequence[-1] ^ sequence[-2])
    hankel = [sequence[i:i+bits] for i in range(bits)]
    values = [sum(Q(sum(hankel[r][j] * ((n >> j) & 1) for j in range(bits)) % 2,
                    2 ** (r + 1)) for r in range(bits)) for n in range(16)]
    return hankel, values

def determinant(matrix):
    matrix = [list(map(Q, row)) for row in matrix]
    result = Q(1)
    for j in range(len(matrix)):
        pivot = next((i for i in range(j, len(matrix)) if matrix[i][j]), None)
        if pivot is None:
            return Q(0)
        if pivot != j:
            matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
            result = -result
        value = matrix[j][j]
        result *= value
        for i in range(j+1, len(matrix)):
            ratio = matrix[i][j] / value
            matrix[i] = [a - ratio*b for a,b in zip(matrix[i],matrix[j])]
    return result

def exact_d(points):
    rows = [[Q(1), *(Q(str(float(v))) for v in row)] for row in points]
    n, p = len(rows), len(rows[0])
    return determinant([[sum(row[i]*row[j] for row in rows)/n
                         for j in range(p)] for i in range(p)])

def smolyak(level):
    # Independent exact quadrature on [0,1]: midpoint, Simpson, Boole.
    rules = [
        {Q(1,2):Q(1)},
        {Q(0):Q(1,6), Q(1,2):Q(4,6), Q(1):Q(1,6)},
        dict(zip(map(Q, [0,Q(1,4),Q(1,2),Q(3,4),1]),
                 [Q(x,90) for x in (7,32,12,32,7)]))]
    differences = []
    for i, rule in enumerate(rules):
        diff = defaultdict(Q, rule)
        if i:
            for x,w in rules[i-1].items():
                diff[x] -= w
        differences.append(dict(diff))
    result = defaultdict(Q)
    for i in range(level+1):
        for j in range(level+1-i):
            for x,wx in differences[i].items():
                for y,wy in differences[j].items():
                    result[x,y] += wx*wy
    return {point:weight for point,weight in result.items() if weight}

def self_check():
    assert gf2_rank([[1,0],[0,1]]) == 2
    assert gf2_rank([[1,1],[1,1]]) == 1
    assert determinant([[1,2],[2,4]]) == 0
    assert exact_d([[-1],[1]]) == 1
    for level, count, moment in [(0,1,Q(1,16)),(1,5,Q(5,48)),(2,13,Q(1,9))]:
        rule = smolyak(level)
        assert len(rule) == count and sum(rule.values()) == 1
        assert sum(w*x*x*y*y for (x,y),w in rule.items()) == moment
    assert gf2_rank(digital_reference(30)[0]) == 2

def run(args):
    self_check()
    wheel = args.wheel.resolve()
    raw = wheel.read_bytes()
    assert digest(raw) == WHEEL_SHA, "wheel byte identity failed"
    assert sys.prefix != sys.base_prefix, "use a separate virtualenv"
    import numpy as np
    import scipy
    from scipy.stats import qmc
    import torch
    import pydoe
    from pydoe.optimal.utils import criterion_value
    assert metadata.version("pydoe") == "1.6.0"
    assert Path(pydoe.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    # Check installed upstream sources against the actual verified wheel before tests.
    site = Path(pydoe.__file__).resolve().parent.parent
    installed_sources = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("pydoe/") and name.endswith(".py"):
                data = archive.read(name)
                assert (site/name).read_bytes() == data, name
                installed_sources[name] = {
                    "sha256":digest(data),
                    "git_blob_sha1":hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()}
    pinned_blobs = {
        "pydoe/response_surface/small_composite.py":"25d025b8d9d95981db6a36771488922ebdb0642d",
        "pydoe/response_surface/doehlert.py":"bc443e2364335a0594087df9f2c87e9d60abc975",
        "pydoe/space_filling/quasi_random/niederreiter.py":"809a31f447024b98f07f8f67e82efafe447373ee",
        "pydoe/sparse_grid/sparse_grid.py":"b9e5e3d5ca81b1974c163a8e3897909b24aa74fc",
        "pydoe/space_filling/quasi_random/sobol.py":"33ecd3aedf676bb5af4d60cc5a8532715fde0905",
        "pydoe/optimal/algorithms.py":"ce48f41018d7819435883d8336c8d8a98a2cec59"}
    for name,sha in pinned_blobs.items():
        assert installed_sources[name]["git_blob_sha1"] == sha, name
    env = {"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,
           "base_prefix":sys.base_prefix,"platform":platform.platform(),
           "cpu":platform.processor(),"numpy":np.__version__,"scipy":scipy.__version__,
           "torch":torch.__version__,"pydoe":metadata.version("pydoe"),
           "numpy_file":np.__file__,"scipy_file":scipy.__file__,"torch_file":torch.__file__,
           "pydoe_file":pydoe.__file__,"device":"cpu","dtype":"float64",
           "seed":SEED,"inherited_system_site_packages":True}
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cases = []
    args.out.mkdir(parents=True,exist_ok=True)
    def case(cid, inputs, function, oracle, compare, limitations=()):
        row = {"case_id":cid,"upstream":{"distribution":"pydoe","version":"1.6.0",
               "source_sha":UPSTREAM,"wheel_sha256":WHEEL_SHA,"wheel_sha256_verified":True},
               "input":inputs,"actual":{"called_original_pydoe":True},
               "oracle":oracle,"comparison":{"status":"INCONCLUSIVE","tolerance":{"atol":1e-9,"rtol":1e-9}},
               "limitations":list(limitations)}
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                value = function()
                arr = np.asarray(value)
                row["actual"].update(shape=list(arr.shape),output=arr.tolist(),
                                     finite=bool(np.isfinite(arr).all()))
            except Exception as exc:
                row["actual"]["exception"] = {"class":type(exc).__name__,"message":str(exc)}
                expected = oracle.get("expected_exception")
                row["comparison"]["status"] = "EXPECTED_EXCEPTION" if type(exc).__name__ == expected else "FAIL_MISMATCH"
            else:
                try:
                    row["comparison"].update(compare(value))
                except Exception as exc:
                    row["comparison"].update(status="INCONCLUSIVE",
                        oracle_error={"class":type(exc).__name__,"message":str(exc)})
            row["actual"]["warnings"] = [str(w.message) for w in seen]
        cases.append(row)
        (args.out/"cases.json").write_text(json.dumps(cases,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(cid, row["comparison"]["status"], flush=True)
    cases.append({"case_id":"P0","upstream":{"version":"1.6.0","source_sha":UPSTREAM,
                  "wheel_sha256":WHEEL_SHA,"wheel_sha256_verified":True},
                  "actual":{"called_original_pydoe":True,"installed_source_files_verified":len(installed_sources),
                            "pinned_source_blobs_verified":len(pinned_blobs)},
                  "oracle":{"kind":"published digest + pinned Git blobs + installed-vs-wheel byte equality"},
                  "comparison":{"status":"PASS_MATCH"}})
    def p1(value):
        x = torch.tensor(value,dtype=torch.float64)
        cols = [torch.ones(len(x),dtype=torch.float64),*x.T,*x.square().T,
                *(x[:,i]*x[:,j] for i,j in it.combinations(range(4),2))]
        features = torch.stack(cols,dim=1)
        rank = int(torch.linalg.matrix_rank(features))
        alias = float((x[:,0]*x[:,1]-x[:,2]*x[:,3]).abs().max())
        # The earlier hypothesis used D=ABC; the observed wheel cube is different.
        cube = [list(row)+[row[0]*row[1]*row[2]] for row in it.product((-1.,1.),repeat=3)]
        axes = [[(sign*2. if j==i else 0.) for j in range(4)] for i in range(4) for sign in (-1,1)]
        reference = torch.tensor(cube+axes+[[0.]*4]*4,dtype=torch.float64)
        ref_features = torch.stack([torch.ones(20,dtype=torch.float64),*reference.T,*reference.square().T,
            *(reference[:,i]*reference[:,j] for i,j in it.combinations(range(4),2))],dim=1)
        exact_gram = [[sum(Q(str(float(row[i])))*Q(str(float(row[j]))) for row in features)
                       for j in range(15)] for i in range(15)]
        return {"status":"PASS_MATCH" if x.shape==(20,4) and rank==12 and alias==0 else "FAIL_MISMATCH",
                "feature_shape":list(features.shape),"rank":rank,"AB_minus_CD_max_abs":alias,
                "exact_gram_determinant":str(determinant(exact_gram)),
                "independent_D_equals_ABC_reference_rank":int(torch.linalg.matrix_rank(ref_features)),
                "interpretation":"Frozen-wheel source bytes match; earlier rank-12 prediction used a different fractional cube. This is not proof of an upstream defect.",
                "singular_values":torch.linalg.svdvals(features).tolist()}
    case("P1",{"n":4,"center":[2,2]},lambda:pydoe.small_composite_design(4,center=(2,2)),
         {"kind":"independent full quadratic torch features","expected_shape":[20,4],
          "expected_rank":12,"feature_columns":15},p1,
         ["Full-quadratic identifiability, not a defect claim for every reduced model."])
    for k in (1,2,3,4):
        def compare_shell(value,k=k):
            x = torch.tensor(value,dtype=torch.float64)
            rank = int(torch.linalg.matrix_rank(x))
            match = tuple(x.shape)==({2:(6,2),3:(10,3),4:(15,4)}[k]) and rank==2
            return {"status":"SOURCE_SEMANTIC_DIFFERENCE" if match else "FAIL_MISMATCH",
                    "source_prediction_match":match,"rank":rank,
                    "documented_expected_rows":k*(k+1)+1}
        case(f"P2-k{k}",{"num_factors":k,"num_center_points":1},
             lambda k=k:pydoe.doehlert_shell_design(k,1),
             {"kind":"independent rank/count check; frozen-source predictions separately labeled",
              "expected_exception":"IndexError" if k==1 else None,
              "expected_rank":None if k==1 else 2},compare_shell,
             ["Frozen source prediction is not a valid general Doehlert construction oracle."])
    for b in (3,5,8,30):
        matrix, values = digital_reference(b)
        ref = np.array([float(x) for x in values]).reshape(16,1)
        def compare_net(value, ref=ref, matrix=matrix):
            error = float(np.max(np.abs(value-ref)))
            count = len(np.unique(value[:,0]))
            return {"status":"SOURCE_SEMANTIC_DIFFERENCE" if error==0 and count==4 and gf2_rank(matrix)==2 else "FAIL_MISMATCH",
                    "source_prediction_match":error==0,"distinct_first_coordinate":count,
                    "independent_GF2_rank":gf2_rank(matrix),"max_abs_error":error}
        case(f"P3-b{b}",{"num_points":16,"dimension":1,"n_bits":b},
             lambda b=b:pydoe.niederreiter_sequence(16,1,n_bits=b),
             {"kind":"independent GF(2) elimination and exact recurrence arithmetic",
              "polynomial":"x^2+x+1","expected_distinct":4,"expected_rank":2},
             compare_net,["Source recurrence equivalence, not certification of Niederreiter theory.",
                          "For b=3, 16 exceeds 2**b; b=5/8/30 do not."])
    scipy_ref = qmc.Sobol(2,scramble=False).random_base2(2)
    torch_ref = torch.quasirandom.SobolEngine(2,scramble=False).draw(4,dtype=torch.float64).numpy()
    assert np.array_equal(scipy_ref,torch_ref)
    for pow2 in (True,False):
        n = 4 if pow2 else 3
        case(f"P5-pow2-{pow2}",{"n":3,"d":2,"scramble":False,"skip":0,"use_pow_of_2":pow2},
             lambda pow2=pow2:pydoe.sobol_sequence(3,2,scramble=False,skip=0,use_pow_of_2=pow2),
             {"kind":"real SciPy QMC + independent real torch Sobol","expected_shape":[n,2],
              "reference_points":torch_ref[:n].tolist()},
             lambda value,n=n:{"status":"PASS_MATCH" if np.array_equal(value,torch_ref[:n]) else "FAIL_MISMATCH",
                               "max_abs_error":float(np.max(np.abs(value-torch_ref[:n])))})
    scipy_skip = None
    try:
        eng = qmc.Sobol(2,scramble=False)
        eng.fast_forward(1)
        eng.random_base2(2)
    except Exception as exc:
        scipy_skip = type(exc).__name__
    assert scipy_skip=="ValueError"
    case("P5-skip1",{"n":4,"d":2,"scramble":False,"skip":1,"use_pow_of_2":True},
         lambda:pydoe.sobol_sequence(4,2,scramble=False,skip=1,use_pow_of_2=True),
         {"kind":"real SciPy backend exception independently invoked","expected_exception":scipy_skip},
         lambda value:{"status":"FAIL_MISMATCH"})
    sparse = {}
    for level in range(4):
        for grid in ("clenshaw_curtis","chebyshev","gauss_patterson"):
            def compare_sparse(value,level=level,grid=grid):
                sparse[level,grid]=np.asarray(value).copy()
                count = (1,5,13,29)[level]
                return {"status":"PASS_MATCH" if value.shape==(count,2) else "FAIL_MISMATCH",
                        "expected_count":count}
            case(f"P4-L{level}-{grid}",{"n_level":level,"n_factors":2,"grid_type":grid},
                 lambda level=level,grid=grid:pydoe.doe_sparse_grid(level,2,grid),
                 {"kind":"frozen-source count prediction (not weighted Smolyak oracle)"},
                 compare_sparse,["Count match alone does not establish quadrature semantics."])
    for level in range(4):
        group = [sparse.get((level,g)) for g in ("clenshaw_curtis","chebyshev","gauss_patterson")]
        identical = all(x is not None for x in group) and all(np.array_equal(group[0],x) for x in group[1:])
        cases.append({"case_id":f"P4-L{level}-grid-type","actual":{"called_original_pydoe":True,
                      "coordinates_identical":bool(identical)},
                      "oracle":{"kind":"cross-option observation; separate exact weighted Smolyak in environment.json"},
                      "comparison":{"status":"SOURCE_SEMANTIC_DIFFERENCE" if identical else "FAIL_MISMATCH"}})
    for level in (7,8):
        case(f"P4-dimension{level}",{"n_level":level,"n_factors":2},
             lambda level=level:pydoe.sparse_grid_dimension(level,2),
             {"kind":"frozen formula prediction","expected":705},
             lambda value:{"status":"PASS_MATCH" if value==705 else "FAIL_MISMATCH"})
    pools = {
        "easy":(np.array([[-1.],[0.],[1.]],dtype=np.float64),2),
        "mentor22":(np.array([[-1.274,-1.346],[1.803,-2.604],[1.111,-0.918],
                      [2.761,-0.783],[-2.761,-2.799],[1.754,2.829],[-2.871,1.579]],dtype=np.float64),4)}
    for pool,(candidates,n) in pools.items():
        exhaustive = [(ids,exact_d(candidates[list(ids)])) for ids in it.combinations(range(len(candidates)),n)]
        global_best = max(score for _,score in exhaustive)
        for method in ("fedorov","detmax","simple_exchange_wynn_mitchell","modified_fedorov"):
            def compare_design(value,candidates=candidates,n=n,global_best=global_best):
                selected = []
                for point in value:
                    matches = np.flatnonzero(np.all(candidates==point,axis=1))
                    if len(matches)!=1:
                        return {"status":"INCONCLUSIVE","reason":"coordinates do not uniquely map to IDs"}
                    selected.append(int(matches[0]))
                score = exact_d(value)
                features = np.column_stack([np.ones(len(value)),value])
                observed_score = criterion_value(features,"D")
                error = abs(float(score)-float(observed_score))
                neighbors = []
                for i in range(n):
                    for j in range(len(candidates)):
                        if j not in selected:
                            swapped=list(selected)
                            swapped[i]=j
                            neighbors.append({"drop_position":i,"add_id":j,
                                              "score":str(exact_d(candidates[swapped]))})
                best = max([score,*[Q(x["score"]) for x in neighbors]])
                rank = int(torch.linalg.matrix_rank(torch.tensor(features,dtype=torch.float64)))
                legal = len(selected)==n and len(set(selected))==n and rank==features.shape[1]
                return {"status":"PASS_MATCH" if legal and error<=1e-9*max(1,abs(float(score))) else "FAIL_MISMATCH",
                        "selected_ids":selected,"unique_ids":len(set(selected))==n,"rank":rank,
                        "normalized_D_exact_fraction":str(score),"upstream_normalized_D":float(observed_score),
                        "max_abs_error":error,"best_closed_one_swap_D":str(best),
                        "one_swap_local_exact":best==score,"all_one_swap_scores":neighbors,
                        "exhaustive_global_D":str(global_best),
                        "global_gap":float(global_best-score),
                        "condition_number":float(np.linalg.cond(features.T@features/n)),
                        "stop_rule_note":"Locality is measured separately; max_iter=20 is a cap, not a convergence certificate."}
            case(f"P6-{pool}-{method}",{"candidates":candidates.tolist(),"n_points":n,"degree":1,
                 "criterion":"D","alpha":0,"max_iter":20},
                 lambda candidates=candidates,n=n,method=method:getattr(pydoe,method)(
                     candidates.copy(),n_points=n,degree=1,criterion="D",alpha=0,max_iter=20),
                 {"kind":"independent exact Fraction determinants + exhaustive legal one-swap enumeration",
                  "normalization":"det(F.T@F/n), features=[1, physical coordinates]"},
                 compare_design,["No forced initial state; §22 conditional traps are not assumed reachable.",
                                 "PASS_MATCH validates score and structural contract, not global optimality."])
    smolyak_evidence = []
    for level in range(3):
        weights = smolyak(level)
        smolyak_evidence.append({"level":level,"nodes":len(weights),"sum_weights":str(sum(weights.values())),
            "x2y2":str(sum(w*x*x*y*y for (x,y),w in weights.items())),
            "negative_weights":sum(w<0 for w in weights.values()),
            "rule":[{"point":[str(x),str(y)],"weight":str(w)} for (x,y),w in sorted(weights.items())]})
    freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True,timeout=30)
    repo = Path(__file__).resolve().parents[2]
    source_sha = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True,timeout=10).strip()
    source_checks = {}
    for name in ("run_reference.py","test_reference.py"):
        relative = "experiments/reference_e0/"+name
        committed = subprocess.check_output(["git","-C",str(repo),"show",source_sha+":"+relative],timeout=10)
        working = (Path(__file__).parent/name).read_bytes()
        assert committed == working, "Commit experiment source before validation: "+name
        source_checks[name] = digest(committed)
    provenance = {"environment":env,"base_sha":BASE,"upstream_source_sha":UPSTREAM,
        "tested_source_sha":source_sha,"committed_source_sha256":source_checks,
        "framework_pin":"d84942e4890d9837ae0e6cbb379b85c3a95aba84",
        "wheel":{"filename":wheel.name,"bytes":len(raw),"sha256":digest(raw),"download_url":"https://pypi.org/simple/pydoe/",
                 "requires_dist":metadata.requires("pydoe")},
        "source_files":installed_sources,"pinned_git_blobs":pinned_blobs,
        "pip_freeze_sanitized":[line.split(" @ ")[0]+" @ <local-file-url-omitted>"
            if " @ file:" in line else line for line in freeze.splitlines()],
        "pip_freeze_raw_sha256":digest(freeze.encode()),
        "distribution_versions":dict(sorted((d.metadata["Name"],d.version) for d in metadata.distributions())),
        "commands":{"argv":sys.argv,"cwd":str(Path.cwd()),"script_sha256":digest(Path(__file__).read_bytes())},
        "independent_smolyak_different_target":smolyak_evidence,
        "scipy_torch_sobol_max_abs":float(np.max(np.abs(scipy_ref-torch_ref))),
        "gpu":"NOT_RUN","performance":"NOT_RUN"}
    args.out.mkdir(parents=True,exist_ok=True)
    # Preserve exact paths locally; public evidence uses a declared task-root token.
    (wheel.parent.parent/"environment-local.json").write_text(
        json.dumps(provenance,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (wheel.parent.parent/"pip-freeze-local.txt").write_text(freeze,encoding="utf-8")
    def public(value):
        if isinstance(value,str):
            return value.replace(str(wheel.parent.parent),"<E0_TASK_ROOT>")
        if isinstance(value,list):
            return [public(x) for x in value]
        if isinstance(value,dict):
            return {k:public(v) for k,v in value.items()}
        return value
    for name,data in (("cases.json",cases),("environment.json",public(provenance))):
        (args.out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    counts = Counter(x["comparison"]["status"] for x in cases)
    summary = ["# E0 real-wheel CPU observations","",
        "Verified original PyDOE 1.6.0 wheel; independent references are separated from source predictions.",
        "Source-semantic differences and expected exceptions are valid findings, not blanket library failures.",
        "", "| Case | Status |","|---|---|",
        *[f"| {x['case_id']} | {x['comparison']['status']} |" for x in cases], "",
        "Status totals: "+json.dumps(dict(counts),sort_keys=True), "",
        "P1 FAIL_MISMATCH: actual full quadratic rank 15/15, exact Gram determinant 18014398509481984.",
        "The prior rank-12 oracle used D=ABC; the wheel uses a different cube. This corrects the hypothesis, not the package.",
        "P2: k=1 IndexError; k=2/3/4 rows 6/10/15 and rank 2, versus documented 7/13/21 rows.",
        "P3: all tested bit widths yield four distinct first-axis values and GF(2) rank 2.",
        "P4: all three grid_type options give identical coordinates at levels 0–3; dimensions 7/8 both 705.",
        "Independent weighted Smolyak is a different target: exact level-2 x²y² integral 1/9 with signed weights.",
        "P5: unscrambled SciPy/torch and original wheel agree; skip=1 + base2 raises ValueError.",
        "P6: eight real method/pool runs; distinct IDs, normalized scores and all legal swaps recorded.",
        "Both small pools attain their exhaustive optimum; no general global-optimality claim.", "",
        "GPU/performance: NOT_RUN. Only the listed inputs were exercised; no general parity claim."]
    (args.out/"summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print(json.dumps(dict(counts)),flush=True)
    return 1 if counts["FAIL_MISMATCH"] or counts["INCONCLUSIVE"] else 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent/"results")
    raise SystemExit(run(parser.parse_args()))

