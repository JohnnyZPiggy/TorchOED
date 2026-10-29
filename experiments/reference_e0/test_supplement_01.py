from copy import deepcopy
import json
from pathlib import Path
import torch
import supplement_01 as supplement

def original_p6():
    rows=json.loads((Path(__file__).parent/"results/cases.json").read_bytes())
    return [row for row in rows if row["case_id"].startswith("P6-")]

def test_pass_predicate_excludes_optimality_diagnostics():
    for row in original_p6():
        assert supplement.p6_legal_score_pass(row)
        changed=deepcopy(row)
        changed["comparison"].update(one_swap_local_exact=False, global_gap=123,
                                     exhaustive_global_D="999999999")
        assert supplement.p6_legal_score_pass(changed)

def test_pass_predicate_rejects_illegal_ids_and_wrong_scores():
    row=deepcopy(original_p6()[0])
    row["comparison"]["selected_ids"]=[0,0]
    assert not supplement.p6_legal_score_pass(row)
    row=deepcopy(original_p6()[0])
    row["comparison"]["upstream_normalized_D"]+=1
    assert not supplement.p6_legal_score_pass(row)

def test_independent_designs_have_different_ranks():
    # A mathematical fixture, not an extra upstream invocation.
    negative=supplement.negative_control()
    positive=[row.copy() for row in negative]
    for row in positive[:8]:
        row[3]=row[0]*row[1]
    full=supplement.features(positive)
    deficient=supplement.features(negative)
    assert tuple(full.shape)==(20,15)
    assert int(torch.linalg.matrix_rank(full,atol=1e-9,rtol=1e-9))==15
    assert supplement.exact_gram_det(full)==supplement.EXPECTED_DETERMINANT
    assert int(torch.linalg.matrix_rank(deficient,atol=1e-9,rtol=1e-9))==12
    assert supplement.exact_gram_det(deficient)==0

def test_original_archive_and_p4_source_ids():
    assert supplement.archive_integrity()["original_manifest_entries_verified"]==10
    rows=json.loads((Path(__file__).parent/"results/cases.json").read_bytes())
    by_id={row["case_id"]:row for row in rows}
    assert len(rows)==39
    for level in range(4):
        ids=[f"P4-L{level}-{g}" for g in ("clenshaw_curtis","chebyshev","gauss_patterson")]
        sources=[by_id[cid]["actual"] for cid in ids]
        assert all(row["called_original_pydoe"] for row in sources)
        assert sources[0]["output"]==sources[1]["output"]==sources[2]["output"]

