from fractions import Fraction as Q
import run_reference as ref

def test_exact_d_and_degenerate_design():
    assert ref.exact_d([[-1],[1]]) == 1
    assert ref.exact_d([[1],[1]]) == 0
    assert ref.exact_d([[-1],[0],[1]]) == Q(2,3)

def test_gf2_rank_is_not_real_rank():
    assert ref.gf2_rank([[1,1,0],[1,0,1],[0,1,1]]) == 2
    for bits in (3,5,8,30):
        matrix, values = ref.digital_reference(bits)
        assert ref.gf2_rank(matrix) == 2
        assert len(set(values)) == 4

def test_exact_smolyak_integration_and_signed_weights():
    weights = ref.smolyak(2)
    assert len(weights) == 13
    assert sum(weights.values()) == 1
    assert sum(w*x*x*y*y for (x,y),w in weights.items()) == Q(1,9)
    assert any(w < 0 for w in weights.values())

