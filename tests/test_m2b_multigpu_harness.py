"""Experiment-harness tests only; frozen production src is not modified."""
from fractions import Fraction as F
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from m2b_multigpu_exact import cayley, exact_case, product, transpose, strict_rational_geometric
from m2b_multigpu_plan import next_batch, sample_positions, break_even_union


def test_cayley_exact_orthogonality_and_oracles():
    q=cayley(3,180030)
    assert product(transpose(q),q)==[[F(i==j) for j in range(3)] for i in range(3)]
    spectrum=strict_rational_geometric(3,2)
    assert spectrum==[F(1),F(1,10),F(1,100)]
    matrix,contrasts=exact_case(q,spectrum,180031)
    assert matrix==transpose(matrix)
    assert contrasts[0]['quadratic']==contrasts[0]['kappa_c']==1
    assert contrasts[1]['quadratic']==contrasts[1]['kappa_c']==100
    assert 1<=contrasts[2]['kappa_c']<=100


def test_exact_geometric_domain_is_not_silently_approximated():
    for p,e in ((3,1),(10,1),(30,14)):
        with pytest.raises(ValueError,match='UNREPRESENTABLE'):
            strict_rational_geometric(p,e)


def test_sampling_scaling_and_union_break_even():
    assert sample_positions(1)==[0]
    positions=sample_positions(1000000)
    assert len(positions)==10000 and len(set(positions))==10000
    assert positions[0]==0 and positions[-1]==999999
    assert next_batch(100,10,100)==200
    assert next_batch(100,60,100)==116
    rows=[dict(device='L4',p=3,dtype='float64',B=b,cpu_seconds=c,gpu_with_transfer_seconds=g)
          for b,c,g in [(8,2,3),(32,2,1),(16,2,2.5)]]
    result=break_even_union(rows)[0]
    assert result['interval_open_closed']==[16,32]
    assert break_even_union(rows[:1])[0]['status']=='NOT_OBSERVED'
