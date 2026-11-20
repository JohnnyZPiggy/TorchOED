import pytest
import torch
from torchoed import design_diagnostics
from torchoed.information import _coefficient_power


def test_oracle():
    F = torch.tensor([[1, a, b, (-1)**(i+j)] for i, a in enumerate((-1,-1,1,1))
                      for j, b in enumerate((-1,-1,1,1))], dtype=torch.float64)
    ids = torch.arange(16); delta = F.new_tensor([.2,.9,.7,.5])
    for er, ec in ((1,1),(3/7,5/3)):
        d = design_diagnostics(F, ids, strip=(4,4), eta_row=er, eta_col=ec, effect=delta, groups=[[0,1],[2],[3]])
        lam = delta.square()*F.new_tensor([16/(1+4*er+4*ec),16/(1+4*er),16/(1+4*ec),16])
        power = torch.cat([_coefficient_power(v[None], df, .05) for v,df in zip(lam,(2,2,2,8))])
        for key, value in dict(power=power, df=ids.new_tensor([2,2,2,8]), correlation=torch.eye(3,dtype=F.dtype),
                               vif=F.new_ones(3), spv=F.new_full((16,),4+8*er+8*ec)).items():
            torch.testing.assert_close(d[key], value, rtol=0, atol=1e-12)
        torch.testing.assert_close(d['group_power'][0], _coefficient_power(lam[:2].sum().reshape(1),2,.05,2)[0],rtol=0,atol=1e-12)
        assert d['group_df'].tolist() == [2,2,8]
    with pytest.raises(ValueError, match='mixed'):
        design_diagnostics(F,ids,strip=(4,4),effect=.5,groups=[[1,2]])


def test_dense():
    gen = torch.Generator().manual_seed(483)
    for r,c in ((3,3),(3,4),(4,4)):
        n=r*c; ids=torch.arange(n)
        F=torch.randint(-2,3,(2*n,3),generator=gen).double(); F[:,0]=1
        ids=torch.randperm(2*n,generator=gen)[:n]; S=F[ids]; R=F[:7]
        row=torch.arange(r).repeat_interleave(c); col=torch.arange(c).repeat(r)
        Zr=torch.nn.functional.one_hot(row).double(); Zc=torch.nn.functional.one_hot(col).double()
        V=torch.eye(n,dtype=F.dtype)+.7*Zr@Zr.T+1.3*Zc@Zc.T
        W=torch.linalg.solve(V,torch.eye(n,dtype=F.dtype)); u=W.sum(1); P=W-torch.outer(u,u)/u.sum()
        C=S[:,1:].T@P@S[:,1:]; inv=torch.linalg.solve(S.T@W@S,torch.eye(3,dtype=F.dtype))
        grid=S.reshape(r,c,3)
        rows=[all(grid[i,:,j].unique().numel()==1 for i in range(r)) for j in range(3)]
        cols=[not rows[j] and all(grid[:,i,j].unique().numel()==1 for i in range(c)) for j in range(3)]
        df=[r-sum(rows) if rows[j] else c-1-sum(cols) if cols[j] else (r-1)*(c-1)-(3-sum(rows)-sum(cols)) for j in range(3)]
        d=design_diagnostics(F,ids,region=R,strip=(r,c),eta_row=.7,eta_col=1.3,effect=.5)
        for key,value in dict(correlation=C/torch.outer(C.diagonal().sqrt(),C.diagonal().sqrt()),
                vif=C.diagonal()*torch.linalg.solve(C,torch.eye(2,dtype=F.dtype)).diagonal(),
                spv=n*(R@inv*R).sum(1),df=ids.new_tensor(df)).items():
            torch.testing.assert_close(d[key],value,rtol=1e-10,atol=1e-10)


def test_errors():
    F=torch.tensor([[1,a,b] for a in (-1,1) for b in (-1,1)],dtype=torch.float64); ids=torch.arange(4)
    with pytest.raises(ValueError, match='cannot be combined'):
        design_diagnostics(F,ids,strip=(2,2),whole_plots=[2,2])
    with pytest.raises(ValueError, match='strip requires'):
        design_diagnostics(F,ids,strip=(2,3))
