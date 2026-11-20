import pytest
import torch
from torchoed import design_diagnostics
from torchoed.information import _coefficient_power


def fixture():
    return torch.tensor([[1,a,b,x] for a in (-1,-1,1,1) for b in (-1,1) for x in (-1,1)], dtype=torch.float64)


def test_oracle():
    F = fixture(); ids = torch.arange(16); delta = F.new_tensor([.2,.9,.7,.5])
    for ew, es in ((2,.5),(3/7,5/3)):
        d = design_diagnostics(F, ids, whole_plots=[4]*4, subplots=[2]*8, eta=ew, eta_sub=es,
                               effect=delta, groups=[[0,1],[2],[3]])
        lam = delta.square()*F.new_tensor([16/(1+2*es+4*ew)]*2+[16/(1+2*es),16])
        power = torch.cat([_coefficient_power(v[None], df, .05) for v,df in zip(lam,(2,2,3,7))])
        for key, value in dict(power=power, df=ids.new_tensor([2,2,3,7]), correlation=torch.eye(3,dtype=F.dtype),
                               vif=F.new_ones(3), spv=F.new_full((16,),4+6*es+8*ew)).items():
            torch.testing.assert_close(d[key],value,rtol=0,atol=1e-12)
        covariance = torch.diag(F.new_tensor([(1+2*es+4*ew)/16]*2+[(1+2*es)/16,1/16]))
        group_lam = (delta[:2].square()/covariance.diagonal()[:2]).sum().reshape(1)
        torch.testing.assert_close(d['group_power'][0],_coefficient_power(group_lam,2,.05,2)[0],rtol=0,atol=1e-12)
        assert d['group_df'].tolist() == [2,3,7]
    with pytest.raises(ValueError,match='mixed'):
        design_diagnostics(F,ids,whole_plots=[4]*4,subplots=[2]*8,effect=.5,groups=[[1,2]])


def test_dense_unbalanced():
    gen = torch.Generator().manual_seed(479)
    for sizes in ([1,2,3,2,1,3,1,2],[2,3,1,2,3,1,2,3],[3,1,2,1,3,2,1,3]):
        wp = [sum(sizes[i:i+2]) for i in range(0,8,2)]; n=sum(wp)
        w=torch.arange(4).repeat_interleave(torch.tensor(wp)); s=torch.arange(8).repeat_interleave(torch.tensor(sizes))
        F=torch.randn(n,4,dtype=torch.float64,generator=gen); F[:,0]=1; F[:,1]=w.double(); F[:,2]=s.double()
        ids=torch.arange(n); R=torch.randn(7,4,dtype=F.dtype,generator=gen)
        Zw=torch.nn.functional.one_hot(w).double(); Zs=torch.nn.functional.one_hot(s).double()
        V=torch.eye(n,dtype=F.dtype)+.7*Zw@Zw.T+1.3*Zs@Zs.T
        W=torch.linalg.solve(V,torch.eye(n,dtype=F.dtype)); u=W.sum(1); P=W-torch.outer(u,u)/u.sum()
        C=F[:,1:].T@P@F[:,1:]; inv=torch.linalg.solve(F.T@W@F,torch.eye(4,dtype=F.dtype))
        whole=[all(v[:,j].unique().numel()==1 for v in F.split(wp)) for j in range(4)]
        sub=[not whole[j] and all(v[:,j].unique().numel()==1 for v in F.split(sizes)) for j in range(4)]
        df=[4-sum(whole) if whole[j] else 8-4-sum(sub) if sub[j] else n-8-(4-sum(whole)-sum(sub)) for j in range(4)]
        d=design_diagnostics(F,ids,region=R,whole_plots=wp,subplots=sizes,eta=.7,eta_sub=1.3,effect=.5)
        for key,value in dict(correlation=C/torch.outer(C.diagonal().sqrt(),C.diagonal().sqrt()),
                vif=C.diagonal()*torch.linalg.solve(C,torch.eye(3,dtype=F.dtype)).diagonal(),
                spv=n*(R@inv*R).sum(1),df=ids.new_tensor(df)).items():
            torch.testing.assert_close(d[key],value,rtol=1e-10,atol=1e-10)


def test_reductions():
    F=fixture(); ids=torch.arange(16); delta=F.new_tensor([.2,.9,.7,.5])
    for sub,eta,scale in (([1]*16,2/1.5,1.5),([4]*4,2.5,1)):
        d=design_diagnostics(F,ids,whole_plots=[4]*4,subplots=sub,eta=2,eta_sub=.5,effect=delta,groups=[[0,1],[2,3]])
        r=design_diagnostics(F,ids,whole_plots=[4]*4,eta=eta,effect=delta/scale**.5,groups=[[0,1],[2,3]])
        for key in d:
            if key=='columns': assert d[key]==r[key]
            else: torch.testing.assert_close(d[key],r[key]*scale if key in ('spv','fds') else r[key],rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='align'):
        design_diagnostics(F,ids,whole_plots=[4]*4,subplots=[3,3,3,3,4])
