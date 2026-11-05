"""Numerical falsification checks for the sharpened analytic certificates.
Independent objectives, spectral extremizers, and a same-sample stochastic recursion.
"""
import json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from bures.core import spd_sqrt,ot_map

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
rng=np.random.default_rng(20261102)
def spd(d):
    x=torch.tensor(rng.normal(size=(d,d)));return x@x.T/d+torch.eye(d)*.8

def basis(d):
    out=[]
    for i in range(d):
        x=torch.zeros(d,d);x[i,i]=1;out.append(x)
    for i in range(d):
        for j in range(i+1,d):
            x=torch.zeros(d,d);x[i,j]=x[j,i]=1/math.sqrt(2);out.append(x)
    return torch.stack(out)

def syl(A,E):
    w,q=torch.linalg.eigh(A);return q@((q.T@E@q)/(w[:,None]+w[None,:]))@q.T

def hessian_checks():
    A=spd(3);Ss=[spd(3) for _ in range(5)];bas=basis(3);root=[spd_sqrt(S) for S in Ss]
    def f(x):
        X=A+torch.einsum('i,ijk->jk',x,bas)
        return torch.stack([.5*(X.trace()+S.trace())-spd_sqrt(R@X@R).trace() for R,S in zip(root,Ss)]).mean()
    H=torch.autograd.functional.hessian(f,torch.zeros(len(bas)))
    aa=float(torch.linalg.eigvalsh(A).min());bb=float(torch.linalg.eigvalsh(A).max())
    ass=min(float(torch.linalg.eigvalsh(S).min()) for S in Ss);bss=max(float(torch.linalg.eigvalsh(S).max()) for S in Ss)
    m=math.sqrt(ass)/(4*bb**1.5);L=math.sqrt(bss)/(4*aa**1.5);eig=torch.linalg.eigvalsh(H)
    assert float(eig.min())>=m-1e-10 and float(eig.max())<=L+1e-10
    # Independent one-dimensional differentiation attains both uniform constants.
    err=0.
    for a,s,c in [(bb,ass,m),(aa,bss,L)]:
        h=torch.autograd.functional.hessian(lambda x:.5*(x+s)-torch.sqrt(x*s),torch.tensor(a))
        err=max(err,abs(float(h)-c))
    # Noncommuting targets constructed to have an exactly known barycenter C.
    C=spd(3);ks=[]
    for _ in range(2):
        Z=torch.tensor(rng.normal(size=(3,3)));Z=(Z+Z.T)/2;ks.append(.28*Z/torch.linalg.matrix_norm(Z,ord=2))
    Ts=[torch.eye(3)+e*ks[0]+z*ks[1] for e in [-1,1] for z in [-1,1]]
    S2=[T@C@T for T in Ts];M=torch.stack([torch.einsum('kij,ij->k',bas,.5*syl(C,E)) for E in bas],dim=1)
    def pop(x):
        X=C+torch.einsum('i,ijk->jk',x,bas)
        return torch.stack([.5*(X.trace()+S.trace())-spd_sqrt(spd_sqrt(S)@X@spd_sqrt(S)).trace() for S in S2]).mean()
    Hc=torch.autograd.functional.hessian(pop,torch.zeros(len(bas)))
    w,U=torch.linalg.eigh(M);Mi=U@torch.diag(w.rsqrt())@U.T;re=torch.linalg.eigvalsh(Mi@Hc@Mi)
    ell=min(float(torch.linalg.eigvalsh(T).min()) for T in Ts);u=max(float(torch.linalg.eigvalsh(T).max()) for T in Ts);lower=ell*u/(ell+u-1)
    assert float(re.max())<=1+1e-9 and float(re.min())>=lower-1e-9
    # Equality family from the theorem, checked by a Sylvester Hessian computation.
    ell=.3;u=1.7;c=1.6;Ceq=c*torch.eye(2);E=torch.tensor([[0.,1.],[1.,0.]])/math.sqrt(2)
    val=0.
    for T in [torch.diag(torch.tensor([ell,u])),torch.diag(torch.tensor([u,ell]))]:
        R=spd_sqrt(T);Hs=.5*R@syl(R@Ceq@R,R@E@R)@R
        val+=float((E*Hs).sum())/2
    attained=val/float((E*(.5*syl(Ceq,E))).sum());err2=abs(attained-ell*u)
    assert err<1e-12 and err2<1e-12
    S_op=torch.stack([torch.einsum('kij,ij->k',bas,C@E+E@C) for E in bas],dim=1)
    J=(2*S_op@Hc).numpy();a=torch.einsum('kii->k',bas).numpy();radial_error=float(np.linalg.norm(a@J-a))
    zs=[]
    for T in Ts:
        Z=(torch.eye(3)-T)@C+C@(torch.eye(3)-T);zs.append(torch.einsum('kij,ij->k',bas,Z).numpy())
    zs=np.array(zs);Q=zs.T@zs/len(zs);nc=len(a)
    covariance=np.linalg.solve(np.kron(np.eye(nc),J)+np.kron(J,np.eye(nc)),Q.reshape(-1,order='F')).reshape(nc,nc,order='F')
    trace_variance_error=abs(float(a@covariance@a)-2*np.var([float(torch.trace(T@C)) for T in Ts]))
    assert radial_error<1e-10 and trace_variance_error<1e-10
    return dict(scalar_Hessian_attainment_error=err,Riemannian_lower_attainment_error=err2,noncommuting_Riemannian_min=float(re.min()),noncommuting_Riemannian_max=float(re.max()),operator_Jensen_lower=lower,radial_left_eigenvector_error=radial_error,trace_variance_identity_error=trace_variance_error)

def spectral_checks():
    max_violation=0.;max_eq=0.
    for p in range(1,9):
        for h in [1.0001,1.4,4.,20.,1000.]:
            k=np.arange(p+1);log_sp=np.max(np.log1p((h-1)*k/p)-k/p*math.log(h))
            continuous=math.log(h-1)+math.log(h)/(h-1)-1-math.log(math.log(h))
            assert log_sp<=continuous+1e-10 and continuous<=math.log(h)**2/8+1e-10
            xs=rng.uniform(1,h,size=(700,p));observed=np.log(xs.mean(1))-np.log(xs).mean(1)
            max_violation=max(max_violation,float(np.max(observed)-log_sp))
            kk=int(np.argmax(np.log1p((h-1)*k/p)-k/p*math.log(h)));x=np.array([h]*kk+[1.]*(p-kk))
            max_eq=max(max_eq,abs(float(np.log(x.mean())-np.log(x).mean())-log_sp))
    assert max_violation<1e-10 and max_eq<1e-12
    # Schur complement coefficient is attained by eliminating the momentum energy.
    q=.08;r=.43;s=1-q-r*r;n=1.7;M=r*n/s;coef=(1-q)/s
    err=abs((-s*M*M+2*r*n*M+n*n)-coef*n*n)
    assert err<1e-12
    angle_err=0.
    for kh in [1.0001,1.4,4.,20.,1000.]:
        kp=math.sqrt(kh);gam=2*math.sqrt(kp)/(1+kp)
        angle_err=max(angle_err,abs(math.log((1+kh)/(2*math.sqrt(kh)))-math.log(2/gam**2-1)))
    assert angle_err<1e-12
    return dict(Specht_random_violation=max(0.,max_violation),Specht_endpoint_attainment_error=max_eq,Schur_coefficient_attainment_error=err,angle_complexity_identity_error=angle_err)

def kl(u,v):
    if v==1:return 0. if u==1 else math.inf
    if v==0:return 0. if u==0 else math.inf
    return (0. if u==0 else u*math.log(u/v))+(0. if u==1 else (1-u)*math.log((1-u)/(1-v)))
def inv(u,c):
    lo=u;hi=1.
    for _ in range(90):
        mid=(lo+hi)/2
        if kl(u,mid)<=c:lo=mid
        else:hi=mid
    return lo

def pac_checks():
    max_err=0.;normalizer={}
    for N in [1,2,8,100,1000]:
        terms=[]
        for k in range(N+1):
            b=math.lgamma(N+1)-math.lgamma(k+1)-math.lgamma(N-k+1)
            a=b+(k*math.log(k/N) if k else 0)+((N-k)*math.log1p(-k/N) if k<N else 0)
            terms.append(a)
        c=sum(math.exp(t) for t in terms);normalizer[N]=c
        assert c<=N+1+1e-9
        if N>=8:assert c<=2*math.sqrt(N)+1e-9
        for mu in [.03,.37,.8]:
            m=sum(math.exp(math.lgamma(N+1)-math.lgamma(k+1)-math.lgamma(N-k+1)+k*math.log(mu)+(N-k)*math.log1p(-mu)+N*kl(k/N,mu)) for k in range(N+1))
            max_err=max(max_err,abs(m-c)/c)
    for u in [0,.01,.2,.95]:
        v=inv(u,.006);assert v<=min(1,u+math.sqrt(.003))+1e-12 and abs(kl(u,v)-.006)<1e-10
    assert max_err<1e-10
    # Compute the Gaussian Gibbs identity via two independent KL/objective formulas.
    p=5;C=spd(p);sigma2=1.4;lam=6.;S=spd(p);B=torch.eye(p)/sigma2+2*lam*C
    Phi=torch.trace(C@S)+(torch.trace(S)/sigma2-p+p*math.log(sigma2)-torch.logdet(S))/(2*lam)
    KL=.5*(torch.trace(B@S)-p-torch.logdet(B)-torch.logdet(S))
    err=abs(float(lam*Phi-KL-.5*torch.logdet(torch.eye(p)+2*lam*sigma2*C)));assert err<1e-10
    return dict(Bernoulli_extremal_moment_relative_error=max_err,c100=normalizer[100],c1000=normalizer[1000],kl_inverse_zero_empirical=inv(0,.006),Pinsker_zero_empirical=math.sqrt(.003),Gaussian_Gibbs_identity_error=err)

def stochastic_checks():
    n=6000;p=3;T=512;beta=.95;delta=1-beta;lo=.7;hi=1.3;L=2.;sigma2=p*.25**2
    kernel=np.zeros((T,T))
    for t in range(1,T+1):
        for k in range(2,t+1):kernel[t-1,k-1]=beta**(t-k+1)*(1-beta**(k-1))/(1-beta**t)
    kexact=float(np.linalg.svd(kernel,compute_uv=False)[0]);assert kexact<=beta/delta+1e-10
    V=sum(delta/(1+beta)*(1+beta**t)/(1-beta**t) for t in range(1,T+1));harm=sum(1/t for t in range(1,T+1));assert V<=(T*delta+2*beta*harm)/(1+beta)
    out={}
    for name,eta in [('old_step',.5*delta*lo/(math.sqrt(96)*L*hi**2)),('larger_certified_step',.5*delta/(L*hi))]:
        local=np.random.default_rng(20261102);theta=np.tile(np.array([.8,-.6,.3]),(n,1));m=np.zeros_like(theta);mh=m.copy();mn=m.copy()
        HH=HB=MB=EB=NN=DD=DH=0.;previous_h=None;max_path_storage=0.
        q=L*eta*hi;r=L*eta*hi*kexact;s=1-q-r*r;assert s>0
        for step in range(1,T+1):
            h=theta*np.array([.5,1.,2.]);xi=local.normal(0,.25,(n,p));g=h+xi;Bdiag=1+.3*np.tanh(g)
            mh=beta*mh+delta*h;mn=beta*mn+delta*xi;m=beta*m+delta*g;hat=m/(1-beta**step);noise=mn/(1-beta**step);D=mh/(1-beta**step)-h;e=hat-h
            HH+=np.mean(np.sum(h*h,1));HB+=np.mean(np.sum(Bdiag*h*h,1));MB+=np.mean(np.sum(Bdiag*hat*hat,1));EB+=np.mean(np.sum(Bdiag*e*e,1));NN+=np.mean(np.sum(noise*noise,1));DD+=np.mean(np.sum(D*D,1))
            if previous_h is not None:DH+=np.mean(np.sum((h-previous_h)**2,1))
            previous_h=h.copy();f0=.5*np.sum(h*theta,1);theta-=eta*Bdiag*hat;f1=.5*np.sum(theta*theta*np.array([.5,1.,2.]),1)
            storage=(f1-f0)+eta/2*np.sum(Bdiag*(h*h+(1-q)*hat*hat-e*e),1)
            max_path_storage=max(max_path_storage,float(np.max(storage)))
        gap=.5*np.sum(np.array([.5,1.,2.])*np.array([.8,-.6,.3])**2)
        new=2*gap/(eta*lo*T)+(hi/lo)*(1-q)/s*sigma2*V/T
        old=8*gap/(eta*lo*T)+13*(hi/lo)**2*sigma2*(delta/(1+beta)+2*beta/(T*delta*(1+beta)))
        assert HH/T<=new and DD<=kexact*kexact*DH+1e-9 and max_path_storage<1e-12
        assert abs(NN/(sigma2*V)-1)<.04
        out[name]=dict(step=eta,q=q,r=r,observed_average_gradient_square=HH/T,new_certificate=new,old_coarse_expression=old,old_expression_is_certified=(name=='old_step'),pathwise_storage_violation=max_path_storage,finite_kernel_drift_ratio=DD/(kexact*kexact*DH),noise_energy_ratio=NN/(sigma2*V))
    return dict(kernel_exact_gain=kexact,infinite_gain_bound=beta/delta,exact_variance_sum=V,harmonic_variance_upper=(T*delta+2*beta*harm)/(1+beta),runs=out)

if __name__=='__main__':
    result={'hessian':hessian_checks(),'spectral':spectral_checks(),'pac':pac_checks(),'same_batch':stochastic_checks()}
    (ROOT/'experiments/tests/polished_theory_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
