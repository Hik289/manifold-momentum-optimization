"""Independent derivative, variational, and stationary-law checks for Appendix C."""
import json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from bures.core import spd_sqrt,spd_inv_sqrt,ot_map
torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(914)
def spd(d):
    x=torch.randn(d,d);return x@x.T/d+torch.eye(d)
def basis(d):
    out=[]
    for i in range(d):
        z=torch.zeros(d,d);z[i,i]=1;out.append(z)
    for i in range(d):
        for j in range(i+1,d):
            z=torch.zeros(d,d);z[i,j]=z[j,i]=1/math.sqrt(2);out.append(z)
    return torch.stack(out)
def syl(a,e):
    w,q=torch.linalg.eigh(a);return q@((q.T@e@q)/(w[:,None]+w[None,:]))@q.T
def relative(x,y):return float((x-y).norm()/(1+y.norm()))
def phi_cov(S,C,sigma,lam):
    p=len(S);return torch.trace(C@S)+(torch.trace(S)/sigma-p+p*math.log(sigma)-torch.logdet(S))/(2*lam)

def deterministic_checks():
    A=spd(3);targets=[spd(3) for _ in range(4)];E=basis(3);zero=torch.zeros(len(E))
    roots=[spd_sqrt(S) for S in targets]
    def F(x):
        B=A+torch.einsum('i,ijk->jk',x,E)
        return torch.stack([.5*(B.trace()+S.trace())-spd_sqrt(Sr@B@Sr).trace() for S,Sr in zip(targets,roots)]).mean()
    grad=torch.autograd.functional.jacobian(F,zero);hess=torch.autograd.functional.hessian(F,zero)
    Tg=[ot_map(A,S) for S in targets];expected_grad=torch.einsum('kij,ij->k',E,.5*(torch.eye(3)-torch.stack(Tg).mean(0)))
    def H(e):
        parts=[]
        for T in Tg:
            Th=spd_sqrt(T);B=Th@A@Th;parts.append(.5*Th@syl(B,Th@e@Th)@Th)
        return torch.stack(parts).mean(0)
    expected_hess=torch.stack([torch.einsum('kij,ij->k',E,H(e)) for e in E],dim=1)
    grad_error=relative(grad,expected_grad);hess_error=relative(hess,expected_hess)
    assert grad_error<1e-9 and hess_error<1e-8,(grad_error,hess_error)
    aa=min(float(torch.linalg.eigvalsh(x).min()) for x in [A]+targets);bb=max(float(torch.linalg.eigvalsh(x).max()) for x in [A]+targets)
    eig=torch.linalg.eigvalsh(hess);assert eig.min()>=aa/(4*bb**2)-1e-10 and eig.max()<=bb/(4*aa**2)+1e-10
    # Differentiate the normalized operator map directly, without differentiating the formula.
    P=spd(5);H0=torch.randn(5,5);H0=(H0+H0.T)/2;K=torch.randn(5,5);K=(K+K.T)/2;m=torch.randn(5)
    def U(x):
        y=(P+x[0]*H0+x[1]*K)@m;return m.norm()*y/y.norm()
    jac=torch.autograd.functional.jacobian(U,torch.zeros(2));mixed=torch.stack([torch.autograd.functional.hessian(lambda x:U(x)[i],torch.zeros(2))[0,1] for i in range(5)])
    y=P@m;r=m.norm();q=y.norm();z=H0@m;w=K@m
    d1=r/q*(torch.eye(5)-torch.outer(y,y)/q**2)@z
    d2=-r/q**3*(z*torch.dot(y,w)+w*torch.dot(y,z)+y*torch.dot(z,w))+3*r/q**5*y*torch.dot(y,z)*torch.dot(y,w)
    d1_error=relative(jac[:,0],d1);d2_error=relative(mixed,d2);assert d1_error<1e-12 and d2_error<1e-12
    # Gaussian PAC objective: optimize covariance and compare dense versus factored penalties.
    C=spd(6);sigma=1.4;lam=7.;B=torch.eye(6)/sigma+2*lam*C;L=spd(3);R=spd(2);HH=torch.kron(R,L);Hi=torch.linalg.inv(HH)
    us=6/torch.trace(B@Hi);free=torch.linalg.inv(B);dense_gap=phi_cov(us*Hi,C,sigma,lam)-phi_cov(free,C,sigma,lam)
    Bh=spd_sqrt(B);KK=Bh@Hi@Bh;gap=(6*torch.log(torch.trace(KK)/6)-torch.logdet(KK))/(2*lam)
    gap_error=abs(float(dense_gap-gap));assert gap_error<1e-10 and gap>=-1e-10
    def shape(H):return .5*(torch.logdet(H)+len(H)*torch.log(torch.linalg.inv(H).trace()/len(H)))
    kron_shape_error=abs(float(shape(HH)-3*shape(R)-2*shape(L)));assert kron_shape_error<1e-12
    # A full off-block curvature matrix against blockwise temperature optimization.
    Cb=spd(10);Bb=torch.eye(10)/sigma+2*lam*Cb;pairs=[(spd(2),spd(3)),(spd(2),spd(2))];blocks=[torch.kron(R,L) for R,L in pairs];SS=[];block_ld=0.;shape_gap=0.;off=0;factored_v=0.;factored_c=0.
    for H,(Rj,Lj) in zip(blocks,pairs):
        p=len(H);Bj=Bb[off:off+p,off:off+p];Hinv=torch.linalg.inv(H);uj=p/torch.trace(Bj@Hinv);SS.append(uj*Hinv);Kj=spd_sqrt(Bj)@Hinv@spd_sqrt(Bj)
        block_ld+=torch.logdet(Bj);shape_gap+=(p*torch.log(torch.trace(Kj)/p)-torch.logdet(Kj))/(2*lam);off+=p
        Ri=torch.linalg.inv(Rj);Li=torch.linalg.inv(Lj);factored_v+=uj**2*torch.trace(Ri@Ri)*torch.trace(Li@Li);factored_c=max(factored_c,float(uj/(torch.linalg.eigvalsh(Rj).min()*torch.linalg.eigvalsh(Lj).min())))
    block_gap=(block_ld-torch.logdet(Bb))/(2*lam);dense=phi_cov(torch.block_diag(*SS),Cb,sigma,lam)-phi_cov(torch.linalg.inv(Bb),Cb,sigma,lam)
    block_error=abs(float(dense-block_gap-shape_gap));assert block_error<1e-10 and block_gap>=-1e-10 and shape_gap>=-1e-10
    Sd=torch.block_diag(*SS);tail_moment_error=abs(float(torch.trace(Sd@Sd)-factored_v));tail_norm_error=abs(float(torch.linalg.eigvalsh(Sd).max())-factored_c);assert tail_moment_error<1e-12 and tail_norm_error<1e-12
    return dict(population_gradient_error=grad_error,population_Hessian_error=hess_error,normalized_operator_derivative_error=d1_error,normalized_operator_second_derivative_error=d2_error,curvature_alignment_gap_error=gap_error,Kronecker_shape_additivity_error=kron_shape_error,block_plus_shape_decomposition_error=block_error,Gaussian_tail_factored_second_moment_error=tail_moment_error,Gaussian_tail_factored_operator_norm_error=tail_norm_error)

def orbit_calculus():
    t=1.;u=.16;v=.11;r2=u*u+v*v;D=torch.diag(torch.tensor([1.,-1.]));O=torch.tensor([[0.,1.],[1.,0.]]);I=torch.eye(2);bas=torch.stack([I,D,O])/math.sqrt(2)
    stats=[(t*I+ep*u*D+ze*v*O)@(t*I+ep*u*D+ze*v*O) for ep in [-1,1] for ze in [-1,1]]
    def T2(A,S):
        k=torch.sqrt(torch.linalg.det(A)*torch.linalg.det(S));return (S+k*torch.linalg.inv(A))/torch.sqrt(torch.trace(A@S)+2*k)
    def gg(x):
        A=t*t*I+torch.einsum('i,ijk->jk',x,bas);zs=[]
        for S in stats:
            K=I-T2(A,S);zs.append(K@A+A@K)
        return torch.einsum('kij,ij->k',bas,torch.stack(zs).mean(0))
    J=torch.autograd.functional.jacobian(gg,torch.zeros(3));Bs=torch.stack([torch.autograd.functional.hessian(lambda x:gg(x)[i],torch.zeros(3)) for i in range(3)])
    expected_J=torch.diag(torch.tensor([1.,1-v*v/t**2,1-u*u/t**2]));assert relative(J,expected_J)<1e-12
    zs=[];Ws=[]
    for S in stats:
        T=T2(t*t*I,S);Z=(I-T)*(t*t)*2;zs.append(torch.einsum('kij,ij->k',bas,Z));Ws.append((T-I)@(t*t*I)@(T-I))
    z=torch.stack(zs);Q=z.T@z/len(z);cov=torch.diag(torch.tensor([0.,4*t*t*u*u/(1-v*v/t**2),4*t*t*v*v/(1-u*u/t**2)]))
    lyap_error=relative(J@cov+cov@J.T,Q);assert lyap_error<1e-12
    VV=torch.einsum('kij,ij->k',bas,torch.stack(Ws).mean(0));bias=torch.linalg.solve(J,VV-.5*torch.einsum('ijk,jk->i',Bs,cov));expected_bias=torch.tensor([r2/math.sqrt(2),0.,0.])
    bias_error=relative(bias,expected_bias);assert bias_error<1e-12
    comm=float((stats[0]@stats[1]-stats[1]@stats[0]).norm());assert comm>0
    return dict(Jacobian_error=relative(J,expected_J),Lyapunov_error=lyap_error,curvature_bias_error=bias_error,noncommuting_commutator_norm=comm),cov.numpy()

def simulate_orbit(alpha,cov,N=6000):
    rng=np.random.default_rng(1203);t=1.;u=.16;v=.11;r2=u*u+v*v;beta=1-alpha;rho=.04;a=np.ones(N);b=np.zeros(N);c=np.ones(N);clips=0;nsteps=math.ceil(8/alpha)
    for _ in range(nsteps):
        ep=2*rng.integers(0,2,N)-1;ze=2*rng.integers(0,2,N)-1;sa=1+r2+2*u*ep;sc=1+r2-2*u*ep;sb=2*v*ze
        k=np.sqrt(np.maximum(a*c-b*b,1e-20))*(1-r2);den=np.sqrt(a*sa+2*b*sb+c*sc+2*k)
        na=beta*beta*a+alpha*alpha*sa+alpha*beta*2*(a*sa+b*sb+k)/den
        nb=beta*beta*b+alpha*alpha*sb+alpha*beta*(b*(sa+sc)+sb*(a+c))/den
        nc=beta*beta*c+alpha*alpha*sc+alpha*beta*2*(c*sc+b*sb+k)/den
        mean=(na+nc)/2;dev=(na-nc)/2;radius=np.sqrt(dev*dev+nb*nb);lp=mean+radius;lm=mean-radius;mask=(lp>1+rho)|(lm<1-rho)
        if np.any(mask):
            clips+=int(mask.sum());pp=np.clip(lp,1-rho,1+rho);mm=np.clip(lm,1-rho,1+rho);newmean=(pp+mm)/2;scale=np.divide((pp-mm)/2,radius,out=np.ones_like(radius),where=radius>0);na=newmean+scale*dev;nc=newmean-scale*dev;nb=scale*nb
        a,b,c=na,nb,nc
    x=np.stack([(a+c-2)/math.sqrt(2),(a-c)/math.sqrt(2),math.sqrt(2)*b],axis=1);emp=np.cov(x,rowvar=False,bias=True)/alpha
    covariance_error=float(np.linalg.norm(emp-cov)/np.linalg.norm(cov));trace_ratio=float(np.mean(a+c-2)/(alpha*r2));root_mean=0.
    for ep in [-1,1]:
        for ze in [-1,1]:
            sa=1+r2+2*u*ep;sc=1+r2-2*u*ep;sb=2*v*ze;k=np.sqrt(a*c-b*b)*(1-r2);root_mean+=np.sqrt(a*sa+2*b*sb+c*sc+2*k)/4
    excess=.5*(a+c+2*(1+r2))-root_mean-r2;risk_ratio=float(excess.mean()/(alpha*r2/2))
    assert covariance_error<.12,(alpha,covariance_error)
    assert abs(trace_ratio-1)<.12,(alpha,trace_ratio)
    assert abs(risk_ratio-1)<.12,(alpha,risk_ratio)
    return dict(alpha=alpha,independent_chains=N,burn_steps=nsteps,covariance_relative_error=covariance_error,trace_bias_empirical_over_asymptotic=trace_ratio,risk_excess_empirical_over_asymptotic=risk_ratio,projection_frequency=clips/(N*nsteps))

def momentum_energy():
    rng=np.random.default_rng(20261102);n=6000;p=3;T=512;beta=.95;delta=1-beta;lo=.7;hi=1.3;L=2.;sigma2=p*.25**2
    eta=.5*delta*lo/(math.sqrt(96)*L*hi**2);aa=beta**2*L**2*eta**2*hi**2/delta**2
    theta=np.tile(np.array([.8,-.6,.3]),(n,1));mh=np.zeros_like(theta);mn=mh.copy();m=mh.copy();HH=DD=NN=MM=EE=dh2=0.;previous_h=None;dep=np.zeros(p);max_decomp=0.
    for step in range(1,T+1):
        h=theta*np.array([.5,1.,2.]);xi=rng.normal(0,.25,(n,p));g=h+xi
        mh=beta*mh+delta*h;mn=beta*mn+delta*xi;m=beta*m+delta*g
        denom=1-beta**step;hat=m/denom;drift=mh/denom-h;noise=mn/denom
        max_decomp=max(max_decomp,float(np.max(np.abs(hat-h-drift-noise))))
        HH+=np.mean(np.sum(h*h,axis=1));DD+=np.mean(np.sum(drift*drift,axis=1));NN+=np.mean(np.sum(noise*noise,axis=1));MM+=np.mean(np.sum(hat*hat,axis=1));EE+=np.mean(np.sum((hat-h)**2,axis=1))
        if previous_h is not None:dh2+=np.mean(np.sum((h-previous_h)**2,axis=1))
        previous_h=h
        # This operator explicitly depends on the SAME noisy gradient.
        Bdiag=1+.3*np.tanh(g);dep+=np.mean(Bdiag*xi,axis=0)/T
        theta-=eta*Bdiag*hat
    convolution_ratio=DD/(beta**2/delta**2*dh2);closure_ratio=EE/(12*aa*HH+(12*aa+2)*NN)
    assert max_decomp<1e-12 and convolution_ratio<=1+1e-10 and closure_ratio<=1+1e-10
    exact_N=sigma2*sum(delta/(1+beta)*(1+beta**t)/(1-beta**t) for t in range(1,T+1));noise_ratio=NN/exact_N;assert abs(noise_ratio-1)<.04
    gap=.5*np.sum(np.array([.5,1.,2.])*np.array([.8,-.6,.3])**2)
    coarse_bound=8*gap/(eta*lo*T)+13*(hi/lo)**2*sigma2*(delta/(1+beta)+2*beta/(T*delta*(1+beta)))
    q=L*eta*hi;r=L*eta*hi*beta/delta
    theorem_bound=2*gap/(eta*lo*T)+(hi/lo)*(1-q)/(1-q-r*r)*exact_N/T
    assert HH/T<=theorem_bound
    return dict(paths=n,steps=T,decomposition_max_error=max_decomp,drift_convolution_ratio=convolution_ratio,error_energy_closure_ratio=closure_ratio,noise_energy_empirical_over_exact=noise_ratio,same_batch_operator_noise_correlation_norm=float(np.linalg.norm(dep)),observed_average_gradient_square=HH/T,theorem_upper_bound=theorem_bound,previous_coarse_upper_bound=coarse_bound)

def threshold_limit():
    rng=np.random.default_rng(1704);N=240000;m=np.array([.6,.8]);P=np.diag([2.,1.]);y=P@m;q=np.linalg.norm(y);r=np.linalg.norm(m);s=r/q
    bas=np.array([[[1.,0],[0,0]],[[0,0],[0,1]],[[0,1/math.sqrt(2)],[1/math.sqrt(2),0]]]);A=np.stack([H@m for H in bas],axis=1);Cx=np.diag([.05,.02,.08]);ll=-r*(y@A)/(q**3)
    Z=rng.multivariate_normal(np.zeros(3),Cx,size=N);a=Z@A.T;ell=Z@ll;Y=s*a+np.maximum(ell,0)[:,None]*y[None,:]
    ss=float(ll@Cx@ll);cross=A@Cx@ll;mean=math.sqrt(ss/(2*math.pi))*y
    cov=s*s*A@Cx@A.T+s/2*(np.outer(cross,y)+np.outer(y,cross))+(0.5-1/(2*math.pi))*ss*np.outer(y,y)
    empirical_mean=Y.mean(0);empirical_cov=np.cov(Y,rowvar=False,bias=True);mean_error=float(np.linalg.norm(empirical_mean-mean)/np.linalg.norm(mean));cov_error=float(np.linalg.norm(empirical_cov-cov)/np.linalg.norm(cov))
    assert mean_error<.025 and cov_error<.025
    alpha=.001;raw_y=y[None,:]+math.sqrt(alpha)*a;raw_q=np.linalg.norm(raw_y,axis=1);gain=np.clip(r/raw_q,s,10.)
    scaled=(gain[:,None]*raw_y-s*y[None,:])/math.sqrt(alpha);finite_alpha_mean_error=float(np.linalg.norm(scaled.mean(0)-mean)/np.linalg.norm(mean))
    assert finite_alpha_mean_error<.04
    return dict(Gaussian_draws=N,threshold_mean_relative_error=mean_error,threshold_covariance_relative_error=cov_error,finite_alpha=alpha,finite_alpha_scaled_mean_relative_error=finite_alpha_mean_error)

def main():
    start=time.time();checks=deterministic_checks();orbit,cov=orbit_calculus();sim=[simulate_orbit(a,cov) for a in [.002,.001]]
    out=dict(passed=True,identities=checks,noncommuting_orbit=orbit,stationary_simulation=sim,same_batch_momentum_energy=momentum_energy(),matching_threshold=threshold_limit(),elapsed_seconds=time.time()-start)
    (ROOT/'experiments/tests/deep_analysis_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
