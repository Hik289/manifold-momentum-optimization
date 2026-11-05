"""Independent numerical checks for the formal noncommutative and stochastic identities."""
import json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bures.core import spd_sqrt,spd_inv_sqrt,ot_map,bw_geodesic,bw_distance

def run():
 torch.manual_seed(321);torch.set_num_threads(1);dt=torch.float64
 def spd(d):
  x=torch.randn(d,d,dtype=dt);return x@x.T/d+torch.eye(d,dtype=dt)
 def syl(C,E):
  w,q=torch.linalg.eigh(C);x=q.T@E@q;return q@(x/(w[:,None]+w[None,:]))@q.T
 def rel(x,y):return float((x-y).norm()/(1+y.norm()))
 worst={'loewner_gap':0.,'trace_gap':0.,'transport_derivative':0.,'square_root_derivative':0.,'constant_target_recursion':0.,'angle_violation':0.}
 for _ in range(25):
  d=4;A,B=spd(d),spd(d);E,F=torch.randn(d,d,dtype=dt),torch.randn(d,d,dtype=dt);E=(E+E.T)/2;F=(F+F.T)/2
  T=ot_map(A,B);alpha=.27;gap=(1-alpha)*A+alpha*B-bw_geodesic(A,B,alpha);expected=alpha*(1-alpha)*(T-torch.eye(d))@A@(T-torch.eye(d))
  worst['loewner_gap']=max(worst['loewner_gap'],rel(gap,expected));assert float(torch.linalg.eigvalsh(gap).min())>-1e-10
  worst['trace_gap']=max(worst['trace_gap'],abs(float(gap.trace())-alpha*(1-alpha)*float(bw_distance(A,B))**2))
  Ah=spd_sqrt(A);Aih=spd_inv_sqrt(A);C=Ah@T@Ah
  Y=Aih@syl(C,Ah@(F-T@E@T)@Ah)@Aih;eps=1e-5
  fd=(ot_map(A+eps*E,B+eps*F)-ot_map(A-eps*E,B-eps*F))/(2*eps)
  worst['transport_derivative']=max(worst['transport_derivative'],rel(fd,Y))
  fd=(spd_sqrt(A+eps*E)-spd_sqrt(A-eps*E))/(2*eps)
  worst['square_root_derivative']=max(worst['square_root_derivative'],rel(fd,syl(Ah,E)))
  state=A.clone()
  for t in range(1,8):
   state=bw_geodesic(state,B,alpha);direct=bw_geodesic(A,B,1-(1-alpha)**t)
   worst['constant_target_recursion']=max(worst['constant_target_recursion'],rel(state,direct))
  P=spd(d);g=torch.randn(d,dtype=dt);w=torch.linalg.eigvalsh(P);gamma=2*torch.sqrt(w[0]*w[-1])/(w[0]+w[-1]);cos=torch.dot(g,P@g)/(g.norm()*(P@g).norm())
  worst['angle_violation']=max(worst['angle_violation'],float(gamma-cos))
 for key,val in worst.items():assert val<2e-7,(key,val)
 # Gaussian KL: dense covariance formula against the factored trace/logdet identity.
 L,R=spd(3),spd(2);H=torch.kron(R,L);p=6;tau=.7;sigma=1.3;mu=torch.randn(p,dtype=dt);Cq=tau**2*torch.linalg.inv(H)
 dense=.5*(mu.square().sum()/sigma**2+Cq.trace()/sigma**2-p+p*math.log(sigma**2)-torch.logdet(Cq))
 factored=.5*(mu.square().sum()/sigma**2+tau**2/sigma**2*torch.linalg.inv(R).trace()*torch.linalg.inv(L).trace()-p+p*math.log(sigma**2/tau**2)+3*torch.logdet(R)+2*torch.logdet(L))
 assert abs(float(dense-factored))<1e-10
 # Normalized-map Jacobian: independent automatic differentiation and exact skew.
 P=spd(4);m=torch.randn(4,dtype=dt,requires_grad=True)
 J=torch.autograd.functional.jacobian(lambda x: x.norm()*(P@x)/(P@x).norm(),m)
 r=m.norm();q=(P@m).norm()
 exact=r/q*P+torch.outer(P@m,m/(r*q)-r*(P@P@m)/q**3)
 jac_error=rel(J,exact);assert jac_error<1e-12
 Pd=torch.diag(torch.tensor([.7,2.3],dtype=dt));md=torch.ones(2,dtype=dt)
 Jd=torch.autograd.functional.jacobian(lambda x:x.norm()*(Pd@x)/(Pd@x).norm(),md)
 rd=md.norm();qd=(Pd@md).norm();p1,p2=Pd.diag()
 skew=(p1-p2)*(qd**2+rd**2*p1*p2)/(rd*qd**3)
 skew_error=abs(float(Jd[0,1]-Jd[1,0]-skew));assert skew_error<1e-12
 # Yule--Walker variance versus independent simulated momentum trajectories.
 beta=.8;eta=.1;b=.7;h=2.;n=20000;theta=torch.zeros(n,dtype=dt);moment=theta.clone()
 for t in range(1500):
  moment=beta*moment+(1-beta)*(h*theta+torch.randn(n,dtype=dt));theta=theta-eta*b*moment
 target=eta*b*(1+beta)/(h*(2*(1+beta)-eta*b*(1-beta)*h));ratio=float(theta.var()/target)
 assert abs(ratio-1)<.06,ratio
 result={'passed':True,'matrix_checks':worst,'gaussian_KL_absolute_error':abs(float(dense-factored)),'normalized_Jacobian_relative_error':jac_error,'normalized_Jacobian_skew_error':skew_error,'momentum_variance_empirical_over_exact':ratio}
 Path(__file__).with_name('formal_analysis_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
