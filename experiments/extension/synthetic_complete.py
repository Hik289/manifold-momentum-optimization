"""Proposal synthetic suite with metric-specific reference targets and raw curves."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'data/synthetic'))
from bw_geom import bw_geodesic,bw_distance,sqrtm,inv_sqrtm,spd_power,sym
rng=np.random.default_rng(42);out={};curves={}
def spd(d):
 q,_=np.linalg.qr(rng.normal(size=(d,d)));return (q*rng.uniform(.3,3,d))@q.T

def update(A,B,alpha,method):
 if method=='bures':return bw_geodesic(A,B,alpha)
 if method=='euclidean':return (1-alpha)*A+alpha*B
 if method=='log_euclidean':
  def log(C):w,q=np.linalg.eigh(C);return (q*np.log(w))@q.T
  w,q=np.linalg.eigh((1-alpha)*log(A)+alpha*log(B));return (q*np.exp(w))@q.T
 ah=sqrtm(A);aih=inv_sqrtm(A);return sym(ah@spd_power(aih@B@aih,alpha)@ah)

# Exact finite-population BW reference via offline fixed point; NOT arithmetic mean.
d=8;sample=[spd(d) for _ in range(128)];C=sum(sample)/len(sample)
for _ in range(1000):
 ah=sqrtm(C);aih=inv_sqrtm(C);H=sum(sqrtm(ah@S@ah) for S in sample)/len(sample);new=sym(aih@H@H@aih)
 if np.linalg.norm(new-C)<1e-12:C=new;break
 C=new
A=spd(d);B=spd(d)
q,_=np.linalg.qr(rng.normal(size=(d,d)))
from scipy.linalg import expm
O=rng.normal(size=(d,d));O=(O-O.T)*.001
for setting in ['static','scale','rotation','geodesic','jump','rotscale']:
 for method in ['euclidean','bures','log_euclidean','ai']:
  state=np.eye(d);rows=[];previous=inv_sqrtm(state)
  for t in range(1000):
   if setting=='static':S=sample[t%len(sample)];target=C
   elif setting=='geodesic':target=bw_geodesic(A,B,(np.sin(t*.006)+1)/2);S=target
   elif setting=='jump':target=A if t<500 else B;S=target
   elif setting=='scale':target=(q*np.exp(.7*np.sin(t*.01+np.arange(d))))@q.T;S=target
   else:
    Q=expm(t*O);lam=np.geomspace(.3,3,d)
    if setting=='rotscale':lam=lam*np.exp(.5*np.sin(t*.01+np.arange(d)))
    target=(Q*lam)@Q.T;S=target
   state=update(state,S,.01,method);root=inv_sqrtm(state)
   rows.append([t,bw_distance(state,target),np.linalg.norm(state-target),np.linalg.norm(root-inv_sqrtm(target),2),np.linalg.norm(root-previous),float(np.linalg.cond(state))]);previous=root
  a=np.asarray(rows);key=f'{setting}_{method}';curves[key]=a
  out[key]=dict(mean_BW=float(a[:,1].mean()),final_BW=float(a[-1,1]),mean_root_error=float(a[:,3].mean()),mean_jitter=float(a[:,4].mean()),max_condition=float(a[:,5].max()),target='offline empirical BW barycenter of 128 SPD atoms' if setting=='static' else 'known noiseless path')
  if setting=='jump':
   threshold=.1*bw_distance(A,B);idx=np.where(a[500:,1]<=threshold)[0]
   out[key]['recovery_steps_to_10percent']=int(idx[0]+1) if len(idx) else None

# Independent replicated scale filters validate alpha/(2-alpha), and drift/noise grids.
noise=[]
for alpha in [.005,.01,.03,.1,.3]:
 y=1+.2*rng.choice([-1.,1.],size=(8000,400));u=np.ones(400)
 for t in range(len(y)):u=(1-alpha)*u+alpha*y[t]
 empirical=float(np.mean((u-1)**2));theory=alpha/(2-alpha)*.04
 noise.append(dict(alpha=alpha,empirical_mse=empirical,exact_mse=theory,ratio=empirical/theory))
out['scalar_noise_scaling']=noise
dynamic=[]
for alpha in [.01,.03,.1]:
 for delta in [0,.0001,.001]:
  u=np.ones(400);errs=[]
  for t in range(2000):
   target=1+delta*t;y=target+.1*rng.choice([-1.,1.],size=400);u=(1-alpha)*u+alpha*y
   if t>1000:errs.append(np.mean((u-target)**2))
  dynamic.append(dict(alpha=alpha,drift=delta,mse=float(np.mean(errs)),noise_proxy=alpha/(2-alpha)*.01,lag_squared=((1-alpha)*delta/alpha)**2))
out['dynamic_grid']=dynamic
# Two-point objective stress test, dimensions and counts exactly as proposal.
optimality=[]
for d in [16,64]:
 worst=-float('inf');resid=0.;fail=0
 for _ in range(1000):
  A=spd(d);B=spd(d);alpha=.3;C=bw_geodesic(A,B,alpha)
  obj=(1-alpha)*bw_distance(C,A)**2+alpha*bw_distance(C,B)**2
  expected=alpha*(1-alpha)*bw_distance(A,B)**2
  resid=max(resid,abs(obj-expected));fail+=int(np.linalg.eigvalsh(C).min()<=0)
  candidate=spd(d);other=(1-alpha)*bw_distance(candidate,A)**2+alpha*bw_distance(candidate,B)**2
  worst=max(worst,obj-other)
 optimality.append(dict(d=d,pairs=1000,max_exact_objective_residual=resid,max_candidate_gap=worst,spd_failures=fail))
out['two_point_1000_pairs']=optimality
p=Path(__file__).resolve().parent/'results';p.mkdir(exist_ok=True)
(p/'synthetic_complete.json').write_text(json.dumps(out,indent=2));np.savez_compressed(p/'synthetic_complete_curves.npz',**curves)
print(json.dumps(out,indent=2))
