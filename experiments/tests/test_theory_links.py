"""Independent numerical checks of the new geometry/direction inequalities."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bures.core import bures_barycenter_2pt,spd_inv_sqrt,bw_distance

def run():
 torch.manual_seed(42);torch.set_num_threads(1);dt=torch.float64
 def spd():
  q,_=torch.linalg.qr(torch.randn(4,4,dtype=dt));return (q*torch.linspace(.5,2,4,dtype=dt))@q.T
 ratios=[]
 for _ in range(100):
  A,B=spd(),spd();delta=float((A-B).norm());d=float(bw_distance(A,B));r=float((spd_inv_sqrt(A)-spd_inv_sqrt(B)).norm(2))
  assert delta/(2*2**.5)<=d+1e-9
  assert d<=delta/(2*.5**.5)+1e-9
  assert r<=delta/(2*.5**1.5)+1e-9
  ratios.append(r/(delta/(2*.5**1.5)))
 A,B=spd().requires_grad_(),spd().requires_grad_()
 gradcheck=torch.autograd.gradcheck(lambda x,y:bures_barycenter_2pt(x,y,.3),(A,B),eps=1e-6,atol=2e-5,rtol=2e-4)
 # Full normalized direction continuity, including zero and tiny moments.
 def U(P,m):
  if float(m.norm())==0:return m
  s=(m.norm()/(torch.linalg.vector_norm(P@m)+1e-12)).clamp(.1,10)
  return s*(P@m)
 maxratio=0
 for scale in [0.,1e-15,1e-6,1.,1e4]:
  for _ in range(100):
   P=spd();m=scale*torch.randn(4,dtype=dt);n=torch.randn(4,dtype=dt)
   ratio=float((U(P,m)-U(P,n)).norm()/((10*2+1+2/.5)*(m-n).norm()))
   assert ratio<=1+1e-10;maxratio=max(maxratio,ratio)
 result=dict(root_bound_max_ratio=max(ratios),barycenter_autograd_gradcheck=bool(gradcheck),normalized_direction_max_ratio=maxratio,passed=True)
 Path(__file__).with_name('theory_links_results.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':run()
