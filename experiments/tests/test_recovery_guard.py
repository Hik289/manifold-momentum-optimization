"""Check the certificate using actual parameter displacements, including startup bias."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from bures.optimizer import MatrixOptimizer
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
w=torch.nn.Parameter(torch.zeros(3,2));lr=.001;opt=MatrixOptimizer([w],lr=lr,betas=(.9,.99),s_min=0,s_max=10,state_dtype=torch.float64,weight_decay=0,tau_root=10)
m=torch.zeros_like(w);max_ratio=0.;max_error=0.;startup=None
for step in range(1,21):
 g=torch.tensor([[1.,2.],[-3.,.2],[.1,-.7]])*((-1)**step)*(1 if step%3 else .05);g=g/g.norm().clamp(min=1);w.grad=g.clone();before=w.detach().clone();opt.step();m=.9*m+.1*g;hat=m/(1-.9**step);actual=(before-w.detach())/lr
 d=opt.last_diag;ratio=d['precond_update_norm']*(1-.9**step)/d['first_moment_norm'];max_ratio=max(max_ratio,ratio);err=abs(float(actual.norm())-d['precond_update_norm']);max_error=max(max_error,err)
 assert actual.norm()<=hat.norm()*(1+1e-10);assert ratio<=1+1e-10;assert err<1e-10
 if step==1:startup=d['precond_update_norm']/d['first_moment_norm']
r=dict(passed=True,steps=20,max_actual_vs_reported_update_norm_error=max_error,max_bias_corrected_update_over_moment=max_ratio,uncorrected_startup_ratio=startup);(ROOT/'experiments/tests/recovery_guard_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
