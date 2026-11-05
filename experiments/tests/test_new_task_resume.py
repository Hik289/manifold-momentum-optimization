"""A resumed FP64 matrix state must match uninterrupted FP32-weight training."""
import copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from bures.optimizer import MatrixOptimizer
from new_tasks.run import restore_optimizers
torch.set_num_threads(1);torch.manual_seed(99)
for rule in ['euclidean','bures']:
 w=torch.nn.Parameter(torch.randn(4,3,dtype=torch.float32));opt=MatrixOptimizer([w],lr=.0003,covariance_rule=rule,state_dtype=torch.float64,block_size=3)
 def step(p,o,t):
  g=torch.randn(p.shape,generator=torch.Generator().manual_seed(t));p.grad=g;o.step()
 step(w,opt,1);saved=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(w.detach().clone());other=MatrixOptimizer([q],lr=.0003,covariance_rule=rule,state_dtype=torch.float64,block_size=3);restore_optimizers([other],[saved])
 assert all(x.dtype==torch.float64 for x in other.state[q]['L'].values())
 for t in [2,3,4]:step(w,opt,t);step(q,other,t)
 assert torch.equal(w,q),(rule,float((w-q).abs().max()))
print({'passed':True,'FP64_covariance_and_cache_preserved':True,'uninterrupted_vs_resumed_parameters':'exact'})
