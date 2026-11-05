"""Regression: a rectangular tile must have the same update in isolation."""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bures.optimizer import MatrixOptimizer

def run():
    torch.manual_seed(42)
    results=[]
    for rule in ("bures","euclidean"):
        whole=torch.nn.Parameter(torch.randn(5,7,dtype=torch.float64))
        opt=MatrixOptimizer([whole], lr=.003,block_size=3,tau_root=2,
            covariance_rule=rule,state_dtype=torch.float64,s_min=1,s_max=1)
        pieces=[]
        for i in range(0,5,3):
            for j in range(0,7,3):
                p=torch.nn.Parameter(whole[i:i+3,j:j+3].detach().clone())
                o=MatrixOptimizer([p],lr=.003,block_size=3,tau_root=2,
                    covariance_rule=rule,state_dtype=torch.float64,s_min=1,s_max=1)
                pieces.append((i,j,p,o))
        worst=0.
        for t in range(4):
            g=torch.randn_like(whole)
            whole.grad=g; opt.step()
            for i,j,p,o in pieces:
                p.grad=g[i:i+3,j:j+3].clone();o.step()
                worst=max(worst,float((whole[i:i+3,j:j+3]-p).abs().max()))
        assert len(opt.state[whole]["L"])==6
        assert len(opt.state[whole]["R"])==6
        assert worst<1e-10,(rule,worst)
        results.append(dict(rule=rule,max_abs_error=worst,tiles=6))
    print(json.dumps(results,indent=2))
if __name__=="__main__": run()
