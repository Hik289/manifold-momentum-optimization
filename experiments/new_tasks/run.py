"""Frozen new-task study: no test evaluation in tuning; all full seeds retained."""
import argparse,fcntl,hashlib,json,math,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from extension import run_task as base
from bures.optimizer import MatrixOptimizer
from new_tasks.data import TaskData
from new_tasks.models import make_model

def opts_for(model,job):
 kw={'lr':job['lr'],'betas':(.9,.99),'weight_decay':job['wd']}
 if job['method']=='adamw':return [torch.optim.AdamW(model.parameters(),**kw)]
 chosen=list(model.parameters()) if job['family']=='denoising' else [p for n,p in model.named_parameters() if n.startswith('rnn.') and p.ndim==2];ids={id(p) for p in chosen};rest=[p for p in model.parameters() if id(p) not in ids]
 matrix=MatrixOptimizer(chosen,covariance_rule=job['method'],state_dtype=torch.float64,block_size=job['block'],tau_root=job['tau'],s_min=job['smin'],s_max=job['smax'],**kw)
 return [matrix]+([torch.optim.AdamW(rest,**kw)] if rest else [])
def restore_optimizers(optimizers,states):
 for opt,saved in zip(optimizers,states):
  opt.load_state_dict(saved)
  if isinstance(opt,MatrixOptimizer):
   for live_group,saved_group in zip(opt.param_groups,saved['param_groups']):
    for param,index in zip(live_group['params'],saved_group['params']):
     original=saved['state'].get(index,{})
     if 'L' in original:
      for key in ['L','R','Lih_cache','Rih_cache']:
       opt.state[param][key]={tile:value.detach().clone().to(device=param.device,dtype=live_group['state_dtype']) for tile,value in original[key].items()}

@torch.no_grad()
def evaluate(model,data,split):
 model.eval();loss_sum=0.;mae_sum=0.;count=0;correct=0
 for x,y in data.eval_batches(split):
  out=model(x);loss=F.cross_entropy(out,y) if data.family=='sequence_classification' else F.mse_loss(out,y);loss_sum+=float(loss)*len(x);count+=len(x)
  if data.family=='sequence_classification':correct+=int((out.argmax(-1)==y).sum())
  else:mae_sum+=float(F.l1_loss(out,y))*len(x)
 assert count>0
 return loss_sum/count,correct/count if data.family=='sequence_classification' else loss_sum/count,mae_sum/count if data.family!='sequence_classification' else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--data',required=True);p.add_argument('--out');p.add_argument('--smoke',action='store_true');args=p.parse_args();job=json.loads(Path(args.job).read_text());assert isinstance(job['lr'],(int,float)) and job['lr']>0
 seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 data=TaskData(args.data,job['task'],seed);model=make_model(data.meta);opts=opts_for(model,job)
 out=Path(args.out) if args.out else ROOT/'experiments/extension/results'/f"{job['id']}.json";out.parent.mkdir(parents=True,exist_ok=True);lock=out.with_suffix('.runlock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed'):return
 protocol=(ROOT/'experiments/new_tasks/protocol.json').read_bytes();meta={**job,'status':'running','device':'cpu','schema_version':2,'history':[],'seed':seed,'initialization_sha256':base.digest(model),'parameters':sum(p.numel() for p in model.parameters()),'protocol_sha256':hashlib.sha256(protocol).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'experiments/bures/optimizer.py').read_bytes()).hexdigest(),'models_sha256':hashlib.sha256((ROOT/'experiments/new_tasks/models.py').read_bytes()).hexdigest(),'data_loader_sha256':hashlib.sha256((ROOT/'experiments/new_tasks/data.py').read_bytes()).hexdigest(),'data_sha256':data.meta['prepared_sha256'],'data_meta':data.meta,'metric_direction':job['direction'],'test_evaluations':0,'test_data_loaded':False,'smoke':args.smoke}
 checkpoint=out.with_suffix('.checkpoint.pt');start_step=0;best=math.inf if job['direction']=='min' else -math.inf;best_step=0;best_weights=None;initial=meta['initialization_sha256'];start=time.monotonic();previous_wall=0.
 if checkpoint.exists() and out.exists() and not args.smoke:
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  for k in ['runner_sha256','models_sha256','data_loader_sha256','source_sha256','data_sha256','protocol_sha256']:
   assert ck['meta'][k]==meta[k],('resume checksum mismatch',k)
  assert ck['job']==job,'resume job mismatch';model.load_state_dict(ck['model'])
  restore_optimizers(opts,ck['opts'])
  meta=ck['meta'];start_step=ck['step'];best=ck['best'];best_step=ck['best_step'];best_weights=ck['best_weights'];previous_wall=meta.get('wall_seconds',0.);torch.set_rng_state(ck['torch_rng']);meta['resumed_from']=start_step
 budget=2 if args.smoke else job['epochs'];meta['epochs']=budget;meta['requested_epochs']=job['epochs'];base.atomic(out,base.clean(meta))
 try:
  for step in range(start_step+1,budget+1):
   model.train();x,y=data.train_batch(step,job['batch'])
   for opt in opts:opt.zero_grad(set_to_none=True)
   predicted=model(x);loss=F.cross_entropy(predicted,y) if data.family=='sequence_classification' else F.mse_loss(predicted,y)
   if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite training loss at update {step}')
   loss.backward();raw=float(torch.nn.utils.clip_grad_norm_(model.parameters(),job['gradient_clip'],error_if_nonfinite=True))
   for opt in opts:opt.step()
   if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError(f'nonfinite parameter at update {step}')
   row={'epoch':step,'step':step,'train_loss':float(loss.detach()),'wall_seconds':previous_wall+time.monotonic()-start,'raw_gradient_norm':raw}
   if step==1 or step%job['eval_every']==0 or step==budget:
    val,metric,mae=evaluate(model,data,'val');assert math.isfinite(val) and math.isfinite(metric);row.update(val_loss=val,val_metric=metric,val_mae=mae)
    if metric<best if job['direction']=='min' else metric>best:
     best=metric;best_step=step;best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
   meta['history'].append(row);meta['wall_seconds']=row['wall_seconds']
   if step==1 or step%job['eval_every']==0 or step==budget:base.atomic(out,base.clean(meta));print(job['id'],step,row.get('val_metric'),flush=True)
   if step%250==0 or step==budget:
    state={'job':job,'step':step,'model':model.state_dict(),'opts':[opt.state_dict() for opt in opts],'meta':meta,'best':best,'best_step':best_step,'best_weights':best_weights,'torch_rng':torch.get_rng_state()};tmp=checkpoint.with_suffix('.saving.pt');torch.save(state,tmp);tmp.replace(checkpoint)
  meta['final']={'epoch':budget,'val_metric':meta['history'][-1]['val_metric']};meta['validation_selected']={'epoch':best_step,'val_metric':best}
  if job['phase']=='full' and not args.smoke:
   assert budget==job['epochs']==5000;data.allow_test=True;tl,tm,mae=evaluate(model,data,'test');assert math.isfinite(tl) and math.isfinite(tm);meta['final'].update(test_loss=tl,test_metric=tm,test_mae=mae);model.load_state_dict(best_weights);tl,tm,mae=evaluate(model,data,'test');assert math.isfinite(tl) and math.isfinite(tm);meta['validation_selected'].update(test_loss=tl,test_metric=tm,test_mae=mae)
  else:assert not data.allow_test and 'test' not in data.cache
  meta.update(status='complete',test_evaluations=data.test_access_events,test_data_loaded='test' in data.cache)
 except Exception as e:meta.update(status='failed',error=repr(e),traceback=traceback.format_exc());print(meta['traceback'],flush=True)
 meta.update(test_evaluations=data.test_access_events,test_data_loaded='test' in data.cache)
 base.atomic(out,base.clean(meta))
 if meta['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
