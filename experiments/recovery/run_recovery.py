"""Explicit stability-recovery cohort; failed historical records are immutable."""
import argparse,fcntl,hashlib,json,math,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from extension import run_task as base
from bures.optimizer import MatrixOptimizer
from pipeline.diagnostics import DiagnosticRecorder

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--data',required=True);raw=p.parse_args();job=json.loads(Path(raw.job).read_text())
 a=SimpleNamespace(**job,data=raw.data,device='cpu');base.seed_all();torch.set_num_threads(1);device=torch.device('cpu')
 out=ROOT/'experiments/extension/results'/f'{a.id}.json';out.parent.mkdir(parents=True,exist_ok=True)
 lock=out.with_suffix('.runlock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed'):return
 meta={**job,'device':'cpu','schema_version':2,'status':'running','history':[],'diagnostics':[],'torch':torch.__version__,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'experiments/bures/optimizer.py').read_bytes()).hexdigest(),'core_sha256':hashlib.sha256((ROOT/'experiments/bures/core.py').read_bytes()).hexdigest(),'recovery_start':'restart from the original initialization; changed safeguards are a separate cohort'}
 base.atomic(out,meta);start=time.perf_counter()
 try:
  model,data,lossfn,evaluate,maximize,data_meta=base.setup(a,device);initial=base.digest(model)
  for parent in a.parent_failed_ids:
   old=json.loads((ROOT/'experiments/extension/results'/f'{parent}.json').read_text())
   assert initial==old['initialization_sha256'],(parent,'initialization mismatch')
   assert data_meta==old['data_meta'],(parent,'data mismatch')
  opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,a.beta2),weight_decay=a.wd) if a.method=='adamw' else MatrixOptimizer(model.parameters(),lr=a.lr,betas=(.9,a.beta2),weight_decay=a.wd,block_size=a.block,tau_root=a.tau,covariance_rule=a.method,s_min=0.,s_max=a.smax,state_dtype=torch.float64)
  rec=DiagnosticRecorder(opt);meta.update(initialization_sha256=initial,data_meta=data_meta,parameters=sum(p.numel() for p in model.parameters()),metric_direction='max' if maximize else 'min')
  best=-math.inf if maximize else math.inf;best_epoch=0;best_weights=None;steps=0;clip_events=0;max_ratio=0.
  for epoch in range(1,a.epochs+1):
   a.current_epoch=epoch;model.train();total=0.;seen=0;raw_norm=0.
   if a.family=='lm':batches=[(None,None)]
   else:
    gen=torch.Generator().manual_seed(42+epoch);idx=torch.randperm(len(data[0][0]),generator=gen)
    batches=((data[0][0][i],data[0][1][i]) for i in idx.split(a.batch))
   for x,y in batches:
    opt.zero_grad(set_to_none=True);loss=lossfn(epoch) if x is None else lossfn(x,y)
    if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite training loss at {epoch}')
    loss.backward();raw_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip,error_if_nonfinite=True));clip_events+=int(raw_norm>a.gradient_clip)
    opt.step();steps+=1
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError(f'nonfinite parameter after step {steps}')
    diag=getattr(opt,'last_diag',{});moment=diag.get('first_moment_norm',0.);direction=diag.get('precond_update_norm',0.)
    if moment>0:
     ratio=direction*(1-.9**steps)/moment;max_ratio=max(max_ratio,ratio)
     if ratio>1+2e-5:raise ArithmeticError(f'norm matching violated its nonamplification certificate: {ratio}')
    n=1 if x is None else len(x);total+=float(loss.detach())*n;seen+=n
   model.eval();vl,metric=evaluate('val',epoch)
   if not math.isfinite(vl) or not math.isfinite(metric):raise FloatingPointError(f'nonfinite validation at {epoch}')
   if metric>best if maximize else metric<best:
    best=metric;best_epoch=epoch;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   row=dict(epoch=epoch,step=steps,train_loss=total/seen,val_loss=vl,val_metric=metric,wall_seconds=time.perf_counter()-start,raw_gradient_norm=raw_norm,clipped_steps=clip_events,max_matrix_direction_over_moment=max_ratio)
   meta['history'].append(row)
   if epoch==1 or epoch%a.diag_every==0 or epoch==a.epochs:
    rec.record(epoch,train_loss=row['train_loss'],val_metric=metric,learning_rate=a.lr,checkpoint_meta={'recovery':True,'epoch':epoch});meta['diagnostics']=base.clean(rec.trajectory)
    meta.update(clipped_steps=clip_events,optimizer_steps=steps,max_matrix_direction_over_moment=max_ratio);base.atomic(out,base.clean(meta));print(a.id,epoch,vl,metric,flush=True)
   if epoch%100==0 or epoch==a.epochs:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best_epoch=best_epoch,best_weights=best_weights),out.with_suffix('.checkpoint.pt'))
  model.eval();tl,tm=evaluate('test',a.epochs);meta['final']=dict(epoch=a.epochs,test_loss=tl,test_metric=tm,val_metric=meta['history'][-1]['val_metric'])
  model.load_state_dict(best_weights);model.eval();tl,tm=evaluate('test',best_epoch)
  meta.update(validation_selected=dict(epoch=best_epoch,val_metric=best,test_loss=tl,test_metric=tm),status='complete',wall_seconds=time.perf_counter()-start)
 except Exception as exc:meta.update(status='failed',error=repr(exc),traceback=traceback.format_exc());print(meta['traceback'],flush=True)
 base.atomic(out,base.clean(meta))
 if meta['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
