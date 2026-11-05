"""Check split isolation, held-out access guards, real batch shapes and fair selection."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from new_tasks.data import TaskData
from new_tasks.models import make_model
from new_tasks.run import opts_for,evaluate
from new_tasks.dispatch import select_rate
torch.set_num_threads(1);jobs=json.loads((ROOT/'experiments/new_tasks/jobs.json').read_text());tasks=json.loads((ROOT/'experiments/new_tasks/protocol.json').read_text())['tasks'];checked=[]
for task in tasks:
 name=task['task'];d=TaskData(ROOT/'data/new_tasks',name,42);assert 'test' not in d.cache
 try:list(d.eval_batches('test'))
 except RuntimeError:pass
 else:raise AssertionError('Test data accessible during tuning')
 assert 'test' not in d.cache and d.test_access_events==0
 x,y=d.train_batch(1,4);initial=[]
 for method in ['adamw','euclidean','bures']:
  torch.manual_seed(42);model=make_model(d.meta);job=next(j for j in jobs if j['task']==name and j['phase']=='tune' and j['method']==method and j['lr']==.0003);initial.append(hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest());opts=opts_for(model,job)
  pred=model(x);loss=torch.nn.functional.cross_entropy(pred,y) if d.family=='sequence_classification' else torch.nn.functional.mse_loss(pred,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  for opt in opts:opt.step()
  val,score,_=evaluate(model,d,'val');assert np.isfinite(val) and np.isfinite(score)
 assert len(set(initial))==1 and 'test' not in d.cache
 z=np.load(ROOT/'data/new_tasks'/f'{name}.npz')
 if name=='uci_har':
  groups=[set(z[k+'_subjects']) for k in ['train','val','test']];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 if name=='olivetti_denoise':
  groups=[set(v) for v in d.meta['subject_ids'].values()];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 if d.family=='forecasting':
  assert np.allclose(z['train_series'][-96:],z['val_series'][:96]);assert np.allclose(z['val_series'][-96:],z['test_series'][:96]);assert np.max(np.abs(z['train_series'].mean(0)))<1e-4
 checked.append(name)
# Selection depends on validation score and never accepts a test-exposed tuning record.
fakejobs=[{'id':'a','direction':'min','lr':.001},{'id':'b','direction':'min','lr':.0001}];fake={j['id']:{'status':'complete','phase':'tune','test_evaluations':0,'test_data_loaded':False,'validation_selected':{'val_metric':1.}} for j in fakejobs};assert select_rate(fakejobs,fake)=='b';fake['a']['test_evaluations']=1
try:select_rate(fakejobs,fake)
except AssertionError:pass
else:raise AssertionError('Test-exposed tuning run accepted')
result={'passed':True,'tasks_checked':checked,'method_model_initializations_match':True,'test_access_blocked':True,'validation_only_rate_selection':True};(ROOT/'analysis/NEW_TASK_PROTOCOL_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
