"""Finite prospective study: all 108 validation-only trials, then all 90 full runs."""
import concurrent.futures,fcntl,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def atomic(path,obj):
 path=Path(path);tmp=path.with_suffix('.dispatcher.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def select_rate(jobs,records):
 eligible=[]
 for job in jobs:
  r=records.get(job['id'],{})
  if r.get('status')!='complete':continue
  assert r.get('phase')=='tune' and r.get('test_evaluations')==0 and not r.get('test_data_loaded'),('test contamination',job['id'])
  v=r.get('validation_selected',{}).get('val_metric')
  if isinstance(v,(int,float)) and math.isfinite(v):eligible.append(((-v if job['direction']=='max' else v),job['lr'],job['id']))
 return min(eligible)[2] if eligible else None

def main():
 lock=(ROOT/'experiments/new_tasks/.dispatch.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 protocol=json.loads((ROOT/'experiments/new_tasks/protocol.json').read_text());expected=(ROOT/'experiments/new_tasks/protocol.sha256').read_text().strip();assert hashlib.sha256((ROOT/'experiments/new_tasks/protocol.json').read_bytes()).hexdigest()==expected
 jobs=json.loads((ROOT/'experiments/new_tasks/jobs.json').read_text());results=ROOT/'experiments/extension/results';results.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+7*24*3600
 def record(j):
  try:return json.loads((results/f"{j['id']}.json").read_text())
  except (OSError,json.JSONDecodeError):return {}
 def run(j):
  old=record(j)
  if old.get('status') in ('complete','failed'):return
  path=ROOT/'experiments/new_tasks/job_files'/f"{j['id']}.json";path.parent.mkdir(parents=True,exist_ok=True);atomic(path,j)
  command=[sys.executable,str(ROOT/'experiments/new_tasks/run.py'),'--job',str(path),'--data',str(ROOT/'data/new_tasks')];log=ROOT/'experiments/extension/logs'/f"{j['id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
  print('START',j['id'],flush=True)
  try:
   with log.open('a') as stream:
    proc=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=21600,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
   if not record(j) and proc.returncode!=0:atomic(results/f"{j['id']}.json",{**j,'status':'failed','history':[],'error':f'Runner failed before initializing record (exit {proc.returncode}); see log','failure_kind':'infrastructure','test_evaluations':0,'test_data_loaded':False})
  except subprocess.TimeoutExpired:
   r=record(j) or {**j,'history':[]};r.update(status='failed',error='Six-hour operational limit reached; the declared update budget did not complete.',failure_kind='operational_timeout');atomic(results/f"{j['id']}.json",r)
  print('END',j['id'],record(j).get('status'),flush=True)
 def phase(arms):
  with concurrent.futures.ThreadPoolExecutor(max_workers=protocol['compute']['workers']) as pool:list(pool.map(run,arms))
  while any(record(j).get('status') not in ('complete','failed') for j in arms):
   if time.monotonic()>deadline:raise TimeoutError('Finite study exceeded its seven-day dispatch window.')
   time.sleep(30)
 tune=[j for j in jobs if j['phase']=='tune'];phase(tune);tune_records={j['id']:record(j) for j in tune};selected={};full=[]
 for task in protocol['tasks']:
  for method in protocol['methods']:
   subset=[j for j in tune if j['task']==task['task'] and j['method']==method];key=select_rate(subset,tune_records)
   selected[task['task']+'::'+method]=None if key is None else {'tuning_run':key,'lr':tune_records[key]['lr'],'validation_metric':tune_records[key]['validation_selected']['val_metric'],'tuning_record_sha256':hashlib.sha256((results/f'{key}.json').read_bytes()).hexdigest()}
 for template in [j for j in jobs if j['phase']=='full']:
  choice=selected[template['task']+'::'+template['method']]
  if choice is None:
   if record(template).get('status') not in ('complete','failed'):atomic(results/f"{template['id']}.json",{**template,'status':'failed','history':[],'failure_kind':'no_stable_validation_configuration','error':'All six declared tuning rates failed or were ineligible; no full run or test evaluation performed.','test_evaluations':0,'test_data_loaded':False})
   continue
  full.append({**template,'lr':choice['lr'],'selected_from_tuning':choice['tuning_run'],'selection_record_sha256':choice['tuning_record_sha256']})
 atomic(results/'_newtask_execution.json',{'protocol_sha256':expected,'selected_rates':selected,'materialized_full_jobs':full,'selection_uses':'validation only; every declared task and seed retained'})
 phase(full);print('COMPLETE finite prospective study',flush=True)
if __name__=='__main__':main()
