"""Run the two already-declared image SGD controls after the seed follow-up.
Uses two low-priority CPU workers and preserves separate CPU logs.
"""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
lock=(root/'experiments/expansion/.pending-sgd.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)
repeat=json.loads((root/'experiments/expansion/multiseed_jobs.json').read_text());deadline=time.monotonic()+7*24*3600
while time.monotonic()<deadline:
 status=[]
 for j in repeat:
  p=root/'experiments/extension/results'/f"{j['id']}.json"
  try:status.append(json.loads(p.read_text()).get('status'))
  except (OSError,json.JSONDecodeError):status.append(None)
 if all(s in ('complete','failed') for s in status):break
 time.sleep(30)
else:raise TimeoutError('The seed follow-up did not finish within seven days.')
jobs=[j for j in json.loads((root/'experiments/expansion/jobs.json').read_text()) if j['family']=='vision' and j['method']=='sgdm']
def run(j):
 out=root/'experiments/extension/results'/f"{j['id']}.json"
 if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed','running'):return
 path=root/'experiments/expansion/job_files'/f"{j['id']}.json";path.write_text(json.dumps(j))
 cmd=[sys.executable,str(root/'experiments/expansion/run_benchmark.py'),'--job',str(path),'--device','cpu','--data',str(root/'data'),'--suite-data',str(root/'data/expansion')]
 print('START',j['id'],flush=True)
 with (root/'experiments/extension/logs'/f"{j['id']}.cpu_priority.log").open('w') as f:
  res=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 print('END',j['id'],res.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
