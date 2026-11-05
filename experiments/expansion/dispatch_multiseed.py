"""Finite seed-43/44 follow-up; every declared method and seed is retained."""
import concurrent.futures,fcntl,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
lock=(root/'experiments/expansion/.multiseed-dispatch.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)
jobs=json.loads((root/'experiments/expansion/multiseed_jobs.json').read_text())
def run(j):
 out=root/'experiments/extension/results'/f"{j['id']}.json"
 if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed'):return
 path=root/'experiments/expansion/job_files'/f"{j['id']}.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(j))
 cmd=[sys.executable,str(root/'experiments/expansion/run_multiseed.py'),'--job',str(path),'--device','cpu','--data',str(root/'data'),'--suite-data',str(root/'data/expansion')]
 print('START',j['id'],flush=True)
 with (root/'experiments/extension/logs'/f"{j['id']}.log").open('w') as f:
  res=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 print('END',j['id'],res.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
