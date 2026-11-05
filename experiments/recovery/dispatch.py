"""Four low-priority CPU workers for the 18 declared failure-recovery controls."""
import concurrent.futures,fcntl,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];lock=(root/'experiments/recovery/.dispatch.lock').open('w')
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)
jobs=json.loads((root/'experiments/recovery/jobs.json').read_text())
def run(j):
 out=root/'experiments/extension/results'/f"{j['id']}.json"
 if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed','running'):return
 path=root/'experiments/recovery/job_files'/f"{j['id']}.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(j))
 data=str(root/'data')
 print('START',j['id'],flush=True)
 with (root/'experiments/extension/logs'/f"{j['id']}.log").open('w') as f:
  x=subprocess.run([sys.executable,str(root/'experiments/recovery/run_recovery.py'),'--job',str(path),'--data',data],cwd=root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 print('END',j['id'],x.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
