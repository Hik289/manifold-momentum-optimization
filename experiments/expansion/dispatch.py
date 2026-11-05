"""Bounded shared-machine worker pool for the declared architecture expansion."""
import argparse,concurrent.futures,json,subprocess,sys,os
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--lane',choices=['cpu','gpu'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='cuda:1');p.add_argument('--family',choices=['lm','graph']);a=p.parse_args()
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
jobs=json.loads((root/'experiments/expansion/jobs.json').read_text());jobs=[j for j in jobs if (j['family']=='vision')==(a.lane=='gpu')]
if a.family:jobs=[j for j in jobs if j['family']==a.family]
def run(j):
    out=root/'experiments/extension/results'/f"{j['id']}.json"
    if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed'):return
    path=root/'experiments/expansion/job_files'/f"{j['id']}.json";path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(j))
    cmd=[sys.executable,str(root/'experiments/expansion/run_benchmark.py'),'--job',str(path),'--device',a.device if a.lane=='gpu' else 'cpu','--data',str(root/'data'),'--suite-data',str(root/'data/expansion')]
    log=root/'experiments/extension/logs'/f"{j['id']}.log"
    print('START',j['id'],flush=True)
    with log.open('w') as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    print('END',j['id'],r.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers if a.lane=='cpu' else 1) as pool:list(pool.map(run,jobs))
