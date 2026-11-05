"""Two-update execution checks stored outside the performance-result directory."""
import json,subprocess,sys,os
from pathlib import Path
root=Path(__file__).resolve().parents[2];jobs=json.loads((root/'experiments/expansion/jobs.json').read_text());summary=[]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
for family,task,model,hidden in [('graph','Cora','gat',64),('graph','Cora','sage',64),('graph','Cora','gcnii',64),('lm','wikitext2','lstm',64),('lm','wikitext2','gru',128),('lm','wikitext2','transformer',64)]:
    j=next(j for j in jobs if (j['family'],j['task'],j['model'],j['hidden'],j['method'])==(family,task,model,hidden,'bures'))
    j={**j,'epochs':2,'eval_every':1,'diag_every':1};folder=root/'experiments/expansion/smoke';folder.mkdir(exist_ok=True,parents=True)
    path=folder/f'{model}{hidden}_job.json';out=folder/f'{model}{hidden}_result.json';path.write_text(json.dumps(j))
    with (folder/f'{model}{hidden}.log').open('w') as log:
        r=subprocess.run([sys.executable,str(root/'experiments/expansion/run_benchmark.py'),'--job',str(path),'--out',str(out),'--device','cpu','--data',str(root/'data'),'--suite-data',str(root/'data/expansion')],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    record=json.loads(out.read_text());summary.append(dict(family=family,model=model,hidden=hidden,status=record['status'],returncode=r.returncode,output=str(out),error=record.get('error')))
    (root/'experiments/extension/logs/expansion_validation.json').write_text(json.dumps(summary,indent=2));print(summary[-1],flush=True)
assert all(r['status']=='complete' for r in summary)
