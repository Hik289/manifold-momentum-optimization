"""Fill missing language/tabular families while GPU ablations continue."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
jobs=[j for j in json.loads((root/('experiments/extension/'+(sys.argv[1] if len(sys.argv)>1 else 'jobs.json'))).read_text()) if j['family'] in ['lm','tabular']]
def run(j):
 out=root/f"experiments/extension/results/{j['id']}.json"
 if out.exists() and json.loads(out.read_text()).get('status')=='complete':return
 args=[sys.executable,str(root/'experiments/extension/run_task.py'),'--device','cpu','--data',str(root/'data'),'--out',str(out)]
 for k,v in j.items():
  if k!='id':args.extend(['--'+k.replace('_','-'),str(v)])
 with (root/f"experiments/extension/logs/{j['id']}_cpu.log").open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=root)
 print(j['id'],r.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
