"""Resumable queue. One process per GPU. Failed jobs remain explicitly failed."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--jobs',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];jobs=json.loads(Path(a.jobs).read_text())
for i,j in enumerate(jobs):
    out=root/'experiments/extension/results'/f"{j['id']}.json"
    if out.exists() and json.loads(out.read_text()).get('status')=='complete':continue
    args=[sys.executable,str(root/'experiments/extension/run_task.py'),'--device',f'cuda:{a.gpu}','--data',str(root/'data'),'--out',str(out)]
    for k,v in j.items():
        if k=='id':continue
        args.extend(['--'+k.replace('_','-'),str(v)])
    if j['family']=='lm':args[args.index('--data')+1]=str(root/'data')
    log=root/'experiments/extension/logs'/f"{j['id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
    print('START',j['id'],flush=True)
    with log.open('w') as f:r=subprocess.run(args,cwd=root,stdout=f,stderr=subprocess.STDOUT)
    print('FINISH',j['id'],r.returncode,flush=True)
