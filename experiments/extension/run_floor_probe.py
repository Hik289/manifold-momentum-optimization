import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
for smin in ['.1','.001','.00001']:
 for method in ['euclidean','bures']:
  name=f'floor_{smin}_{method}'
  with (root/f'experiments/extension/logs/{name}.log').open('w') as f:
   subprocess.run([sys.executable,str(root/'experiments/extension/run_task.py'),'--family','graph','--task','Cora','--method',method,'--epochs','1000','--lr','.01','--smin',smin,'--device','cpu','--data',str(root/'data'),'--out',str(root/f'experiments/extension/results/{name}.json')],stdout=f,stderr=subprocess.STDOUT,cwd=root)
