import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
(root/'experiments/extension/logs').mkdir(parents=True,exist_ok=True)
for task in ['poisson','burgers']:
 for method in ['adamw','euclidean','bures']:
  out=root/f'experiments/extension/results/sciml_{task}_{method}_full.json'
  with (root/f'experiments/extension/logs/sciml_{task}_{method}_full.log').open('w') as f:
   subprocess.run([sys.executable,str(root/'experiments/extension/run_task.py'),'--family','sciml','--task',task,'--method',method,'--epochs','2000','--lr','.001','--diag-every','20','--device','cpu','--data',str(root/'data'),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,cwd=root)
