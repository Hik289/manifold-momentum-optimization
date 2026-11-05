"""Use GPU 0 for the finite expansion only after all original arms terminate."""
import json,time,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];jobs=[j for j in json.loads((root/'experiments/extension/jobs_all.json').read_text()) if not j['id'].startswith('exp2_')]
for _ in range(7*24*60):
    statuses=[]
    for j in jobs:
        p=root/'experiments/extension/results'/f"{j['id']}.json"
        try:statuses.append(json.loads(p.read_text()).get('status'))
        except (OSError,json.JSONDecodeError):statuses.append(None)
    if all(s in ('complete','failed') for s in statuses):
        sys.exit(subprocess.call([sys.executable,'experiments/expansion/dispatch.py','--lane','gpu','--device','cuda:0'],cwd=root))
    time.sleep(60)
raise TimeoutError('Original finite queue did not terminate within seven days; GPU 0 was not claimed.')
