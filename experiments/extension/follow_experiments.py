"""Follow the finite experiment queue and update the open manuscript's sources.
Compilation stays in the existing desktop editor; no PDF is exported here.
"""
import hashlib,json,subprocess,time,traceback,sys
from pathlib import Path
from sync_results import sync
ROOT=Path(__file__).resolve().parents[2]
last=None;start=time.monotonic()
while time.monotonic()-start<7*24*3600:
 try:
  sync();jobs=json.loads((ROOT/'experiments/extension/jobs_all.json').read_text())
  records={}
  for p in (ROOT/'experiments/extension/results').glob('*.json'):
   try:
    obj=json.loads(p.read_text())
    if obj.get('status'):records[p.stem]=obj
   except json.JSONDecodeError:pass
  groups={k:[] for k in ['complete','failed','running','not_started']}
  for job in jobs:groups[records.get(job['id'],{}).get('status','not_started')].append(job['id'])
  lines=['Experiment status (from recorded results)',f'Declared runs: {len(jobs)}',f'Completed at the requested budget: {len(groups["complete"])}',f'Failed: {len(groups["failed"])}',f'Running: {len(groups["running"])}',f'Not started: {len(groups["not_started"])}','',
   'A completed screening run covers only its screening budget. These runs do not cover every large model or dataset in the proposal.',
   'Manuscript source: paper/main.tex in the open editor',
   'This follower syncs results and tables. It does not build or export a PDF.',
   'Proposal coverage: analysis/PROPOSAL_COVERAGE.json','']
  recovery_plan=ROOT/'analysis/FAILURE_RECOVERY_PLAN.json'
  if recovery_plan.exists():
   rp=json.loads(recovery_plan.read_text());covered=sum(len(c['parents']) for c in rp['cases'] if all(records.get(k,{}).get('status')=='complete' for k in c['repaired_run_ids']));recovery_ids=[k for c in rp['cases'] for k in c['repaired_run_ids']];completed_recovery=sum(records.get(k,{}).get('status')=='complete' for k in recovery_ids)
   lines.extend([f'Failure-recovery controls: {completed_recovery}/{len(recovery_ids)} complete; covering {covered}/{rp["original_failure_count"]} original failures. Original failure records remain available.',''])
  new_jobs=[j for j in jobs if j.get('cohort')=='prospective_new_tasks']
  if new_jobs:
   lines.append('New-task study (6 tasks, 3 families, 198 fixed runs):')
   for phase,total in [('tune',108),('full',90)]:
    completed=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='complete' for j in new_jobs)
    running=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='running' for j in new_jobs)
    failed=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='failed' for j in new_jobs)
    lines.append(f'  {phase}: {completed}/{total} complete, {running} running, {failed} failed')
   lines.extend(['  All declared tasks and seeds remain in the record. Assess family-level support only after the study ends.',''])
  for key,title in [('running','Running'),('failed','Failed'),('not_started','Not started')]:
   lines.append(title+':')
   for name in groups[key]:
    r=records.get(name,{});history=r.get('history',[])
    progress=f', progress {history[-1]["epoch"] if history else 0}/{r.get("epochs","?")} {r.get("budget_unit","epochs")}, {len(history)} records'
    lines.append('  '+name+(progress if key=='running' else (', '+r.get('error','') if key=='failed' else '')))
   lines.append('')
  (ROOT/'analysis/MANUSCRIPT_STATUS.txt').write_text('\n'.join(lines))
  signature=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
  if signature!=last:
   subprocess.run(['python3',str(ROOT/'experiments/extension/build_paper.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
   last=signature
  print({k:len(v) for k,v in groups.items()},flush=True)
  if '--once' in sys.argv or (not groups['running'] and not groups['not_started']):break
 except Exception:
  print(traceback.format_exc(),flush=True)
  state_path=ROOT/'analysis/REMOTE_SYNC_STATE.json'
  if state_path.exists():
   state=json.loads(state_path.read_text())
   if state.get('status')=='unreachable':
    status_path=ROOT/'analysis/MANUSCRIPT_STATUS.txt';cached=status_path.read_text() if status_path.exists() else ''
    if cached.startswith('Remote connection failed'):
     cached='\n'.join(cached.splitlines()[3:])
    prefix='Remote connection failed. The following is the last synced status, not a live result.\nLast successful sync: '+state.get('last_successful_sync','unknown')+'\n\n'
    status_path.write_text(prefix+cached)

 time.sleep(60)
