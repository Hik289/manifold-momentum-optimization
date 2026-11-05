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
  lines=['实验状态（自动从真实记录更新）',f'已声明实验：{len(jobs)}',f'完成完整请求预算：{len(groups["complete"])}',f'数值失败：{len(groups["failed"])}',f'运行中：{len(groups["running"])}',f'尚未启动：{len(groups["not_started"])}','',
   '注意：screen 的完整请求预算仍然只是筛选预算。这里的全部已声明实验，不等于 proposal 所有大模型/数据集已完成。',
   '当前版本：已打开的 paper/main.tex 编辑器及其内置预览',
   '本次跟随仅同步结果与论文表格；不另编译或导出 PDF。',
   '全部 proposal 缺项：analysis/PROPOSAL_COVERAGE.json','']
  recovery_plan=ROOT/'analysis/FAILURE_RECOVERY_PLAN.json'
  if recovery_plan.exists():
   rp=json.loads(recovery_plan.read_text());covered=sum(len(c['parents']) for c in rp['cases'] if all(records.get(k,{}).get('status')=='complete' for k in c['repaired_run_ids']));recovery_ids=[k for c in rp['cases'] for k in c['repaired_run_ids']];completed_recovery=sum(records.get(k,{}).get('status')=='complete' for k in recovery_ids)
   lines.extend([f'失败修复对照：{completed_recovery}/{len(recovery_ids)} 项完成；覆盖原失败记录 {covered}/{rp["original_failure_count"]} 条。原失败历史保留。',''])
  new_jobs=[j for j in jobs if j.get('cohort')=='prospective_new_tasks']
  if new_jobs:
   lines.append('新任务（6任务、3任务族、198次固定实验）:')
   for phase,total in [('tune',108),('full',90)]:
    completed=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='complete' for j in new_jobs)
    running=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='running' for j in new_jobs)
    failed=sum(j.get('phase')==phase and records.get(j['id'],{}).get('status')=='failed' for j in new_jobs)
    lines.append(f'  {phase}: 完成 {completed}/{total}，运行 {running}，失败 {failed}')
   lines.extend(['  所有任务和种子保留；完整研究结束后按冻结规则判定优势任务族。',''])
  for key,title in [('running','运行中'),('failed','失败'),('not_started','待启动')]:
   lines.append(title+':')
   for name in groups[key]:
    r=records.get(name,{});history=r.get('history',[])
    progress=f'，最后进度 {history[-1]["epoch"] if history else 0}/{r.get("epochs","?")} {r.get("budget_unit","epochs")}，{len(history)} 条训练记录'
    lines.append('  '+name+(progress if key=='running' else ('，'+r.get('error','') if key=='failed' else '')))
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
    if cached.startswith('服务器连接失败'):
     cached='\n'.join(cached.splitlines()[3:])
    prefix='服务器连接失败：以下为最近成功同步的缓存，不代表实时运行状态。\n上次成功同步：'+state.get('last_successful_sync','unknown')+'\n\n'
    status_path.write_text(prefix+cached)

 time.sleep(60)
