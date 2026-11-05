"""Recovery lineage, budgets and matched comparisons; historical failures remain failed."""
import collections,hashlib,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];SEC=ROOT/'paper/sections';FIG=ROOT/'paper/figures';METHODS=['adamw','euclidean','bures'];COL={'adamw':'#667085','euclidean':'#3977AA','bures':'#15803d'};NAME={'adamw':'AdamW','euclidean':'Euclidean','bures':'Bures'}
def build():
 plan=json.loads((ROOT/'analysis/FAILURE_RECOVERY_PLAN.json').read_text());jobs=json.loads((ROOT/'experiments/recovery/jobs.json').read_text());records={p.stem:json.loads(p.read_text()) for p in (ROOT/'experiments/extension/results').glob('*.json')};out=[];parents_covered=[];curves=[]
 lines=[r'\subsection{Stability recovery of the originally failed trials}',r'\label{sec:failure_recovery}',r'The eight historical failures correspond to six unique configurations: the default and grid WikiText rate-$10^{-3}$ trials duplicate one configuration, and the two California matrix-method failures share another. We declare an 18-run recovery cohort with AdamW, Euclidean, and Bures at every configuration. Model, data, initialization, learning rate, budget, block size, and root interval are retained. The common changes are global gradient clipping at norm 1, a zero lower matching bound for matrix methods, and FP64 matrix statistics/roots. Runs restart from the original initialization; original failures remain recorded. This separately labeled cohort tests whether these safeguards complete the requested budget, without replacing the original fixed-protocol outcomes.']
 table=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llrrrrr}\toprule Recovery task / metric & LR & Budget & Complete & AdamW & Euclidean & Bures \\ \midrule']
 for case in plan['cases']:
  arms={j['method']:records.get(j['id'],{}) for j in jobs if j['case']==case['case']};done=all(r.get('status')=='complete' for r in arms.values());completed=sum(r.get('status')=='complete' for r in arms.values());cfg=case['original_config'];vals={}
  if done:
   rs=list(arms.values());assert len({r['initialization_sha256'] for r in rs})==1 and len({r['source_sha256'] for r in rs})==1
   assert all(r['data_meta']==rs[0]['data_meta'] and r['epochs']==cfg['epochs'] and r['lr']==cfg['lr'] for r in rs)
   for r in rs:
    assert r['history'][-1]['epoch']==r['epochs'] and all(math.isfinite(x['train_loss']) and math.isfinite(x['val_loss']) for x in r['history'])
   vals={m:r['validation_selected']['test_metric'] for m,r in arms.items()};parents_covered.extend(case['parents'])
  label=cfg['task'].replace('_','-')+(' / byte PPL' if cfg['family']=='lm' else ' / RMSE' if cfg['task']=='california' else ' / acc. (\\%)')
  show=lambda m: ('---' if not done else f'{100*vals[m]:.2f}' if cfg['task']=='covertype' else f'{vals[m]:.3f}')
  table.append(f"{label} & {cfg['lr']:.3g} & {cfg['epochs']} & {completed}/3 & {show('adamw')} & {show('euclidean')} & {show('bures')} "+r'\\')
  out.append(dict(case=case['case'],parents=case['parents'],complete_triplet=done,completed_methods=completed,statuses={m:r.get('status','queued') for m,r in arms.items()},selected_test=vals,runs=case['repaired_run_ids']))
 table += [r'\bottomrule\end{tabular}',r'\caption{Separately labeled stability-recovery cohort, seed 42. A row reports performance only after all three methods finish their original requested budget. Byte perplexity and California RMSE are lower-is-better; Covertype accuracy is higher-is-better. LM budgets are updates and retain the screening scope; tabular budgets are epochs. All methods share the declared recovery safeguards.}',r'\label{tab:failure_recovery}\end{table*}']
 lines.append(f"This snapshot completes {sum(r.get('status')=='complete' for a in out for r in [records.get(k,{}) for k in a['runs']])}/18 recovery runs; {len(parents_covered)}/8 historical failed records have a complete three-method recovery comparison.")
 lines.append(r'The recovery cohort establishes numerical completion, not predictive superiority: none of its six fixed-rate cases improves the selected test metric over both references. Full metrics and loss traces are retained in Table~\ref{tab:failure_recovery} and Figure~\ref{fig:failure_recovery_losses}.')
 (SEC/'recovery_summary.tex').write_text('\n'.join(lines)+'\n')
 (SEC/'recovery_table.tex').write_text('\n'.join(table).replace(r'\begin{table*}[t]',r'\begin{table}[htbp]').replace(r'\end{table*}',r'\end{table}')+'\n')
 # Full loss traces, including all observed incomplete arms, receive explicit status labels.
 fig,axes=plt.subplots(2,3,figsize=(9,5));plt.rcParams.update({'pdf.fonttype':42})
 for ax,case in zip(axes.flat,plan['cases']):
  cfg=case['original_config']
  for job in [j for j in jobs if j['case']==case['case']]:
   r=records.get(job['id'],{});hist=[x for x in r.get('history',[]) if isinstance(x.get('val_loss'),(int,float)) and math.isfinite(x['val_loss'])]
   if not hist:continue
   status=r['status'];method=job['method'];ax.plot([x['epoch'] for x in hist],[x['val_loss'] for x in hist],color=COL[method],lw=.95,ls='-' if status=='complete' else '--' if status=='failed' else ':',label=NAME[method]+('' if status=='complete' else ' '+status));curves.append(dict(run=job['id'],status=status,field='val_loss',points=len(hist),last=hist[-1]['epoch']))
  ax.set(title=cfg['task']+f" / {cfg['lr']:g}",xlabel='Update' if cfg['family']=='lm' else 'Epoch',ylabel='Validation loss',yscale='log');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=6);ax.tick_params(labelsize=7)
 fig.tight_layout();fig.savefig(FIG/'failure_recovery_losses.pdf',bbox_inches='tight');fig.savefig(FIG/'failure_recovery_losses.png',dpi=170,bbox_inches='tight');plt.close(fig)
 result=dict(original_failure_count=8,unique_cases=6,declared_recovery_runs=18,complete_recovery_runs=sum(records.get(j['id'],{}).get('status')=='complete' for j in jobs),historical_failures_with_complete_recovery=len(parents_covered),historical_failures_covered=parents_covered,cases=out,curves=curves)
 coverage_path=ROOT/'analysis/PROPOSAL_COVERAGE.json'
 if coverage_path.exists():
  coverage=json.loads(coverage_path.read_text());coverage['failure_recovery']={k:result[k] for k in ['original_failure_count','unique_cases','declared_recovery_runs','complete_recovery_runs','historical_failures_with_complete_recovery']};coverage['failure_recovery']['protocol']='gradient clip 1; zero lower matching bound; FP64 matrix states; original configuration/rate/budget retained; distinct recovery cohort';coverage_path.write_text(json.dumps(coverage,indent=2)+'\n')
 (ROOT/'analysis/RECOVERY_PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['complete_recovery_runs','historical_failures_with_complete_recovery']});return result
if __name__=='__main__':build()
