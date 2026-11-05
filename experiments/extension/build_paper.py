"""Generate paper tables/figures ONLY from completed run records."""
import json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper';RESULTS=ROOT/'experiments/extension/results'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
records={};failed_records={}
for p in RESULTS.glob('*.json'):
 try:d=json.loads(p.read_text())
 except Exception:continue
 if d.get('status')=='complete':records[p.stem]=d
 if d.get('status')=='failed':failed_records[p.stem]=d
names={'adamw':'AdamW','euclidean':'Euclidean-Kron.','bures':'BuresMomentum','shampoo':'Shampoo ref.','scale_only':'Scale-only'}
colors={'adamw':'#72757e','euclidean':'#2f68a3','bures':'#15803d','shampoo':'#609879','scale_only':'#a977ad'}
fig,ax=plt.subplots(1,3,figsize=(10,2.7),sharey=True)
for axis,task in zip(ax,['Cora','CiteSeer','PubMed']):
 for method in names:
  r=records.get(f'graph_{task}_{method}_full')
  if r:axis.plot([x['epoch'] for x in r['history']],[100*x['val_metric'] for x in r['history']],label=names[method],color=colors[method],lw=1)
 axis.set(title=task,xlabel='Epoch',ylim=(0,90));axis.grid(alpha=.16)
ax[0].set_ylabel('Validation accuracy (%)');ax[1].legend(fontsize=6,loc='lower right');fig.tight_layout();fig.savefig(PAPER/'figures/graph_trajectories.pdf');plt.close(fig)
lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',r'Dataset & Method & Final test (\%) & Val-selected test (\%) & Selected epoch & Seconds \\',r'\midrule']
for task in ['Cora','CiteSeer','PubMed']:
 for method in names:
  r=records.get(f'graph_{task}_{method}_full')
  if r:
   b=r['validation_selected'];f=r['final'];lines.append(f"{task} & {names[method]} & {100*f['test_metric']:.2f} & {100*b['test_metric']:.2f} & {b['epoch']} & {r['wall_seconds']:.1f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Independent-tile schema-2 graph runs, seed 42, 1000 epochs, common learning rate 0.01. The selected epoch maximizes validation accuracy; the test set is evaluated only after training at the final and selected checkpoints. Seconds are descriptive shared-machine measurements, not a controlled throughput benchmark. Shampoo is the unblocked reference implementation.}',r'\label{tab:graphs}',r'\end{table*}']
(PAPER/'sections/graph_table.tex').write_text('\n'.join(lines))
abl=[]
for k,r in records.items():
 if k.startswith('ablation_'):abl.append((k.removeprefix('ablation_') if hasattr(k,'removeprefix') else k[9:],r))
lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Configuration & Final test (\%) & Selected test (\%) & Selected epoch & Final val (\%) & Seconds \\',r'\midrule']
full=records.get('graph_Cora_bures_full')
if full:abl=[('Full Bures',full)]+sorted(abl)
for k,r in abl:
 b=r['validation_selected'];lines.append(f"{k.replace('_',' ')} & {100*r['final']['test_metric']:.2f} & {100*b['test_metric']:.2f} & {b['epoch']} & {100*r['final']['val_metric']:.2f} & {r['wall_seconds']:.1f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Completed Cora ablations. Every configuration runs for 1000 epochs with seed 42. Left/right-only force the opposite factor to identity; identity forces both. Diagonal retains coordinate diagonals, permute reassigns state eigenvectors, and rotate uses independent orthogonal frames. No momentum applies $\beta_1=0$ also to nonmatrix fallback coordinates.}',r'\label{tab:ablations}',r'\end{table*}']
(PAPER/'sections/ablation_table.tex').write_text('\n'.join(lines))
other=[(k,v) for k,v in records.items() if v.get('family') in ['lm','tabular','shift','sciml','vision'] and v.get('cohort') not in ('flow_expansion','failure_recovery')]
lines=[r'\begin{longtable}{llrrrrl}',r'\caption{Completed additional tasks. LM entries are byte-level small-model screening and are not GPT-2 benchmarks. Regression/PDE metrics are lower-is-better; classification accuracy is higher-is-better.}\\',r'\toprule',r'Task & Method & LR & Budget & Final test & Selected test & Stage \\',r'\midrule',r'\endfirsthead',r'\toprule Task & Method & LR & Budget & Final test & Selected test & Stage \\ \midrule',r'\endhead']
for k,r in sorted(other):
 lines.append(f"{r['task'].replace('_',' ')} & {names.get(r['method'],r['method'])} & {r['lr']:.2g} & {r['epochs']} & {r['final']['test_metric']:.4g} & {r['validation_selected']['test_metric']:.4g} & {r['stage']} "+r'\\')
if not other:lines.append(r'\multicolumn{7}{l}{No additional run has completed in this snapshot.}\\')
lines.extend([r'\bottomrule',r'\end{longtable}'])
(PAPER/'sections/additional_table.tex').write_text('\n'.join(lines))
fail_lines=[r'\begin{longtable}{llrrp{.40\textwidth}}',r'\caption{Failed new runs. A failure is retained and is never counted as a completed performance result.}\\',r'\toprule Task & Method & LR & Last epoch & Reason \\ \midrule']
for k,r in sorted(failed_records.items()):
    error=r.get('error','unspecified')
    if 'nonfinite validation' in error:error='Nonfinite validation loss'
    elif 'nonfinite loss at' in error:error='Nonfinite training loss'
    elif 'non-finite values' in error:error='Nonfinite matrix in decomposition'
    elif 'linalg.svd' in error:error='Matrix decomposition failed'
    else:error=error.replace('_',' ').replace('%',r'\%').replace('&',r'\&')[:70]
    lr_text=f"{r['lr']:.2g}" if isinstance(r.get('lr'),(int,float)) else '---'
    task_text=r.get('task','').replace('_',' ')
    fail_lines.append(f"{task_text} & {r.get('method','')} & {lr_text} & {len(r.get('history',[]))} & {error} "+r'\\')
if not failed_records:fail_lines.append(r'\multicolumn{5}{l}{No failed record in this snapshot.}\\')
fail_lines.extend([r'\bottomrule',r'\end{longtable}'])
(PAPER/'sections/failure_table.tex').write_text('\n'.join(fail_lines))
# State-scale diagnostics measure actual optimizer quantities, independent of target interpretation.
fig,axes=plt.subplots(1,2,figsize=(8,2.8))
for method in ['euclidean','bures','scale_only']:
 r=records.get(f'graph_Cora_{method}_full')
 if r:
  rows=r['diagnostics'];xx=[v['step'] for v in rows]
  axes[0].semilogy(xx,[v.get('lambda_max_L',float('nan')) for v in rows],label=names[method],color=colors[method])
  axes[1].semilogy(xx,[v.get('scale_mean',float('nan')) for v in rows],label=names[method],color=colors[method])
axes[0].set(xlabel='Epoch',ylabel='Mean max eigenvalue, left states');axes[1].set(xlabel='Epoch',ylabel='Mean matching scale');axes[1].axhline(.1,ls='--',color='gray',lw=.7);axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(PAPER/'figures/state_scale.pdf');plt.close(fig)
# Synthetic metrics are generated independently, not inferred from training curves.
syn=json.loads((RESULTS/'synthetic_complete.json').read_text())
cur=np.load(RESULTS/'synthetic_complete_curves.npz')
fig,axs=plt.subplots(2,3,figsize=(10,5))
for ax,setting in zip(axs.flat,['static','scale','rotation','geodesic','jump','rotscale']):
 for method in ['euclidean','bures','log_euclidean','ai']:
  arr=cur[f'{setting}_{method}'];ax.plot(arr[:,0],arr[:,1],label=method,lw=1)
 ax.set(title=setting,xlabel='Update',ylabel='BW target error');ax.grid(alpha=.16)
axs[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(PAPER/'figures/synthetic_trajectories.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7,2.7))
noise=syn['scalar_noise_scaling'];axs[0].loglog([r['alpha'] for r in noise],[r['empirical_mse'] for r in noise],'o-',label='400 independent filters');axs[0].loglog([r['alpha'] for r in noise],[r['exact_mse'] for r in noise],'--',label='Exact alpha/(2-alpha)');axs[0].set(xlabel='Smoothing alpha',ylabel='Stationary MSE');axs[0].legend(fontsize=7)
grid=syn['dynamic_grid']
for alpha in [.01,.03,.1]:
 rows=[r for r in grid if r['alpha']==alpha];axs[1].plot([r['drift'] for r in rows],[r['mse'] for r in rows],'o-',label=f'alpha={alpha}')
axs[1].set(xlabel='Target drift per update',ylabel='Mean square tracking error',yscale='log');axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(PAPER/'figures/noise_drift.pdf');plt.close(fig)
lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Setting & Euclidean BW error & Bures BW error & Euclidean root error & Bures root error \\',r'\midrule']
for st in ['static','scale','rotation','geodesic','jump','rotscale']:
 e,b=syn[st+'_euclidean'],syn[st+'_bures'];lines.append(f"{st} & {e['mean_BW']:.5f} & {b['mean_BW']:.5f} & {e['mean_root_error']:.5f} & {b['mean_root_error']:.5f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{New synthetic checks: $d=8$, 1000 updates, $\alpha=0.01$, identical identity initialization. Static error is against an offline finite-population BW barycenter. Other targets are noiseless known paths. Geodesic observations alone do not ensure an on-geodesic initialization. BW error and inverse-root error can rank methods differently.}',r'\label{tab:synthetic}',r'\end{table*}']
(PAPER/'sections/synthetic_table.tex').write_text('\n'.join(lines))
lines=[r'\begin{longtable}{rlrrr}',r'\caption{CPU matching-floor intervention on Cora, 1000 epochs, common learning rate 0.01. Only completed arms are shown.}\\',r'\toprule Floor & Method & Final test (\%) & Selected test (\%) & Selected epoch \\ \midrule']
for key,r in sorted(records.items()):
 if key.startswith('floor_'):
  b=r['validation_selected'];lines.append(f"{r['smin']:.3g} & {names[r['method']]} & {100*r['final']['test_metric']:.2f} & {100*b['test_metric']:.2f} & {b['epoch']} "+r'\\')
lines.extend([r'\bottomrule',r'\end{longtable}']);(PAPER/'sections/floor_table.tex').write_text('\n'.join(lines))
grid_jobs=json.loads((ROOT/'experiments/extension/jobs_lm_lrgrid.json').read_text())
grid_terminal=sum(j['id'] in records or j['id'] in failed_records for j in grid_jobs)
grid_caption=('All declared trials have terminated. Validation-only selection among runs completing 1000 updates is reported below; numerical failures remain listed separately.' if grid_terminal==len(grid_jobs) else 'This remains a partial grid until all declared trials terminate.')
lines=[r'\begin{longtable}{lrrrr}',r'\caption{WikiText excerpt rate trials: the same small byte-level model and 1000 requested updates per trial. '+grid_caption+r'}\\',r'\toprule Method & LR & Final test byte PPL & Selected test byte PPL & Best validation byte PPL \\ \midrule']
for key,r in sorted(records.items()):
 if key.startswith('lm_lrgrid_'):
  b=r['validation_selected'];lines.append(f"{names[r['method']]} & {r['lr']:.2g} & {r['final']['test_metric']:.3f} & {b['test_metric']:.3f} & {b['val_metric']:.3f} "+r'\\')
lines.extend([r'\bottomrule',r'\end{longtable}']);(PAPER/'sections/lm_grid_table.tex').write_text('\n'.join(lines))

lines=[r'\begin{longtable}{lrrrr}',r'\caption{Validation-selected learning rates in the completed six-rate WikiText excerpt grid. Selection uses the best validation checkpoint among full-budget runs only. Two numerically failed Bures trials are ineligible and are retained in the failure table.}\\',r'\toprule Method & Selected LR & Selected epoch & Validation byte PPL & Test byte PPL \\ \midrule']
if grid_terminal==len(grid_jobs):
 for method in ['adamw','euclidean','bures']:
  candidates=[r for k,r in records.items() if k.startswith('lm_lrgrid_') and r['method']==method]
  if candidates:
   r=min(candidates,key=lambda r:r['validation_selected']['val_metric']);v=r['validation_selected']
   lines.append(f"{names[method]} & {r['lr']:.2g} & {v['epoch']} & {v['val_metric']:.3f} & {v['test_metric']:.3f} "+r'\\')
else:lines.append(r'\multicolumn{5}{l}{Grid selection pending all declared trials.}\\')
lines.extend([r'\bottomrule',r'\end{longtable}']);(PAPER/'sections/lm_selected_table.tex').write_text('\n'.join(lines))

# Completion ledger: all unfinished jobs stay unfinished, irrespective of partial histories.
jobs=json.loads((ROOT/'experiments/extension/jobs_all.json').read_text());jobids={j['id'] for j in jobs}
completed=[j for j in jobs if j['id'] in records]
coverage=dict(total_queued=len(jobs),completed=len(completed),completed_by_family={f:sum(j['family']==f for j in completed) for f in sorted({j['family'] for j in jobs})},failed_runs=len(failed_records),new_full_completed=sum(r.get('stage')=='full' for r in records.values()),new_screen_completed=sum(r.get('stage')=='screen' for r in records.values()),sciml_completed=sum(r.get('family')=='sciml' for r in records.values()),pending=[j['id'] for j in jobs if j['id'] not in records])
(ROOT/'analysis/extension_status.json').write_text(json.dumps(coverage,indent=2))
(PAPER/'sections/completion_snapshot.tex').write_text(f"This snapshot includes {len(records)} completed new training runs: {coverage['new_full_completed']} at their declared full budget and {coverage['new_screen_completed']} at screening scope. Of the {len(jobs)} declared extension jobs, {len(completed)} are complete. Failures remain incomplete budget outcomes and are reported separately. A running, failed, or absent record is never used as a completed result. The queue covers representative gaps; it does not exhaust all large-model and dataset combinations in the proposal.\n")
sys.path.insert(0,str(ROOT/'experiments'))
from expansion.build_figures import build as build_loss_figures
build_loss_figures()
from expansion.build_evidence import build as build_completed_evidence
build_completed_evidence()
from expansion.build_core_losses import build as build_core_losses
build_core_losses()
from recovery.build_recovery import build as build_recovery
build_recovery()
from new_tasks.build_report import build as build_new_tasks
build_new_tasks()
from expansion.build_vision_positive import build as build_vision_positive
build_vision_positive()
from extension.restyle_paper import build as restyle_paper
restyle_paper()
print(json.dumps(coverage,indent=2))
