"""All declared prospective tasks stay visible; no family claim before the fixed study ends."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[2];SECTIONS=ROOT/'paper/sections';METHODS=['adamw','euclidean','bures'];NAMES={'adamw':'AdamW','euclidean':'Euclidean-Kron.','bures':'BuresMomentum'}
TASKS={'etth1':'ETTh1','etth2':'ETTh2','uci_har':'UCI HAR','forda':'FordA','digits_denoise':'Digits denoising','olivetti_denoise':'Olivetti denoising'}
FAMILIES={'forecasting':'Forecasting','sequence_classification':'Sequence classification','denoising':'Denoising reconstruction'}

def paired_summary(bures,reference,direction):
 b=np.asarray(bures,dtype=float);r=np.asarray(reference,dtype=float);assert b.shape==r.shape==(5,) and np.isfinite(b).all() and np.isfinite(r).all()
 d=b-r if direction=='max' else r-b;mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(5);q=float(student_t.ppf(.975,4));p=float(student_t.sf(mean/se,4)) if se else (0. if mean>0 else (1. if mean<0 else .5))
 return {'paired_differences':d.tolist(),'mean_gain':mean,'ci95':[mean-q*se,mean+q*se],'one_sided_p':p,'relative_mean_gain':mean/float(r.mean()) if r.mean()!=0 else None}

def holm_adjust(pvalues):
 ordered=sorted(pvalues,key=pvalues.get);out={};last=0.
 for k,key in enumerate(ordered):last=max(last,min(1.,(len(ordered)-k)*pvalues[key]));out[key]=last
 return out

def table(header,rows,caption,label,long=False):
 cols='l'+'r'*(len(header)-1)
 if long:return '\n'.join([r'\begin{longtable}{'+cols+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',' & '.join(header)+r' \\ \midrule',r'\endfirsthead',r'\toprule',' & '.join(header)+r' \\ \midrule',r'\endhead']+rows+[r'\bottomrule',r'\end{longtable}'])
 return '\n'.join([r'\begin{table*}[t]\centering\small',r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(header)+r' \\ \midrule']+rows+[r'\bottomrule\end{tabular}',r'\caption{'+caption+r'}\label{'+label+'}',r'\end{table*}'])

def make_losses(records,jobs,tasks,path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 tasks=[task for task in tasks if any(d.get('task')==task and d.get('phase')=='full' and d.get('history') for d in records.values())]
 cols=min(3,len(tasks));blocks=math.ceil(len(tasks)/cols);fig,axs=plt.subplots(2*blocks,cols,figsize=(3.5*cols,4.0*blocks),squeeze=False)
 colors={'adamw':'#747983','euclidean':'#3166a2','bures':'#15803d'};used=[]
 for k,task in enumerate(tasks):
  r,c=2*(k//cols),k%cols
  for m in METHODS:
   runs=[records[j['id']] for j in jobs if j['phase']=='full' and j['task']==task and j['method']==m and j['id'] in records and records[j['id']].get('history')]
   complete=[rr for rr in runs if rr['status']=='complete']
   for index,rr in enumerate(runs):
    used.append(rr['id'])
    for ax,key in [(axs[r,c],'train_loss'),(axs[r+1,c],'val_loss')]:
     h=[v for v in rr['history'] if isinstance(v.get(key),(int,float)) and math.isfinite(v[key])];ax.plot([v['epoch'] for v in h],[v[key] for v in h],color=colors[m],lw=.5,alpha=.35,ls='--' if rr['status']=='failed' else ':' if rr['status']=='running' else '-',label=NAMES[m]+f' ({len(complete)}/5 complete)' if index==0 and len(complete)!=5 else None)
   complete=[rr for rr in runs if rr['status']=='complete']
   if len(complete)==5:
    for ax,key in [(axs[r,c],'train_loss'),(axs[r+1,c],'val_loss')]:
     maps=[{v['epoch']:v[key] for v in rr['history'] if isinstance(v.get(key),(int,float)) and math.isfinite(v[key])} for rr in complete];xx=sorted(set.intersection(*(set(v) for v in maps)));yy=[np.mean([v[x] for v in maps]) for x in xx];ax.plot(xx,yy,color=colors[m],lw=1.2,label=NAMES[m])
  for ax,key in [(axs[r,c],'Training loss'),(axs[r+1,c],'Validation loss')]:ax.set(title=TASKS[task]+' / '+key.lower(),xlabel='Optimizer update',ylabel='Cross-entropy' if task in ['uci_har','forda'] else 'MSE');ax.grid(alpha=.15)
  if not any(d.get('task')==task and d.get('phase')=='full' and d.get('history') for d in records.values()):
   for ax in [axs[r,c],axs[r+1,c]]:
    ax.set_axis_off();ax.text(.5,.5,'No recorded full-run trajectory',transform=ax.transAxes,ha='center',va='center',color='#747983',fontsize=8)
  handles,labels=axs[r,c].get_legend_handles_labels()
  if handles:axs[r,c].legend(fontsize=6)
 for k in range(len(tasks),blocks*cols):axs[2*(k//cols),k%cols].axis('off');axs[2*(k//cols)+1,k%cols].axis('off')
 fig.tight_layout();fig.savefig(path);fig.savefig(path.with_suffix('.png'),dpi=170);plt.close(fig)
 return sorted(set(used))

def build():
 protocol=ROOT/'experiments/new_tasks/protocol.json';p=json.loads(protocol.read_text());sha=hashlib.sha256(protocol.read_bytes()).hexdigest();assert sha==(protocol.with_suffix('.sha256')).read_text().strip();jobs=json.loads((ROOT/'experiments/new_tasks/jobs.json').read_text());records={}
 for j in jobs:
  q=ROOT/'experiments/extension/results'/f"{j['id']}.json"
  if q.exists():
   d=json.loads(q.read_text());assert not d.get('smoke');assert d['id']==j['id']
   if d['status']=='complete':
    assert d['history'][-1]['epoch']==j['epochs'] and len(d['history'])==j['epochs'] and d.get('protocol_sha256')==sha
    assert (d.get('test_evaluations')==0 and not d.get('test_data_loaded')) if j['phase']=='tune' else (d.get('test_evaluations')==2 and d.get('test_data_loaded'))
   records[j['id']]=d
 counts={phase:{s:sum(j['phase']==phase and records.get(j['id'],{}).get('status','queued')==s for j in jobs) for s in ['complete','running','failed','queued']} for phase in ['tune','full']}
 full_jobs=[j for j in jobs if j['phase']=='full'];final=all(records.get(j['id'],{}).get('status') in ('complete','failed') for j in full_jobs)
 task_results={};rows=[];contrast_rows=[]
 for task in p['tasks']:
  name=task['task'];arms={m:[records.get(next(j['id'] for j in full_jobs if j['task']==name and j['method']==m and j['seed']==seed),{}) for seed in p['full_seeds']] for m in METHODS};ready=all(d.get('status')=='complete' for rr in arms.values() for d in rr);tr={'all_15_complete':ready,'methods':{},'contrasts':{}}
  for m,rr in arms.items():
   complete=sum(d.get('status')=='complete' for d in rr);failed=sum(d.get('status')=='failed' for d in rr);metric='pending';final_metric='pending';lr='pending'
   if rr[0].get('lr') is not None:lr=f"{rr[0]['lr']:.2g}"
   if ready:
    values=[d['validation_selected']['test_metric'] for d in rr];fv=[d['final']['test_metric'] for d in rr];assert all(math.isfinite(v) for v in values+fv);tr['methods'][m]={'seeds':p['full_seeds'],'selected_test':values,'final_test':fv,'lr':rr[0]['lr']};scale=100 if task['direction']=='max' else 1;metric=f"{scale*np.mean(values):.4g} $\\pm$ {scale*np.std(values,ddof=1):.2g}";final_metric=f"{scale*np.mean(fv):.4g} $\\pm$ {scale*np.std(fv,ddof=1):.2g}"
   rows.append(f"{TASKS[name]} & {NAMES[m]} & {lr} & {complete}/5; {failed} fail & {metric} & {final_metric} "+r'\\')
  if ready:
   for seed_index in range(5):
    for field in ['initialization_sha256','source_sha256','runner_sha256','models_sha256','data_loader_sha256','data_sha256','protocol_sha256']:
     assert len({arms[m][seed_index][field] for m in METHODS})==1,('pairing mismatch',name,seed_index,field)
   for ref in ['euclidean','adamw']:
    c=paired_summary(tr['methods']['bures']['selected_test'],tr['methods'][ref]['selected_test'],task['direction']);threshold=c['mean_gain']>=.005 if task['direction']=='max' else c['relative_mean_gain'] is not None and c['relative_mean_gain']>=.01;c['practical_threshold_passed']=bool(threshold);tr['contrasts'][ref]=c;scale=100 if task['direction']=='max' else 1;lo,hi=c['ci95'];contrast_rows.append(f"{TASKS[name]} & {NAMES[ref]} & {scale*c['mean_gain']:.4g} & [{scale*lo:.4g}, {scale*hi:.4g}] & {c['one_sided_p']:.3g} "+r'\\')
  task_results[name]=tr
 families={}
 for f in FAMILIES:
  ts=[t['task'] for t in p['tasks'] if t['family']==f];ready=all(task_results[t]['all_15_complete'] for t in ts);contrasts=[c for t in ts for c in task_results[t]['contrasts'].values()];families[f]={'tasks':ts,'all_30_complete':ready,'family_p':max(c['one_sided_p'] for c in contrasts) if ready else 1.,'all_intervals_positive':ready and all(c['ci95'][0]>0 for c in contrasts),'all_practical_thresholds':ready and all(c['practical_threshold_passed'] for c in contrasts)}
 adjusted=holm_adjust({k:v['family_p'] for k,v in families.items()});supported=[];family_rows=[]
 for f,d in families.items():
  d['holm_adjusted_p']=adjusted[f] if final else None;d['supported']=bool(final and d['all_30_complete'] and d['all_intervals_positive'] and d['all_practical_thresholds'] and adjusted[f]<=.05);d['status']='supported' if d['supported'] else ('mixed/unsupported' if final else 'pending fixed-study completion')
  if d['supported']:supported.append(f)
  family_rows.append(FAMILIES[f]+' & '+('30/30' if d['all_30_complete'] else 'incomplete')+' & '+(f"{adjusted[f]:.3g}" if final else 'pending')+' & '+d['status']+r' \\')
 report={'protocol_sha256':sha,'counts':counts,'full_study_terminal':final,'tasks':task_results,'families':families,'supported_families':supported,'complete_results_only':True,'negative_outcomes_retained':True};(ROOT/'analysis/NEW_TASK_PROGRESS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 header=r'\subsection{Prospective evaluation on three additional task families}\label{sec:new_tasks}'
 text=header+'\nThe new fixed study covers six previously untested tasks: ETTh1/ETTh2 forecasting, UCI HAR/FordA sequence classification, and Digits/Olivetti denoising reconstruction. Every task compares AdamW, Euclidean-Kronecker, and BuresMomentum using the same six-rate validation-only search budget (2000 updates per rate), followed by five paired seeds at 5000 updates per method. The 108 tuning runs never load test arrays; all 90 full runs remain declared regardless of their outcomes. Rates are selected within each method without test scores. Appendix~\\ref{app:new_tasks} gives the frozen splits, models, full results, uncertainty rule, and all failures.\n'
 text+=f"The current snapshot contains {counts['tune']['complete']}/108 completed tuning runs and {counts['full']['complete']}/90 completed full runs. "
 if not final:text+='Family-level inference is pending completion of the fixed study; ongoing runs do not establish additional performance gains.\n'
 elif supported:text+='The predeclared paired-seed criterion supports '+', '.join(FAMILIES[f].lower() for f in supported)+f" ({len(supported)}/3 planned families). The complete six-task table, including unsupported families, remains in the appendix.\n"
 else:text+='No additional family satisfies the predeclared superiority criterion. All six tasks remain in the appendix.\n'
 text+=table(['Family','Full-budget coverage','Holm $p$','Conclusion'],family_rows,'Prospective family-level inference. A claim requires both tasks, all five seeds, positive paired intervals against both references, the prespecified effect floor, and Holm-adjusted significance across all three planned families. Interim completion does not trigger a claim.','tab:new_family_status')+'\n'
 if supported:
  positive=[t['task'] for t in p['tasks'] if t['family'] in supported];used=make_losses(records,jobs,positive,ROOT/'paper/figures/new_tasks_supported_losses.pdf');report['supported_loss_runs']=used
  text+=r'\begin{figure*}[t]\centering\includegraphics[width=.99\textwidth]{figures/new_tasks_supported_losses.pdf}\caption{Completed trajectories in task families satisfying the frozen criterion. Thin curves show every seed; thick curves show the five-seed mean at recorded updates. No temporal smoothing. All other families remain in Appendix~\ref{app:new_tasks}.}\label{fig:new_supported_losses}\end{figure*}'+'\n'
 (SECTIONS/'new_task_round.tex').write_text(text)
 details=table(['Task','Method','Selected LR','Complete; failed','Selected test','Final test'],rows,'All six new tasks and all three methods. Accuracy is in percent; forecasting and reconstruction use MSE. Mean and sample SD are displayed only when all 15 full-budget runs for a task complete, so missing seeds never disappear from a mean. An incomplete row remains pending.','tab:new_task_results',True)+'\n'
 details+=table(['Task','Reference','Bures gain','Paired 95\\% CI','One-sided $p$'],contrast_rows or [r'\multicolumn{5}{l}{No complete five-seed task contrast yet.}\\'],'Paired gains on fixed test splits: accuracy in percentage points and MSE as reference minus Bures. These are marginal seed-variability intervals, not simultaneous data-population intervals. Family claims additionally require Holm correction and the practical effect floor.','tab:new_task_contrasts',True)+'\n'
 tune_rows=[]
 for task in p['tasks']:
  for m in METHODS:
   cells=[]
   for lr in p['common_rates']:
    j=next(j for j in jobs if j['phase']=='tune' and j['task']==task['task'] and j['method']==m and j['lr']==lr);d=records.get(j['id'],{});v=d.get('validation_selected',{}).get('val_metric');cells.append(f"{v:.4g}" if d.get('status')=='complete' and v is not None else d.get('status','queued'))
   tune_rows.append(TASKS[task['task']]+' & '+NAMES[m]+' & '+' & '.join(cells)+r' \\')
 details+=table(['Task','Method']+[f'{x:.2g}' for x in p['common_rates']],tune_rows,'All 108 declared validation-only tuning trials. Column headers are learning rates; values are best validation accuracy (fraction) or MSE after 2000 updates. Failed, running, and queued cells remain explicit. No tuning trial evaluates the test set.','tab:new_tuning',True)+'\n'
 if any(d.get('phase')=='full' and d.get('history') for d in records.values()):
  report['all_loss_runs']=make_losses(records,jobs,[t['task'] for t in p['tasks']],ROOT/'paper/figures/new_tasks_all_losses.pdf');details+=r'\begin{figure}[p]\centering\includegraphics[width=.98\textwidth]{figures/new_tasks_all_losses.pdf}\caption{Every new task with a recorded full-run trajectory, including unsupported and unfinished arms. Thin lines show every recorded seed trajectory; a thick mean is drawn only after all five seeds for that method complete. Tasks without a recorded full-run trajectory remain pending in Table~\ref{tab:new_task_results}. Partial lines do not establish full-budget results. Training and validation are separate; there is no temporal smoothing.}\label{fig:new_all_losses}\end{figure}'+'\n'
 (SECTIONS/'new_task_results.tex').write_text(details);(ROOT/'analysis/NEW_TASK_PROGRESS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report
if __name__=='__main__':print(json.dumps(build()['counts'],indent=2))
