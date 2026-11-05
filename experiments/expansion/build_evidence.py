"""Exhaustive completed comparisons with explicitly post-exploration representative panels and seed follow-up."""
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];SEC=ROOT/'paper/sections';FIG=ROOT/'paper/figures'
METHODS=['adamw','euclidean','bures'];NAMES={'adamw':'AdamW','euclidean':'Euclidean','bures':'Bures'};COL={'adamw':'#667085','euclidean':'#3977AA','bures':'#15803d'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
def build():
 records={p.stem:json.loads(p.read_text()) for p in (ROOT/'experiments/extension/results').glob('*.json')}
 jobs=json.loads((ROOT/'experiments/extension/jobs_all.json').read_text());exp=json.loads((ROOT/'experiments/expansion/jobs.json').read_text())
 groups=collections.defaultdict(dict)
 for j in exp:
  r=records.get(j['id'],{})
  if r.get('status')=='complete':groups[(r['family'],r['task'],r['model'],r['hidden'],r['lr'])][r['method']]=r
 comparisons=[]
 for config,arms in sorted(groups.items()):
  if not all(m in arms for m in METHODS):continue
  # The paired methods must share model initialization and all declared optimization settings.
  pair=[arms[m] for m in ['euclidean','bures']]
  assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256'],config
  assert pair[0]['source_sha256']==pair[1]['source_sha256'],config
  assert pair[0]['data_meta']==pair[1]['data_meta'],config
  # Device IDs identify scheduling lanes, not a paired experimental setting.
  # The initialization, data split, source, update budget, and optimizer
  # settings above and below are the quantities that must match.
  for key in ['hidden','epochs','batch','block','eval_every','diag_every','lr','seed']:
   assert pair[0][key]==pair[1][key],(config,key)
  comparisons.append(dict(config=config,runs={m:arms[m]['id'] for m in METHODS},selected_test={m:arms[m]['validation_selected']['test_metric'] for m in METHODS},final_test={m:arms[m]['final']['test_metric'] for m in METHODS}))
 graph=[c for c in comparisons if c['config'][0]=='graph'];lm=[c for c in comparisons if c['config'][0]=='lm']
 def fmt(v,acc):return f'{100*v:.1f}' if acc else f'{v:.2f}'
 def table(name,rows,caption,label,graph=False):
  ls=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llrrrrr}\toprule',r'Dataset & Model & AdamW & Euclidean & Bures & $\Delta$ vs E & $\Delta$ vs A \\ \midrule']
  for c in rows:
   _,task,model,h,lr=c['config'];v=c['selected_test'];good=max(v.values()) if graph else min(v.values());values=[]
   for m in METHODS:
    st=fmt(v[m],graph);values.append(r'\textbf{'+st+'}' if v[m]==good else st)
   delta=lambda baseline:100*(v['bures']-v[baseline]) if graph else 100*(1-v['bures']/v[baseline])
   model={'sage':'GraphSAGE','gcnii':'GCNII','gat':'GAT'}.get(model,model.upper()+'-'+str(h))
   ls.append(task.replace('_','-')+' & '+model+' & '+' & '.join(values)+f" & {delta('euclidean'):+.2f} & {delta('adamw'):+.2f} "+r'\\')
  ls += [r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+r'}\end{table*}'];(SEC/name).write_text('\n'.join(ls)+'\n')
 table('graph_architecture_comparison.tex',graph,'All nine completed graph architecture configurations, seed 42, 1000 epochs, common rate $10^{-3}$. Entries are test accuracy (\\%) at the checkpoint maximizing validation accuracy; deltas are percentage points. Bold marks the best method within a row. Every declared graph architecture/dataset combination is included; final checkpoints and raw trajectories remain in the appendix.','tab:graph_architecture',True)
 table('word_architecture_comparison.tex',lm,'All ten completed word-language architecture configurations, seed 42, 5000 updates, common rate $3\\cdot10^{-4}$. Entries are test word perplexity at the minimum-validation-perplexity checkpoint (lower is better); deltas are relative perplexity reductions in \\%. Bold marks the best method within a row. The vocabulary is capped at 8192; recurrent/Transformer core matrices use the paired optimizer and embeddings/output use common AdamW.','tab:word_architecture')
 # A full rate-selected tabular table retains the unfavorable configurations.
 tab=collections.defaultdict(list)
 for key,arms in groups.items():
  if key[0]=='tabular':
   for m,r in arms.items():
    if m in METHODS:tab[(key[1],m)].append(r)
 lines=[r'\begin{table}[ht]\centering\small',r'\begin{tabular}{llrrr}\toprule Task & Method & Selected LR & Validation & Test \\ \midrule']
 selected=[]
 for (task,m),arms in sorted(tab.items()):
  maximize=arms[0]['metric_direction']=='max';r=sorted(arms,key=lambda r:((-1 if maximize else 1)*r['validation_selected']['val_metric'],r['lr'],r['id']))[0];v=r['validation_selected']
  selected.append(dict(task=task,method=m,run=r['id'],lr=r['lr'],validation=v['val_metric'],test=v['test_metric']))
  lines.append(f"{task.replace('_',' ')} & {NAMES[m]} & {r['lr']:.1g} & {v['val_metric']:.4f} & {v['test_metric']:.4f} "+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\caption{Tabular selection over the two common declared rates uses validation only, with lower LR breaking exact ties. Classification entries are accuracy fractions; California uses RMSE. This table includes all five tabular datasets and all three methods.}',r'\label{tab:tabular_rate_selected}\end{table}'];(SEC/'tabular_rate_selected.tex').write_text('\n'.join(lines)+'\n')
 # Representative completed plots, explicitly labeled; exhaustive tables determine win counts.
 specs=[('graph','Cora','sage',64,.001),('graph','Cora','gcnii',64,.001),('graph','PubMed','gat',64,.001),('lm','wikitext2','lstm',64,.0003),('lm','wikitext2','lstm',128,.0003),('lm','wikitext103','lstm',128,.0003)]
 provenance=[];fig,axes=plt.subplots(2,3,figsize=(10,5))
 for i,(ax,key) in enumerate(zip(axes.flat,specs)):
  arms=groups[key];family,task,model,h,lr=key
  for m in METHODS:
   r=arms[m];rows=[x for x in r['history'] if isinstance(x.get('val_loss'),(int,float)) and math.isfinite(x['val_loss'])]
   ax.plot([x['epoch'] for x in rows],[x['val_loss'] for x in rows],label=NAMES[m],color=COL[m],lw=1.05)
   provenance.append(dict(run=r['id'],sha256=hashlib.sha256((ROOT/'experiments/extension/results'/f"{r['id']}.json").read_bytes()).hexdigest(),field='val_loss',points=len(rows),first=rows[0]['epoch'],last=rows[-1]['epoch']))
  model={'sage':'GraphSAGE','gcnii':'GCNII','gat':'GAT'}.get(model,'LSTM-'+str(h))
  ax.set(title=f'({chr(97+i)}) {task} / {model}',xlabel='Epoch' if family=='graph' else 'Update',ylabel='Validation cross-entropy',yscale='log');ax.grid(alpha=.15);ax.legend(frameon=False);ax.margins(x=.01)
 fig.tight_layout(pad=1.);fig.savefig(FIG/'evidence_completed_losses.pdf',bbox_inches='tight');fig.savefig(FIG/'evidence_completed_losses.png',dpi=170,bbox_inches='tight');plt.close(fig)
 # Follow-up status is per entire three-seed triplet, never based on a favorable partial result.
 repeat=json.loads((ROOT/'experiments/expansion/multiseed_jobs.json').read_text());configs=[];follow=[]
 for task,model in [('Cora','gcnii'),('Cora','sage'),('PubMed','gat')]:
  arms={};ids=[]
  for seed in [42,43,44]:
   for m in METHODS:
    key=next(j['id'] for j in exp if j['family']=='graph' and j['task']==task and j['model']==model and j['method']==m) if seed==42 else f'repeat_graph_{task}_{model}_{m}_seed{seed}'
    rr=records.get(key,{})
    if rr.get('status')=='complete':arms[(seed,m)]=rr
    ids.append(key)
  if len(arms)==9:
   for seed in [42,43,44]:
    shared=[arms[(seed,m)] for m in METHODS]
    assert len({r['initialization_sha256'] for r in shared})==1,(task,model,seed,'initialization')
    assert len({r['source_sha256'] for r in shared})==1,(task,model,seed,'optimizer source')
    assert all(r['seed']==seed and r['epochs']==1000 for r in shared)
   assert len({arms[(seed,'bures')]['initialization_sha256'] for seed in [42,43,44]})==3,(task,model,'seed variation')
   vals={m:np.array([100*arms[(s,m)]['validation_selected']['test_metric'] for s in [42,43,44]]) for m in METHODS}
   stats={m:dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),values=v.tolist()) for m,v in vals.items()}
   stats.update(task=task,model=model,seeds=[42,43,44],paired_delta={m:dict(mean=float((vals['bures']-vals[m]).mean()),sd=float((vals['bures']-vals[m]).std(ddof=1)),wins=int(np.sum(vals['bures']>vals[m]))) for m in ['adamw','euclidean']},runs=ids)
   follow.append(stats)
  configs.append(dict(task=task,model=model,completed=len(arms),required=9))
 complete_follow=sum(records.get(j['id'],{}).get('status')=='complete' for j in repeat)
 lines=[r'\subsection{Prespecified seed follow-up on the promising graph settings}',r'\label{sec:seed_followup}',r'The Cora GCNII, Cora GraphSAGE, and PubMed GAT settings were selected after the seed-42 exploration. Seeds 43 and 44 and all three optimizers were then declared together, adding 18 runs at the unchanged 1000-epoch budget and rate $10^{-3}$. Splits stay fixed, and each seed shares initialization across methods. These repeats measure sensitivity to training randomness on the selected settings; they do not constitute an independent task-selection experiment. Every declared repeat is retained.']
 if follow:
  lines.append(f'This snapshot has {complete_follow}/18 completed additional seed runs and {len(follow)}/3 complete three-seed configurations. The table includes only complete configurations.')
  for st in follow:
   de=st['paired_delta']['euclidean'];da=st['paired_delta']['adamw'];title={'sage':'GraphSAGE','gcnii':'GCNII','gat':'GAT'}[st['model']]
   lines.append(f"For {st['task']} {title}, the paired mean change is {de['mean']:+.2f} percentage points relative to Euclidean and {da['mean']:+.2f} relative to AdamW; Bures wins {de['wins']}/3 and {da['wins']}/3 individual seeds, respectively.")
  lines += [r'\begin{table*}[t]\centering\small',r'\begin{tabular}{llrrrrr}\toprule Dataset & Model & AdamW & Euclidean & Bures & $\Delta$ vs E & $\Delta$ vs A \\ \midrule']
  for st in follow:
   vals=[f"{st[m]['mean']:.2f} $\\pm$ {st[m]['sd']:.2f}" for m in METHODS];diff=st['paired_delta'];lines.append(st['task']+' & '+st['model'].upper()+' & '+' & '.join(vals)+f" & {diff['euclidean']['mean']:+.2f} & {diff['adamw']['mean']:+.2f} "+r'\\')
  lines += [r'\bottomrule\end{tabular}',r'\caption{Completed three-seed follow-up: validation-selected test accuracy in \\%, mean $\\pm$ sample standard deviation over seeds 42, 43, 44. Deltas are paired mean percentage-point changes. Only configurations with all nine method/seed runs complete are included; no unfinished seed is substituted or discarded.}',r'\label{tab:seed_followup}\end{table*}']
 else:lines.append(f'This snapshot has {complete_follow}/18 completed additional seed runs. No multi-seed performance conclusion is drawn before a complete method/seed set is available.')
 if follow:
  fig,axes=plt.subplots(len(follow),2,figsize=(8,2.45*len(follow)),squeeze=False)
  seed_provenance=[]
  for row,st in enumerate(follow):
   for col,field in enumerate(['train_loss','val_loss']):
    ax=axes[row,col]
    for m in METHODS:
     series=[]
     for seed in [42,43,44]:
      key=next(j['id'] for j in exp if j['family']=='graph' and j['task']==st['task'] and j['model']==st['model'] and j['method']==m) if seed==42 else f"repeat_graph_{st['task']}_{st['model']}_{m}_seed{seed}"
      rr=records[key];hist=[x for x in rr['history'] if isinstance(x.get(field),(int,float)) and math.isfinite(x[field])];xs=np.array([x['epoch'] for x in hist]);ys=np.array([x[field] for x in hist]);series.append((xs,ys));ax.plot(xs,ys,color=COL[m],lw=.45,alpha=.3)
      seed_provenance.append(dict(run=key,field=field,points=len(hist),last=int(xs[-1])))
     assert all(np.array_equal(series[0][0],ss[0]) for ss in series)
     ys=np.stack([ss[1] for ss in series]);mean=ys.mean(0);sd=ys.std(0,ddof=1)
     ax.plot(xs,mean,color=COL[m],lw=1.15,label=NAMES[m]);ax.fill_between(xs,ys.min(0),ys.max(0),color=COL[m],alpha=.10,lw=0)
    ax.set(title=st['task']+' / '+st['model'].upper(),xlabel='Epoch',ylabel='Training cross-entropy' if col==0 else 'Validation cross-entropy',yscale='log');ax.grid(alpha=.15);ax.legend(frameon=False);ax.margins(x=.01)
  fig.tight_layout(pad=1.05);fig.savefig(FIG/'multiseed_loss_followup.pdf',bbox_inches='tight');fig.savefig(FIG/'multiseed_loss_followup.png',dpi=170,bbox_inches='tight');plt.close(fig)
  lines += [r'\begin{figure*}[t]\centering\includegraphics[width=.96\textwidth]{figures/multiseed_loss_followup.pdf}',r'\caption{Three-seed graph follow-up on fully completed configurations. Left: training loss; right: validation loss. Thin traces show each saved seed trajectory; bold traces show the three-seed mean at the same recorded epochs, with the observed minimum-to-maximum seed envelope shaded. The bands describe the observed seed spread and are not confidence intervals. No smoothing or interpolation is applied.}',r'\label{fig:multiseed_followup}\end{figure*}']
 else:seed_provenance=[]
 (SEC/'multiseed_followup.tex').write_text('\n'.join(lines)+'\n')
 # Full exact per-seed table gives paired deltas without pretending n=3 is a calibrated CI.
 lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Every completed seed triplet in the follow-up settings. Validation-selected and final-checkpoint test accuracies are both reported in \%; unfavorable checkpoint outcomes remain visible.}',r'\label{tab:all_seed_values}',r'\begin{tabular}{llrrrrrrr}',r'\toprule Dataset & Model & Seed & \multicolumn{3}{c}{Validation-selected test} & \multicolumn{3}{c}{Final test} \\',r' & & & AdamW & Euclid. & Bures & AdamW & Euclid. & Bures \\ \midrule']
 for c in configs:
  for seed in [42,43,44]:
   vals={};finalvals={}
   for m in METHODS:
    key=next(j['id'] for j in exp if j['family']=='graph' and j['task']==c['task'] and j['model']==c['model'] and j['method']==m) if seed==42 else f"repeat_graph_{c['task']}_{c['model']}_{m}_seed{seed}"
    rr=records.get(key,{})
    if rr.get('status')=='complete':
     vals[m]=100*rr['validation_selected']['test_metric'];finalvals[m]=100*rr['final']['test_metric']
   if len(vals)==3:lines.append(f"{c['task']} & {c['model'].upper()} & {seed} & {vals['adamw']:.2f} & {vals['euclidean']:.2f} & {vals['bures']:.2f} & {finalvals['adamw']:.2f} & {finalvals['euclidean']:.2f} & {finalvals['bures']:.2f} "+r'\\')
 lines.append(r'\bottomrule\end{tabular}\end{table}');(SEC/'multiseed_all_values.tex').write_text('\n'.join(lines)+'\n')
 # Raw curves for all nine graph architectures and all ten LM settings remain in the full atlas.
 statuses=collections.Counter(records.get(j['id'],{}).get('status','queued') for j in jobs)
 out=dict(status_counts=dict(statuses),all_declared=len(jobs),fixed_configuration_comparisons=comparisons,tabular_validation_selection=selected,multiseed=follow,multiseed_progress=configs,multiseed_additional_complete=complete_follow,figure_provenance=provenance,multiseed_figure_provenance=seed_provenance,graph_wins_against_both=sum(c['selected_test']['bures']>max(c['selected_test']['euclidean'],c['selected_test']['adamw']) for c in graph),graph_total=len(graph),word_lstm_wins_against_both=sum(c['selected_test']['bures']<min(c['selected_test']['euclidean'],c['selected_test']['adamw']) for c in lm if c['config'][2]=='lstm'),word_lstm_total=sum(c['config'][2]=='lstm' for c in lm))
 for filename in ['multiseed_followup.tex','multiseed_all_values.tex']:
  path=SEC/filename;path.write_text(path.read_text().replace(r'\\%',r'\%').replace(r'\\pm',r'\pm'))
 coverage_path=ROOT/'analysis/PROPOSAL_COVERAGE.json'
 if coverage_path.exists():
  import datetime
  coverage=json.loads(coverage_path.read_text());coverage['flow_expansion']['counts']=dict(collections.Counter(records.get(j['id'],{}).get('status','queued') for j in exp));coverage['multiseed_followup']['status']=configs;coverage['multiseed_followup']['additional_complete']=complete_follow;coverage['all_declared_status']=dict(statuses);coverage['all_declared_count']=len(jobs);coverage['updated_at']=datetime.datetime.now().astimezone().isoformat();coverage_path.write_text(json.dumps(coverage,indent=2)+'\n')
 (ROOT/'analysis/COMPLETED_EVIDENCE.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ['all_declared','status_counts','graph_wins_against_both','graph_total','word_lstm_wins_against_both','word_lstm_total','multiseed_additional_complete']})
if __name__=='__main__':build()
