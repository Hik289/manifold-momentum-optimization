"""Reproducible unsmoothed loss panels and the architecture-expansion ledger."""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];FIG=ROOT/'paper/figures';SEC=ROOT/'paper/sections'
NAMES={'adamw':'AdamW','sgdm':'SGD + momentum','euclidean':'Euclidean-Kron.','bures':'BuresMomentum','shampoo':'Shampoo ref.','scale_only':'Scale-only'}
COLORS={'adamw':'#667085','sgdm':'#9A6FB0','euclidean':'#3977AA','bures':'#15803d','shampoo':'#4A9C7A','scale_only':'#A887BF'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
def build():
    records={}
    for p in (ROOT/'experiments/extension/results').glob('*.json'):
        try:r=json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):continue
        if r.get('status') in ('complete','failed','running'):records[p.stem]=r
    manifest={}
    def panel(axis,ids,field,title,xlabel='Epoch',labels=None):
        used=[]
        for key in ids:
            r=records.get(key)
            if not r:continue
            rows=[v for v in r['history'] if isinstance(v.get(field),(int,float)) and math.isfinite(v[field]) and v[field]>=0]
            if not rows:continue
            method=r['method'];status=r['status'];name=(labels or {}).get(key,NAMES.get(method,method))
            name+=(' [failed]' if status=='failed' else ' [running]' if status=='running' else '')
            axis.plot([v['epoch'] for v in rows],[v[field] for v in rows],color=COLORS.get(method,'#777777'),lw=1.1,ls='--' if status=='failed' else ':' if status=='running' else '-',label=name,alpha=.94)
            if status!='complete':axis.scatter(rows[-1]['epoch'],rows[-1][field],marker='x' if status=='failed' else 'o',s=13,color=COLORS.get(method,'#777777'),zorder=4)
            used.append(dict(run=key,status=status,field=field,points=len(rows),first=rows[0]['epoch'],last=rows[-1]['epoch']))
        axis.set(title=title,xlabel=xlabel,ylabel='Training loss' if field=='train_loss' else 'Validation loss',yscale='log')
        if any(any(y==0 for y in line.get_ydata()) for line in axis.lines):axis.set_yscale('symlog',linthresh=1e-8)
        axis.grid(True,alpha=.15,linewidth=.5);axis.legend(frameon=False,loc='best');axis.margins(x=.015)
        return used
    def group(name,specs,cols=3):
        rows=(len(specs)+cols-1)//cols;fig,axes=plt.subplots(rows,cols,figsize=(3.25*cols,2.45*rows),squeeze=False)
        entries=[]
        for i,(ax,spec) in enumerate(zip(axes.flat,specs)):
            ids,field,title,*rest=spec;entries.extend(panel(ax,ids,field,f'({chr(97+i)}) {title}',*(rest or [])))
        for ax in list(axes.flat)[len(specs):]:ax.set_visible(False)
        fig.tight_layout(pad=1.05);fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig);manifest[name]=entries
    methods=['adamw','euclidean','bures']
    graph=lambda task:[f'graph_{task}_{m}_full' for m in ['adamw','euclidean','bures','shampoo']]
    tab=lambda task,stage:[f'tabular_{task}_{m}_{stage}' for m in methods]
    group('loss_main_long',[(graph('Cora'),'train_loss','GCN / Cora'),(graph('CiteSeer'),'train_loss','GCN / CiteSeer'),(graph('PubMed'),'train_loss','GCN / PubMed'),(tab('adult','full'),'train_loss','MLP / Adult, 1000 epochs'),([f'lm_lrgrid_wikitext103_{m}_3e-05' for m in methods],'val_loss','Byte Transformer / WikiText excerpt','Update'),([f'sciml_poisson_{m}_full' for m in methods],'train_loss','PINN / Poisson','Collocation update')])
    group('loss_graph_full',[(graph(t),f,t) for f in ['train_loss','val_loss'] for t in ['Cora','CiteSeer','PubMed']])
    group('loss_tabular_full',[(tab(t,s),f,f'{t.title()} / {s}') for f in ['train_loss','val_loss'] for t,s in [('adult','full'),('covertype','screen'),('california','full')]])
    group('loss_language_full',[( [f'lm_{t}_{m}_screen' for m in methods],f,t+' / byte model','Update') for f in ['train_loss','val_loss'] for t in ['wikitext103','openwebtext','c4']])
    group('loss_vision_full',[( [f'vision_{t}_{m}_screen' for m in methods],f,'ResNet-18 / '+t) for f in ['train_loss','val_loss'] for t in ['cifar10','cifar100']],cols=2)
    group('loss_pinn_full',[( [f'sciml_{t}_{m}_full' for m in methods],f,t.title()+' PINN','Collocation update') for f in ['train_loss','val_loss'] for t in ['poisson','burgers']],cols=2)
    group('loss_shift_full',[( [f'shift_mnist_{m}_full' for m in methods],f,'Rotated MNIST') for f in ['train_loss','val_loss']],cols=2)
    group('loss_matching_floor',[( [f'floor_{floor}_{m}' for m in ['euclidean','bures']],f,'Cora / matching floor '+floor) for f in ['train_loss','val_loss'] for floor in ['.1','.001','.00001']])
    abl_groups=[('Factor channels',['left_only','right_only','identity']),('Normalization',['no_norm','no_momentum','no_clip']),('State geometry',['diagonal','rotate','permute','ai']),('Root refresh',['tau1','tau50']),('Block size',['block64','block128']),('Second moment',['beta0.9','beta0.995','beta0.999'])]
    fig,axes=plt.subplots(2,3,figsize=(9.75,4.9));entries=[]
    for i,(ax,(title,keys)) in enumerate(zip(axes.flat,abl_groups)):
        for k,color in zip(['Full']+keys,['#15803d','#3977AA','#4A9C7A','#9A6FB0','#667085']):
            key='graph_Cora_bures_full' if k=='Full' else 'ablation_'+k;r=records.get(key)
            if r:
                rows=r['history'];ax.plot([v['epoch'] for v in rows],[v['train_loss'] for v in rows],lw=.9,color=color,label=k);entries.append(dict(run=key,status=r['status'],field='train_loss',points=len(rows),last=rows[-1]['epoch']))
        ax.set(title=f'({chr(97+i)}) {title}',xlabel='Epoch',ylabel='Training loss',yscale='log');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=6)
    fig.tight_layout();fig.savefig(FIG/'loss_ablation_full.pdf',bbox_inches='tight');fig.savefig(FIG/'loss_ablation_full.png',dpi=170,bbox_inches='tight');plt.close(fig);manifest['loss_ablation_full']=entries
    specs=[]
    for m in methods:
        ids=[f'lm_lrgrid_wikitext103_{m}_{lr}' for lr in ['1e-05','3e-05','0.0001','0.0003','0.001','0.003']]
        for f in ['train_loss','val_loss']:specs.append((ids,f,NAMES[m]+' / six rates','Update',{k:k.rsplit('_',1)[1] for k in ids}))
    # Rate curves need distinct colors, independently of optimizer identity.
    fig,axes=plt.subplots(3,2,figsize=(8,7));entries=[]
    for ax,(ids,field,title,xlab,labels) in zip(axes.flat,specs):
        for key,color in zip(ids,plt.cm.viridis([.05,.22,.39,.56,.73,.90])):
            r=records.get(key)
            if not r:continue
            rows=[v for v in r['history'] if v.get(field,0)>0];ax.plot([v['epoch'] for v in rows],[v[field] for v in rows],label=labels[key]+(' failed' if r['status']=='failed' else ''),lw=.9,color=color,ls='--' if r['status']=='failed' else '-');entries.append(dict(run=key,status=r['status'],field=field,points=len(rows),last=rows[-1]['epoch']))
        ax.set(title=title,xlabel=xlab,ylabel=field.replace('_',' '),yscale='log');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=6)
    fig.tight_layout();fig.savefig(FIG/'loss_rate_grid_full.pdf',bbox_inches='tight');fig.savefig(FIG/'loss_rate_grid_full.png',dpi=170,bbox_inches='tight');plt.close(fig);manifest['loss_rate_grid_full']=entries
    jobs=json.loads((ROOT/'experiments/expansion/jobs.json').read_text());complete=[r for k,r in records.items() if r.get('cohort')=='flow_expansion' and r['status']=='complete']
    lines=[r'\begin{longtable}{lllrrrr}',r'\caption{Completed architecture-expansion runs. Each row is one declared run; pending and failed runs do not enter performance comparisons. LM budgets are word-token updates, while all other budgets are epochs. Classification accuracy is a fraction; California RMSE and word perplexity are lower-is-better. Wine targets are binary quality $\ge6$.}\\',r'\toprule Task & Model & Method & LR & Budget & Final test & Val-selected test \\ \midrule',r'\endfirsthead',r'\toprule Task & Model & Method & LR & Budget & Final test & Val-selected test \\ \midrule\endhead']
    for r in sorted(complete,key=lambda r:(r['family'],r['task'],r['model'],r['lr'],r['method'])):
        lines.append(f"{r['task'].replace('_',' ')} & {r['model'].replace('_',' ')}-{r['hidden']} & {NAMES[r['method']]} & {r['lr']:.1g} & {r['epochs']} & {r['final']['test_metric']:.4g} & {r['validation_selected']['test_metric']:.4g} "+r'\\')
    if not complete:lines.append(r'\multicolumn{7}{l}{No architecture-expansion run has completed in this snapshot.}\\')
    lines.extend([r'\bottomrule\end{longtable}']);(SEC/'expansion_results_table.tex').write_text('\n'.join(lines))
    groups=[('Tabular MLP',lambda j:j['family']=='tabular'),('Image MLP',lambda j:j['family']=='vision' and j['model']=='mlp'),('Word language models',lambda j:j['family']=='lm'),('GAT / SAGE / GCNII',lambda j:j['family']=='graph'),('CNN classifier heads',lambda j:j['model'].endswith('_head'))]
    lines=[r'\begin{table*}[t]\centering\small',r'\begin{tabular}{lrrrrr}\toprule Family & Declared & Complete & Failed & Running & Queued \\ \midrule']
    for name,pred in groups:
        arms=[j for j in jobs if pred(j)];counts={s:sum(records.get(j['id'],{}).get('status','queued')==s for j in arms) for s in ['complete','failed','running','queued']}
        lines.append(f"{name} & {len(arms)} & {counts['complete']} & {counts['failed']} & {counts['running']} & {counts['queued']} "+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\caption{Declared expansion beyond the original 105-run queue. Status is operational progress, not experimental evidence. Non-LM budgets are 1000 epochs; word LMs request 5000 updates (2.56M processed tokens).}',r'\label{tab:expansion_scope}\end{table*}']);(SEC/'expansion_scope_table.tex').write_text('\n'.join(lines))
    if complete:
        configs=sorted({(r['family'],r['task'],r['model'],r['hidden'],r['lr']) for r in complete})
        # Plot all observed arms at a completed configuration, without selecting by test outcomes.
        specs=[]
        for family,task,model,h,lr in configs:
            ids=[j['id'] for j in jobs if (j['family'],j['task'],j['model'],j['hidden'],j['lr'])==(family,task,model,h,lr)]
            specs.extend([(ids,'train_loss',f'{model}-{h} / {task}, {lr:g}','Update' if family=='lm' else 'Epoch'),(ids,'val_loss',f'{model}-{h} / {task}, {lr:g}','Update' if family=='lm' else 'Epoch')])
        chunks=[]
        for off in range(0,len(specs),6):
            name=f'loss_expansion_{off//6+1:02d}';group(name,specs[off:off+6],cols=2);chunks.append(r'\begin{figure}[p]\centering\includegraphics[width=.94\textwidth]{figures/'+name+r'.pdf}\caption{Architecture-expansion loss trajectories. Every observed point is retained; incomplete arms are explicitly marked.}\end{figure}')
        (SEC/'expansion_loss_figures.tex').write_text('\n'.join(chunks))
    else:(SEC/'expansion_loss_figures.tex').write_text('% Awaiting completed architecture-expansion configurations.\n')
    (ROOT/'analysis/LOSS_FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
if __name__=='__main__':print({k:len(v) for k,v in build().items()})
