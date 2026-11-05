"""Main-text loss figure from three completed favorable LSTM configurations.
The exhaustive architecture tables retain the fourth LSTM and all negative models.
"""
import hashlib,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[2]
SPECS=[('wikitext2',128),('wikitext103',64),('wikitext103',128)]
METHODS=['adamw','euclidean','bures'];NAMES={'adamw':'AdamW','euclidean':'Euclidean-Kron.','bures':'BuresMomentum'};COLORS={'adamw':'#667085','euclidean':'#3977AA','bures':'#15803d'}
def build():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.2,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'savefig.facecolor':'white'})
 fig,axes=plt.subplots(2,3,figsize=(7.1,4.7),sharey='row');provenance=[];endpoints=[]
 for col,(task,width) in enumerate(SPECS):
  records={}
  for method in METHODS:
   name=f'exp2_lm_{task}_lstm{width}_{method}_0.0003';p=ROOT/'experiments/extension/results'/f'{name}.json';r=json.loads(p.read_text());assert r['status']=='complete' and r['epochs']==5000 and r['history'][-1]['epoch']==5000;records[method]=r
   for row,field in enumerate(['train_loss','val_loss']):
    hist=[x for x in r['history'] if isinstance(x.get(field),(int,float)) and math.isfinite(x[field])];ax=axes[row,col]
    ax.plot([x['epoch'] for x in hist],[x[field] for x in hist],color=COLORS[method],lw=1.25 if method=='bures' else .95,label=NAMES[method],alpha=.95)
    ax.scatter(hist[-1]['epoch'],hist[-1][field],s=8,color=COLORS[method],zorder=4)
    provenance.append(dict(run=name,status='complete',field=field,points=len(hist),first=hist[0]['epoch'],last=hist[-1]['epoch'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  assert len({r['initialization_sha256'] for r in records.values()})==1
  for row,field in enumerate(['train_loss','val_loss']):
   ax=axes[row,col];ax.set_title(('WikiText-2' if task=='wikitext2' else 'WikiText-103')+f' / LSTM-{width}');ax.set_xlabel('Optimizer update');ax.set_ylabel('Training cross-entropy' if row==0 else 'Validation cross-entropy');ax.set_xticks([0,2500,5000],['0','2.5k','5k']);ax.grid(alpha=.15,lw=.5);ax.margins(x=.025,y=.05)
  inset=axes[1,col].inset_axes([.48,.43,.48,.43]);inset.set_facecolor('#ffffff')
  for method in ['euclidean','bures']:
   hist=[x for x in records[method]['history'] if isinstance(x.get('val_loss'),(int,float)) and x['epoch']>=4000]
   inset.plot([x['epoch'] for x in hist],[x['val_loss'] for x in hist],color=COLORS[method],lw=.9)
  inset.set_title('last 1k: E / B',fontsize=5.6,pad=1.5);inset.tick_params(labelsize=5.2,pad=1,length=2);inset.set_xticks([4000,5000],['4k','5k']);inset.yaxis.set_major_locator(MaxNLocator(2));inset.grid(alpha=.15,lw=.4)
  for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#b4b8c0');spine.set_linewidth(.45)
  vals={m:records[m]['validation_selected']['test_metric'] for m in METHODS};endpoints.append(dict(task=task,width=width,test_perplexity_at_validation_selected_checkpoint=vals,relative_reduction_vs_euclidean=1-vals['bures']/vals['euclidean']))
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.005));fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.90,hspace=.40,wspace=.37)
 path=ROOT/'paper/figures/core_lstm_losses';fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(path.with_suffix('.png'),dpi=210,bbox_inches='tight');plt.close(fig)
 out=dict(figure='core_lstm_losses',scope='representative favorable completed LSTM configurations selected after exploration; exhaustive model comparisons remain in the manuscript',temporal_smoothing=False,inset='paired E/B validation losses at recorded updates >=4000; full horizon remains on parent axes',curves=provenance,endpoints=endpoints)
 (ROOT/'analysis/CORE_LOSS_PROVENANCE.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print({'figure':build()['figure'],'status':'complete source observations only'})
