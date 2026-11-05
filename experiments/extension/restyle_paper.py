"""Reproducible WWW-style tables; active graphical assets are loss curves only."""
import hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];SEC=ROOT/'paper/sections';RES=ROOT/'experiments/extension/results';MARK='% WWW_STYLE_TABLE_V1'

def balanced(text,start):
 assert text[start]=='{';depth=0
 for i in range(start,len(text)):
  if text[i] in '{}':
   # Only unescaped braces delimit TeX groups.
   back=0;k=i-1
   while k>=0 and text[k]=='\\':back+=1;k-=1
   if back%2:continue
   depth+=1 if text[i]=='{' else -1
   if depth==0:return i+1
 raise ValueError('Unbalanced TeX group')

def row_style(block):
 block=block.replace(r'\toprule',r'\toprule'+'\n').replace(r'\midrule',r'\midrule'+'\n').replace(r'\bottomrule','\n'+r'\bottomrule')
 indices=[];lines=[]
 for line in block.splitlines():
  if '&' in line and r'\\' in line and r'\caption' not in line and r'\multicolumn' not in line:
   raw=line.split(r'\\',1)[0];cells=re.split(r'(?<!\\)&',raw)
   is_header=any(w in raw for w in ['Method','Dataset','Setting','Configuration','Task','Reference','Family']) and not re.search(r'\d\.\d',raw)
   if is_header:
    indices=[i for i,c in enumerate(cells) if re.search(r'\bBures(?:Momentum)?\b',c)]
   elif len(cells)>1:
    is_ours=any(re.search(r'\bBures(?:Momentum)?\b',c) for c in cells[:2]) and not any('Euclidean BW' in c for c in cells[:2])
    if is_ours:
     line=r'\rowcolor{oursbg}'+'\n'+line
    elif indices:
     for i in indices:
      if i<len(cells):
       value=cells[i].strip()
       if value.startswith(r'\textbf{') and value.endswith('}'):cells[i]=r'\ourbest{'+value[len(r'\textbf{'):-1]+'}'
       else:cells[i]=r'\ourscell{'+value+'}'
     line=' & '.join(cells)+line[len(raw):]
  lines.append(line)
 return '\n'.join(lines)

def style_block(match):
 s=match.group()
 if MARK in s:return s
 # A header or highlight has no numerical meaning; existing best-value bold is retained.
 s=s.replace(r'\rowcolor{oursbg}','').replace(r'\rowcolor{blue!7}','')
 if not s.startswith(r'\begin{longtable}'):
  m=re.search(r'\\caption\s*\{',s)
  if m:
   e=balanced(s,s.index('{',m.start()));cap=s[m.start():e];s=s[:m.start()]+s[e:]
   labs=list(re.finditer(r'\\label\{[^}]+\}',s));labels=''.join(x.group() for x in labs)
   for x in reversed(labs):s=s[:x.start()]+s[x.end():]
   # Put captions above the table and any resizing wrapper.
   m=re.match(r'\\begin\{table\*?\}(?:\[[^]]*\])?',s);at=m.end();s=s[:at]+'\n'+MARK+'\n'+cap+labels+'\n'+s[at:]
  s=s.replace(r'\centering\small',r'\centering\footnotesize')
  s=row_style(s)
 else:
  m=re.match(r'\\begin\{longtable\}\{',s);at=balanced(s,m.end()-1);s=s[:at]+'\n'+MARK+'\n'+s[at:];s=row_style(s)
  s='\\begingroup\\footnotesize\n'+s+'\n\\endgroup'
 return s

def write_table(filename,headers,rows,caption,label,aliases=(),long=False):
 if long:
  text='\\begin{longtable}{'+'l'+'r'*(len(headers)-1)+'}\n\\caption{'+caption+'}\\label{'+label+'}'+''.join('\\label{'+x+'}' for x in aliases)+r'\\'+'\n\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n\\endfirsthead\n\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n\\endhead\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{longtable}\n'
 else:
  text='\\begin{table*}[t]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}'+''.join('\\label{'+x+'}' for x in aliases)+'\n\\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n'
 (SEC/filename).write_text(text)

def build():
 provenance={};methods=['adamw','euclidean','bures','shampoo','scale_only'];names={'adamw':'AdamW','euclidean':'Euclidean-Kron.','bures':'BuresMomentum','shampoo':'Shampoo','scale_only':'Scale-only','log_euclidean':'Log-Euclidean','ai':'Affine-invariant'}
 def read(key):
  path=RES/(key+'.json');provenance[key]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
 rows=[]
 for task in ['Cora','CiteSeer','PubMed']:
  arms={m:read(f'graph_{task}_{m}_full') for m in methods}
  for epoch in [1,100,200,500,1000]:
   values=[100*next(x['val_metric'] for x in arms[m]['history'] if x['epoch']==epoch) for m in methods];best=max(values);cells=[r'\textbf{'+f'{v:.2f}'+'}' if v==best else f'{v:.2f}' for v in values];rows.append(task+f' & {epoch} & '+' & '.join(cells)+r' \\')
 write_table('graph_checkpoint_table.tex',['Dataset','Epoch']+[names[m] for m in methods],rows,r'\textbf{Graph validation accuracy at fixed checkpoints (\% $\uparrow$).} These sampled recorded epochs replace the validation-accuracy plot. All five methods complete 1000 epochs; the full training and validation loss trajectories remain in the loss atlas. Bold is the best method at the same dataset/epoch.','fig:graphcurves')
 rows=[]
 for m in ['euclidean','bures','scale_only']:
  r=read('graph_Cora_'+m+'_full')
  for epoch in [1,100,200,500,1000]:
   d=next(x for x in r['diagnostics'] if x['step']==epoch);rows.append(f"{names[m]} & {epoch} & {d['lambda_max_L']:.4g} & {d['scale_mean']:.4g} & {d['kappa_L']:.4g} & {d['clip_activation_rate']:.4g} "+r'\\')
 write_table('state_diagnostics_table.tex',['Method','Epoch',r'$\lambda_{\max}(L)$','Matching scale',r'$\kappa(L)$','Clip activation'],rows,r'\textbf{Cora matrix-state diagnostics.} Values are the recorded aggregate tile statistics, sampled at the same epochs as the replaced diagnostic plot. The matching lower bound is 0.1. These quantities describe the implemented optimizer state and do not measure population covariance drift.','fig:statescale')
 syn=read('synthetic_complete');rows=[]
 for task in ['static','scale','rotation','geodesic','jump','rotscale']:
  ms=['euclidean','bures','log_euclidean','ai'];best={key:min(syn[task+'_'+m][key] for m in ms) for key in ['mean_BW','final_BW','mean_root_error','mean_jitter']}
  for m in ms:
   d=syn[task+'_'+m];cells=[r'\textbf{'+f'{d[k]:.5f}'+'}' if d[k]==best[k] else f'{d[k]:.5f}' for k in best];rows.append(task+' & '+names[m]+' & '+' & '.join(cells)+r' \\')
 write_table('synthetic_comparison_full.tex',['Setting','Method',r'Mean BW $\downarrow$',r'Final BW $\downarrow$',r'Root error $\downarrow$',r'State jitter $\downarrow$'],rows,r'\textbf{All synthetic geometries and all four averaging rules.} Recorded metrics over 1000 updates, $d=8$, $\alpha=0.01$, common identity initialization. The static target is the offline finite-population BW barycenter; dynamic targets are specified noiseless paths. Means include initialization transients. Bold marks the minimum per setting/metric. The full source trajectories are preserved in the experiment archive.','tab:synthetic',aliases=['fig:synthetic'])
 rows=[]
 for d in syn['scalar_noise_scaling']:rows.append(f"{d['alpha']:.3g} & {d['empirical_mse']:.6g} & {d['exact_mse']:.6g} & {d['ratio']:.4f} "+r'\\')
 write_table('noise_drift_tables.tex',[r'$\alpha$','Empirical MSE','Exact stationary MSE','Empirical / exact'],rows,r'\textbf{Scalar stationary noise: numerical values.} Four hundred independent filters at each smoothing value; the known stationary MSE is $\alpha v/(2-\alpha)$. These are the observations underlying the replaced noise-scaling plot.','fig:noisedrift')
 text=(SEC/'noise_drift_tables.tex').read_text();rows=[]
 for d in syn['dynamic_grid']:rows.append(f"{d['alpha']:.3g} & {d['drift']:.4g} & {d['mse']:.6g} & {d['noise_proxy']:.6g} & {d['lag_squared']:.6g} & {d['noise_proxy']+d['lag_squared']:.6g} "+r'\\')
 write_table('noise_drift_grid.tex',[r'$\alpha$',r'Drift $\delta$','Observed MSE','Noise term','Squared lag','Noise + lag'],rows,r'\textbf{Complete drift/noise grid.} All nine declared cells are retained. The population model identifies the stationary noise term and the squared tracking lag $[(1-\alpha)\delta/\alpha]^2$. This table replaces the drift plot without inferring population terms from neural-network logs.','tab:drift_grid')
 (SEC/'noise_drift_tables.tex').write_text(text+'\n\\input{sections/noise_drift_grid}\n')

 rows=[]
 for task in ['wikitext2','wikitext103']:
  for width in [64,128]:
   arms={m:read(f'exp2_lm_{task}_lstm{width}_{m}_0.0003') for m in ['adamw','euclidean','bures']};values={m:r['validation_selected']['test_metric'] for m,r in arms.items()};best=min(values.values());cells=[r'\textbf{'+f'{values[m]:.2f}'+'}' if values[m]==best else f'{values[m]:.2f}' for m in ['adamw','euclidean','bures']];rows.append(('WikiText-2' if task=='wikitext2' else 'WikiText-103')+f' & LSTM-{width} & '+' & '.join(cells)+f" & {100*(1-values['bures']/values['euclidean']):+.2f} & {100*(1-values['bures']/values['adamw']):+.2f} "+r'\\')
 write_table('word_lstm_core.tex',['Dataset','Model','AdamW','Euclidean','Bures',r'$\Delta$ vs E',r'$\Delta$ vs A'],rows,r'\textbf{All four completed LSTM width/corpus configurations.} Validation-selected test word perplexity ($\downarrow$), seed 42, 5000 updates, common rate $3\cdot10^{-4}$. Deltas are relative perplexity reductions in \%. The architecture focus was chosen after exploration; the complete GRU/LSTM/Transformer comparison remains in Table~\ref{tab:word_architecture}.','tab:word_lstm_core')
 rows=[]
 evidence=json.loads((ROOT/'analysis/COMPLETED_EVIDENCE.json').read_text())
 for d in evidence['multiseed']:
  ms=['adamw','euclidean','bures'];top=max(d[m]['mean'] for m in ms);values=[]
  for m in ms:
   value=f"{d[m]['mean']:.2f} $\pm$ {d[m]['sd']:.2f}";values.append(r'\textbf{'+value+'}' if d[m]['mean']==top else value)
  de=d['paired_delta']['euclidean']['mean'];da=d['paired_delta']['adamw']['mean'];rows.append(d['task']+' & '+d['model'].upper()+' & '+' & '.join(values)+f' & {de:+.2f} & {da:+.2f} '+r'\\')
 write_table('seed_core.tex',['Dataset','Model','AdamW','Euclidean','Bures',r'$\Delta$ vs E',r'$\Delta$ vs A'],rows,r'\textbf{Completed three-seed graph follow-up.} Validation-selected test accuracy in \%, mean $\pm$ sample SD over seeds 42, 43, 44. Deltas are paired mean percentage-point changes. Bold marks the largest mean. All nine method/seed runs are complete within each configuration; the settings were selected after seed-42 exploration.','tab:seed_core')
 for path in SEC.glob('*.tex'):
  if path.name=='table_style.tex':continue
  s=path.read_text();s=re.sub(r'\\begin\{table\*?\}(?:\[[^]]*\])?.*?\\end\{table\*?\}',style_block,s,flags=re.S);s=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',style_block,s,flags=re.S);path.write_text(s)
 audit={'table_reference':'paper/sections/experiments.tex','colors':{'oursbg':[230,244,233],'oursgoodbg':[198,232,205],'Bures_loss':'#15803d'},'converted_nonloss_figures':['method_update_detailed','graphcurves','statescale','synthetic','noisedrift'],'source_hashes':provenance,'loss_only_graphics':True};(ROOT/'analysis/ICML_TABLE_RESTYLE.json').write_text(json.dumps(audit,indent=2)+'\n');return audit
if __name__=='__main__':build()
