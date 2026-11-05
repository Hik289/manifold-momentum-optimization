"""Long-budget architecture expansion. All comparisons retain seed 42 and validation-only selection."""
import argparse, fcntl, hashlib, json, math, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from extension import run_task as base
from bures.optimizer import MatrixOptimizer
from pipeline.diagnostics import DiagnosticRecorder

class WordLM(nn.Module):
    def __init__(self,kind,hidden,vocab):
        super().__init__();self.emb=nn.Embedding(vocab,hidden);self.kind=kind
        if kind in ('lstm','gru'):
            self.core=(nn.LSTM if kind=='lstm' else nn.GRU)(hidden,hidden,2,dropout=.1,batch_first=True)
        else:
            self.core=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,2*hidden,dropout=.1,batch_first=True,norm_first=True),2)
            self.pos=nn.Embedding(32,hidden)
        self.out=nn.Linear(hidden,vocab)
    def forward(self,x):
        h=self.emb(x)
        if self.kind in ('lstm','gru'):h=self.core(h)[0]
        else:
            h=h+self.pos(torch.arange(x.shape[1],device=x.device))
            h=self.core(h,mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1))
        return self.out(h)

class GraphNet(nn.Module):
    def __init__(self,kind,di,do):
        super().__init__();from torch_geometric.nn import GATConv,SAGEConv,GCN2Conv
        self.kind=kind
        if kind=='gat':self.layers=nn.ModuleList([GATConv(di,16,heads=4,dropout=.5),GATConv(64,do,heads=1,concat=False,dropout=.5)])
        elif kind=='sage':self.layers=nn.ModuleList([SAGEConv(di,64),SAGEConv(64,do)])
        else:
            self.input=nn.Linear(di,64);self.output=nn.Linear(64,do)
            self.layers=nn.ModuleList([GCN2Conv(64,alpha=.1,theta=.5,layer=i+1,cached=True) for i in range(8)])
    def forward(self,x,edges):
        if self.kind=='gcnii':
            x=F.relu(self.input(x));x0=x
            for layer in self.layers:x=F.relu(layer(F.dropout(x,.5,self.training),x0,edges))
            return self.output(F.dropout(x,.5,self.training))
        x=F.dropout(x,.5,self.training);x=self.layers[0](x,edges)
        x=F.elu(x) if self.kind=='gat' else F.relu(x)
        return self.layers[1](F.dropout(x,.5,self.training),edges)

def setup(a,device):
    if a.family=='tabular':
        if a.task.startswith('wine_'):
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            df=pd.read_csv(Path(a.suite_data)/f'{a.task}.csv',sep=';')
            y=(df['quality'].to_numpy()>=6).astype('int64');x=df.drop(columns='quality').to_numpy();idx=np.arange(len(y))
            tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=42)
            tr,va=train_test_split(tr,test_size=.125,stratify=y[tr],random_state=42)
            scaler=StandardScaler().fit(x[tr]);data=[(torch.tensor(scaler.transform(x[i]),dtype=torch.float32,device=device),torch.tensor(y[i],device=device)) for i in [tr,va,te]]
            reg=False;do=2;ystd=1.;meta=dict(train=len(tr),validation=len(va),test=len(te),target='quality >= 6',split_sha256=hashlib.sha256(np.concatenate([tr,va,te]).tobytes()).hexdigest())
        else:
            data,reg,do,ystd,meta=base.get_tabular(a.task,Path(a.data)/'tabular')
            data=[(x.to(device),y.to(device)) for x,y in data]
        model=base.MLP(data[0][0].shape[1],do,h=a.hidden).to(device)
        def loss(x,y):return F.mse_loss(model(x).squeeze(-1),y) if reg else F.cross_entropy(model(x),y)
        return model,data,loss,not reg,ystd,meta,None
    if a.family=='graph':
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        ds=Planetoid(str(Path(a.data)/'graph'/a.task),name=a.task,transform=NormalizeFeatures());d=ds[0].to(device)
        model=GraphNet(a.model,ds.num_features,ds.num_classes).to(device)
        def loss(x,y):return F.cross_entropy(model(d.x,d.edge_index)[d.train_mask],d.y[d.train_mask])
        def evaluate(split):
            mask=getattr(d,split+'_mask');o=model(d.x,d.edge_index)[mask];y=d.y[mask]
            return float(F.cross_entropy(o,y)),float((o.argmax(-1)==y).float().mean())
        return model,None,loss,True,1.,dict(nodes=d.num_nodes,train=int(d.train_mask.sum()),layers=8 if a.model=='gcnii' else 2),evaluate
    if a.family=='lm':
        path=Path(a.suite_data)/'word'/f'{a.task}.npz';obj=np.load(path)
        tokens={k:torch.tensor(obj[k].astype('int64'),device=device) for k in ['train','validation','test']}
        assert all(len(v)>33 for v in tokens.values()),'Invalid empty word-corpus split'
        meta=json.loads(path.with_suffix('.json').read_text());model=WordLM(a.model,a.hidden,meta['vocab_size']).to(device)
        def batch(tok,starts):
            z=tok[starts[:,None]+torch.arange(33,device=device)[None,:]];return z[:,:-1],z[:,1:]
        def loss(x,y):return F.cross_entropy(model(x).flatten(0,1),y.flatten())
        def evaluate(split):
            tok=tokens['validation' if split=='val' else 'test'];total=0.;count=0
            for off in range(0,len(tok)-33,16*32):
                starts=torch.arange(off,min(off+16*32,len(tok)-33),32,device=device);x,y=batch(tok,starts)
                total+=float(loss(x,y))*y.numel();count+=y.numel()
            val=total/count;return val,math.exp(min(val,50))
        meta.update(context=32,batch=16,tokens_per_update=512,matrix_scope='recurrent core or Transformer encoder; embeddings/output use common AdamW')
        return model,tokens,loss,False,1.,meta,evaluate
    if a.family=='vision':
        from torchvision.datasets import FashionMNIST,CIFAR10,CIFAR100
        root=str(Path(a.data)/'vision');cls=FashionMNIST if a.task=='fashion_mnist' else (CIFAR10 if a.task=='cifar10' else CIFAR100)
        train=cls(root,train=True,download=True);test=cls(root,train=False,download=True)
        x=torch.as_tensor(np.asarray(train.data),device=device).float()/255;xe=torch.as_tensor(np.asarray(test.data),device=device).float()/255
        if a.task=='fashion_mnist':x=x.flatten(1);xe=xe.flatten(1)
        else:
            x=x.permute(0,3,1,2);xe=xe.permute(0,3,1,2)
            mean=x.new_tensor([.4914,.4822,.4465])[None,:,None,None];std=x.new_tensor([.247,.2435,.2616])[None,:,None,None];x=(x-mean)/std;xe=(xe-mean)/std
            if a.model=='mlp':x=x.flatten(1);xe=xe.flatten(1)
        y=torch.as_tensor(np.asarray(train.targets),device=device).long();ye=torch.as_tensor(np.asarray(test.targets),device=device).long()
        idx=np.random.default_rng(42).permutation(len(x));cut=int(.9*len(x));data=[(x[idx[:cut]],y[idx[:cut]]),(x[idx[cut:]],y[idx[cut:]]),(xe,ye)]
        do=100 if a.task=='cifar100' else 10
        if a.model=='mlp':model=base.MLP(x.shape[1],do,h=a.hidden).to(device)
        else:
            from torchvision import models
            model=(models.resnet18 if a.model=='resnet18_head' else models.resnet50)(weights=None)
            model.conv1=nn.Conv2d(3,64,3,1,1,bias=False);model.maxpool=nn.Identity();model.fc=nn.Linear(model.fc.in_features,do);model=model.to(device)
        def loss(x,y):return F.cross_entropy(model(x),y)
        meta=dict(train=cut,validation=len(x)-cut,test=len(xe),augmentations='none',split_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),matrix_scope='all MLP matrices' if a.model=='mlp' else 'classifier head; backbone uses common AdamW')
        return model,data,loss,True,1.,meta,None
    raise ValueError(a.family)

def optimizers(model,a):
    kw=dict(lr=a.lr,weight_decay=1e-4)
    if a.method=='adamw':return [torch.optim.AdamW(model.parameters(),betas=(.9,.99),**kw)]
    if a.method=='sgdm':return [torch.optim.SGD(model.parameters(),momentum=.9,**kw)]
    if a.family=='lm':chosen=[p for n,p in model.named_parameters() if n.startswith('core.') and p.ndim==2]
    elif a.model.endswith('_head'):chosen=list(model.fc.parameters())
    else:chosen=list(model.parameters())
    picked={id(p) for p in chosen};rest=[p for p in model.parameters() if id(p) not in picked]
    opt=MatrixOptimizer(chosen,covariance_rule=a.method,betas=(.9,.99),block_size=a.block,tau_root=10,s_min=.1,s_max=10,**kw)
    return [opt]+([torch.optim.AdamW(rest,betas=(.9,.99),**kw)] if rest else [])

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--device',default='cpu');p.add_argument('--data',required=True);p.add_argument('--suite-data',required=True);p.add_argument('--out')
    raw=p.parse_args();job=json.loads(Path(raw.job).read_text());a=argparse.Namespace(**job,device=raw.device,data=raw.data,suite_data=raw.suite_data)
    base.seed_all();torch.set_num_threads(1);device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.30,device);torch.cuda.reset_peak_memory_stats(device)
    out=Path(raw.out) if raw.out else ROOT/'experiments/extension/results'/f'{a.id}.json';out.parent.mkdir(parents=True,exist_ok=True)
    lock=out.with_suffix('.runlock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if out.exists() and json.loads(out.read_text()).get('status') in ('complete','failed'):return
    meta={**job,'device':str(device),'status':'running','seed':42,'schema_version':2,'cohort':'flow_expansion','stage':'full','history':[],'diagnostics':[],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'experiments/bures/optimizer.py').read_bytes()).hexdigest()}
    base.atomic(out,meta);start=time.perf_counter();best_weights=None
    try:
        model,data,lossfn,maximize,ystd,data_meta,special_ev=setup(a,device);opts=optimizers(model,a);rec=DiagnosticRecorder(opts[0]);meta.update(data_meta=data_meta,initialization_sha256=base.digest(model),parameters=sum(p.numel() for p in model.parameters()),metric_direction='max' if maximize else 'min')
        def evaluate(split):
            if special_ev:return special_ev(split)
            x,y=data[1 if split=='val' else 2];total=0.;correct=0
            for off in range(0,len(x),256 if a.family=='vision' else 2048):
                xx=x[off:off+(256 if a.family=='vision' else 2048)];yy=y[off:off+len(xx)];o=model(xx)
                total+=float((F.cross_entropy(o,yy) if maximize else F.mse_loss(o.squeeze(-1),yy)))*len(xx)
                if maximize:correct+=int((o.argmax(-1)==yy).sum())
            val=total/len(x);return val,correct/len(x) if maximize else math.sqrt(val)*ystd
        best=-math.inf if maximize else math.inf;steps=0
        for unit in range(1,a.epochs+1):
            model.train();total=0.;seen=0
            if a.family=='graph':batches=[(None,None)]
            elif a.family=='lm':
                gen=torch.Generator(device=device).manual_seed(42+unit);starts=torch.randint(len(data['train'])-33,(16,),generator=gen,device=device);z=data['train'][starts[:,None]+torch.arange(33,device=device)[None,:]];batches=[(z[:,:-1],z[:,1:])]
            else:
                gen=torch.Generator(device=device).manual_seed(42+unit);idx=torch.randperm(len(data[0][0]),generator=gen,device=device)
                batches=((data[0][0][i],data[0][1][i]) for i in idx.split(a.batch))
            for x,y in batches:
                for opt in opts:opt.zero_grad(set_to_none=True)
                loss=lossfn(x,y)
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite training loss at unit {unit}')
                loss.backward();raw_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                for opt in opts:opt.step()
                steps+=1;n=1 if x is None else len(x);total+=float(loss.detach())*n;seen+=n
            # Training objective is recorded at EVERY epoch/update, without smoothing.
            row=dict(epoch=unit,step=steps,train_loss=total/seen,wall_seconds=time.perf_counter()-start,raw_gradient_norm=raw_norm)
            if unit==1 or unit%a.eval_every==0 or unit==a.epochs:
                model.eval()
                with torch.no_grad():vl,metric=evaluate('val')
                if not math.isfinite(vl):raise FloatingPointError(f'nonfinite validation loss at unit {unit}')
                row.update(val_loss=vl,val_metric=metric)
                if metric>best if maximize else metric<best:
                    best=metric;best_epoch=unit;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if unit%a.diag_every==0 or unit==1 or unit==a.epochs:
                    rec.record(unit,train_loss=row['train_loss'],val_metric=metric,learning_rate=a.lr,checkpoint_meta={'epoch':unit,'cohort':'flow_expansion'})
                    meta['diagnostics']=base.clean(rec.trajectory)
                meta['history'].append(row);base.atomic(out,base.clean(meta));print(a.id,unit,row['train_loss'],vl,flush=True)
            else:meta['history'].append(row)
            if unit%100==0 or unit==a.epochs:
                torch.save(dict(model=model.state_dict(),optimizers=[o.state_dict() for o in opts],unit=unit,best_weights=best_weights,best_epoch=best_epoch,best=best,torch_rng=torch.get_rng_state()),out.with_suffix('.checkpoint.pt'))
        model.eval()
        with torch.no_grad():tl,tm=evaluate('test')
        meta['final']=dict(epoch=a.epochs,test_loss=tl,test_metric=tm,val_metric=meta['history'][-1]['val_metric'])
        model.load_state_dict(best_weights);model.eval()
        with torch.no_grad():tl,tm=evaluate('test')
        meta.update(validation_selected=dict(epoch=best_epoch,val_metric=best,test_loss=tl,test_metric=tm),status='complete',wall_seconds=time.perf_counter()-start,peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else None)
    except Exception as exc:meta.update(status='failed',error=repr(exc),traceback=traceback.format_exc());print(meta['traceback'],flush=True)
    base.atomic(out,base.clean(meta))
    if meta['status']=='failed':sys.exit(1)
if __name__=='__main__':main()
