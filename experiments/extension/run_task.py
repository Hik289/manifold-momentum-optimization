"""Audited extension runner. Screening and full budgets are separate metadata.
No test metric is used to choose an optimizer, hyperparameter, or checkpoint.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bures.optimizer import MatrixOptimizer, ShampooRef
from pipeline.diagnostics import DiagnosticRecorder

def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False

def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj,float) and not math.isfinite(obj): return None
    return obj

def digest(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def optimizer(model,a):
    if a.method=='adamw':return torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd,betas=(.9,.99))
    if a.method=='sgdm':return torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=a.wd)
    if a.method=='shampoo':return ShampooRef(model.parameters(),lr=a.lr,weight_decay=a.wd,update_freq=a.tau,block_size=a.block)
    return MatrixOptimizer(model.parameters(),lr=a.lr,weight_decay=a.wd,
        covariance_rule=a.method,betas=(0 if a.ablation=='no_momentum' else .9,a.beta2),
        block_size=a.block,tau_root=a.tau,ablation=a.ablation,
        s_min=a.smin,s_max=a.smax)

class MLP(nn.Module):
    def __init__(self,di,do,h=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(di,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,do))
    def forward(self,x):return self.net(x)

class ByteLM(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(256,64);self.pos=nn.Embedding(128,64)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True,norm_first=True) for _ in range(2)])
        self.out=nn.Linear(64,256)
    def forward(self,x):
        h=self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        for layer in self.layers:h=layer(h,src_mask=mask)
        return self.out(h)

def get_tabular(task,root):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    regression=task=='california'
    if task=='california':
        from sklearn.datasets import fetch_california_housing
        X,y=fetch_california_housing(return_X_y=True)
    elif task=='covertype':
        from sklearn.datasets import fetch_covtype
        X,y=fetch_covtype(return_X_y=True);y=y-1
    elif task=='adult':
        import pandas as pd, requests
        root.mkdir(parents=True,exist_ok=True)
        p=root/'adult.data'
        if not p.exists():
            r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',timeout=90);r.raise_for_status();p.write_bytes(r.content)
        df=pd.read_csv(p,header=None,skipinitialspace=True,na_values='?').dropna(subset=[14])
        X=df.drop(columns=[14]);y=(df[14].str.strip()=='>50K').to_numpy().astype('int64')
    else: raise ValueError(task)
    idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.2,random_state=42,stratify=None if regression else y)
    tr,va=train_test_split(tr,test_size=.125,random_state=42,stratify=None if regression else y[tr])
    if task=='adult':
        nums=list(X.select_dtypes(include='number').columns);cats=[c for c in X.columns if c not in nums]
        trans=ColumnTransformer([('num',make_pipeline(SimpleImputer(),StandardScaler()),nums),
            ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)])
        xt=trans.fit_transform(X.iloc[tr]);xv=trans.transform(X.iloc[va]);xe=trans.transform(X.iloc[te])
    else:
        scaler=StandardScaler().fit(X[tr]);xt=scaler.transform(X[tr]);xv=scaler.transform(X[va]);xe=scaler.transform(X[te])
    if regression:
        mean,std=y[tr].mean(),y[tr].std();y=(y-mean)/std
    else:std=1.
    do=1 if regression else int(y.max()+1)
    data=[(torch.tensor(x,dtype=torch.float32),torch.tensor(y[i],dtype=torch.float32 if regression else torch.long)) for x,i in [(xt,tr),(xv,va),(xe,te)]]
    return data,regression,do,float(std),dict(train=len(tr),validation=len(va),test=len(te),split_sha256=hashlib.sha256(np.concatenate([tr,va,te]).tobytes()).hexdigest())

def setup(a,device):
    root=Path(a.data)
    if a.family=='graph':
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        from stage1.models import SmallGCN
        ds=Planetoid(str(root/'graph'/a.task),name=a.task,transform=NormalizeFeatures());d=ds[0].to(device)
        model=SmallGCN(ds.num_features,64,ds.num_classes).to(device)
        def train_loss(epoch):
            return F.cross_entropy(model(d.x,d.edge_index)[d.train_mask],d.y[d.train_mask])
        @torch.no_grad()
        def evaluate(split,epoch):
            mask=getattr(d,split+'_mask');logits=model(d.x,d.edge_index)[mask]
            return float(F.cross_entropy(logits,d.y[mask])),float((logits.argmax(-1)==d.y[mask]).float().mean())
        return model,[(None,None)],train_loss,evaluate,True,dict(nodes=d.num_nodes,train=int(d.train_mask.sum()))
    if a.family=='tabular':
        data,reg,do,ystd,meta=get_tabular(a.task,root/'tabular');data=[(x.to(device),y.to(device)) for x,y in data]
        model=MLP(data[0][0].shape[1],do).to(device)
        def loss(x,y):return F.mse_loss(model(x).squeeze(-1),y) if reg else F.cross_entropy(model(x),y)
        @torch.no_grad()
        def evaluate(split,epoch):
            x,y=data[1 if split=='val' else 2];total=0.;good=0.
            for off in range(0,len(x),2048):
                xx,yy=x[off:off+2048],y[off:off+2048];o=model(xx)
                total+=float(loss(xx,yy))*len(xx)
                if not reg:good+=int((o.argmax(-1)==yy).sum())
            return total/len(x),math.sqrt(total/len(x))*ystd if reg else good/len(x)
        return model,data,loss,evaluate,not reg,meta
    if a.family=='shift':
        from torchvision.datasets import MNIST
        from torchvision.transforms.functional import rotate
        tr=MNIST(str(root/'vision'),train=True,download=True);te=MNIST(str(root/'vision'),train=False,download=True)
        # Fixed train/validation split; every epoch visits all 50k training images.
        perm=np.random.default_rng(42).permutation(60000)
        splits=[(tr.data[perm[:50000]],tr.targets[perm[:50000]]),(tr.data[perm[50000:]],tr.targets[perm[50000:]]),(te.data,te.targets)]
        data=[(x.float().unsqueeze(1).to(device)/255,y.to(device)) for x,y in splits]
        model=MLP(784,10).to(device)
        def angle(epoch):return ((epoch-1)//100%4)*30
        def loss(x,y):return F.cross_entropy(model(rotate(x,angle(a.current_epoch)).flatten(1)),y)
        @torch.no_grad()
        def evaluate(split,epoch):
            x,y=data[1 if split=='val' else 2];total=0.;correct=0
            for off in range(0,len(x),2048):
                xx=rotate(x[off:off+2048],angle(epoch)).flatten(1);yy=y[off:off+2048];o=model(xx)
                total+=float(F.cross_entropy(o,yy))*len(xx);correct+=int((o.argmax(-1)==yy).sum())
            return total/len(x),correct/len(x)
        return model,data,loss,evaluate,True,dict(train=50000,validation=10000,test=10000,angle_every_100_epochs=[0,30,60,90])
    if a.family=='sciml':
        from stage1.models import PINNNet
        from stage1.train_sciml import poisson_loss,burgers_loss,poisson_reference,burgers_reference
        model=PINNNet(1 if a.task=='poisson' else 2).to(device)
        fn=poisson_loss if a.task=='poisson' else burgers_loss
        x,y=(poisson_reference(device,1024) if a.task=='poisson' else burgers_reference(device,nx=256,dt=2.5e-4))
        gen=torch.Generator();gen.manual_seed(42)
        def loss(epoch):return fn(model,256,gen,device)[0]
        @torch.no_grad()
        def evaluate(split,epoch):
            parity=0 if split=='val' else 1;xx,yy=x[parity::2],y[parity::2]
            pred=model(xx);mse=float((pred-yy).square().mean());rel=float((pred-yy).norm()/(yy.norm()+1e-12))
            return mse,rel
        return model,[(None,None)],loss,evaluate,False,dict(validation_reference='disjoint even grid points',test_reference='odd grid points',reference='sin(pi*x)' if a.task=='poisson' else 'first-order upwind FD, nx=256, dt=2.5e-4; approximate reference',gradient_clip=1.,scope='2000 independent collocation updates')
    if a.family=='vision':
        from torchvision.datasets import CIFAR10,CIFAR100
        from torchvision import transforms
        from stage1.models import ResNet18Small
        cls=CIFAR10 if a.task=='cifar10' else CIFAR100
        tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
        train=cls(str(root/'vision'),train=True,download=True,transform=tf);test=cls(str(root/'vision'),train=False,download=True,transform=tf)
        # Stored tensors avoid loader-worker randomness; augmentations are intentionally absent.
        gen=np.random.default_rng(42);idx=gen.permutation(50000);X=torch.tensor(train.data.transpose(0,3,1,2),dtype=torch.float32,device=device)/255
        Xe=torch.tensor(test.data.transpose(0,3,1,2),dtype=torch.float32,device=device)/255
        mean=torch.tensor([.4914,.4822,.4465],device=device)[None,:,None,None];std=torch.tensor([.247,.2435,.2616],device=device)[None,:,None,None]
        X=(X-mean)/std;Xe=(Xe-mean)/std;y=torch.tensor(train.targets,device=device);ye=torch.tensor(test.targets,device=device)
        data=[(X[idx[:45000]],y[idx[:45000]]),(X[idx[45000:]],y[idx[45000:]]),(Xe,ye)]
        model=ResNet18Small(10 if a.task=='cifar10' else 100).to(device)
        def loss(x,y):return F.cross_entropy(model(x),y)
        @torch.no_grad()
        def evaluate(split,epoch):
            x,y=data[1 if split=='val' else 2];total=0.;correct=0
            for off in range(0,len(x),256):
                xx,yy=x[off:off+256],y[off:off+256];o=model(xx);total+=float(F.cross_entropy(o,yy))*len(xx);correct+=int((o.argmax(-1)==yy).sum())
            return total/len(x),correct/len(x)
        return model,data,loss,evaluate,True,dict(train=45000,validation=5000,test=10000,model='ResNet18Small',augmentations='none',split_sha256=hashlib.sha256(idx.tobytes()).hexdigest())
    if a.family=='lm':
        datafile=root/'lm'/f'{a.task}.npz'
        if not datafile.exists():raise FileNotFoundError('Run prepare_text.py before language screening')
        d=np.load(datafile);data={k:torch.tensor(d[k].astype('int64'),device=device) for k in d.files}
        model=ByteLM().to(device)
        def pairs(tokens,starts):
            idx=starts[:,None]+torch.arange(65,device=device)[None,:];seq=tokens[idx];return seq[:,:-1],seq[:,1:]
        gen=torch.Generator(device=device);gen.manual_seed(42)
        def loss(epoch):
            starts=torch.randint(len(data['train'])-65,(16,),device=device,generator=gen)
            x,y=pairs(data['train'],starts);return F.cross_entropy(model(x).flatten(0,1),y.flatten())
        @torch.no_grad()
        def evaluate(split,epoch):
            tok=data['validation' if split=='val' else 'test'];starts=torch.arange(0,min(len(tok)-65,8192),64,device=device)
            total=0.;n=0
            for off in range(0,len(starts),16):
                x,y=pairs(tok,starts[off:off+16]);total+=float(F.cross_entropy(model(x).flatten(0,1),y.flatten()))*y.numel();n+=y.numel()
            v=total/n;return v,math.exp(min(50,v))
        return model,[(None,None)],loss,evaluate,False,dict(tokenizer='UTF-8 bytes, vocab=256',train_tokens=len(data['train']),tokens_per_step=1024,scope='small-model screening; not GPT-2/WikiText full benchmark')
    raise ValueError(a.family)

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--task',required=True)
    p.add_argument('--method',default='bures');p.add_argument('--epochs',type=int,default=1000);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--wd',type=float,default=1e-4);p.add_argument('--block',type=int,default=256);p.add_argument('--tau',type=int,default=10)
    p.add_argument('--beta2',type=float,default=.99);p.add_argument('--ablation',default='full');p.add_argument('--smin',type=float,default=.1);p.add_argument('--smax',type=float,default=10)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--diag-every',type=int,default=20)
    p.add_argument('--stage',choices=['screen','full'],default='full');p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.family in ('lm','tabular'):a.device='cpu'
    seed_all();torch.set_num_threads(2);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    run_lock=out.with_suffix('.runlock').open('w')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Run already active:',out,flush=True);return
    if out.exists():
        try:
            if json.loads(out.read_text()).get('status') in ('complete','failed'):return
        except json.JSONDecodeError:pass
    meta={**vars(a),'seed':42,'schema_version':2,'status':'running','torch':torch.__version__,'history':[],'diagnostics':[],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'bures/optimizer.py').read_bytes()).hexdigest()}
    atomic(out,meta);start=time.perf_counter()
    try:
        model,data,lossfn,ev,maximize,data_meta=setup(a,device);opt=optimizer(model,a);rec=DiagnosticRecorder(opt)
        meta.update(data_meta=data_meta,initialization_sha256=digest(model),parameters=sum(p.numel() for p in model.parameters()),metric_direction='max' if maximize else 'min')
        best=-float('inf') if maximize else float('inf');best_epoch=0;best_weights=None;previous=None;steps=0
        for epoch in range(1,a.epochs+1):
            a.current_epoch=epoch;model.train();loss_total=0.;seen=0;raw_norm=0.;jitter=0.
            if a.family in ('graph','lm','sciml'):batches=[(None,None)]
            else:
                gen=torch.Generator(device=device);gen.manual_seed(42+epoch);idx=torch.randperm(len(data[0][0]),device=device,generator=gen)
                batches=[(data[0][0][i],data[0][1][i]) for i in idx.split(a.batch)]
            for x,y in batches:
                opt.zero_grad(set_to_none=True);loss=lossfn(epoch) if x is None else lossfn(x,y)
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at epoch {epoch}')
                loss.backward()
                if a.family=='sciml':torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                raw_norm=math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
                before=[p.detach().clone() for p in model.parameters()] if epoch%a.diag_every==0 else None
                opt.step();steps+=1
                if before is not None:
                    delta=torch.cat([(p.detach()-v).flatten() for p,v in zip(model.parameters(),before)])
                    if previous is not None:jitter=float((delta-previous).norm())
                    previous=delta
                n=1 if x is None else len(x);loss_total+=float(loss.detach())*n;seen+=n
            model.eval();vl,metric=ev('val',epoch)
            if not math.isfinite(vl):raise FloatingPointError('nonfinite validation loss')
            if (metric>best if maximize else metric<best):
                best=metric;best_epoch=epoch;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(epoch=epoch,step=steps,train_loss=loss_total/seen,val_loss=vl,val_metric=metric,wall_seconds=time.perf_counter()-start,raw_gradient_norm=raw_norm,update_jitter=jitter)
            meta['history'].append(row)
            if epoch%a.diag_every==0 or epoch==1 or epoch==a.epochs:
                rec.record(epoch,train_loss=row['train_loss'],val_metric=metric,learning_rate=a.lr,peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else None,checkpoint_meta={'epoch':epoch,'source_schema':2})
                meta['diagnostics']=clean(rec.trajectory);meta['eigh_events']=clean(__import__('bures.core',fromlist=['EIGH_FALLBACK_STATS']).EIGH_FALLBACK_STATS)
                atomic(out,clean(meta));print(a.task,a.method,a.ablation,epoch,vl,metric,round(row['wall_seconds'],1),flush=True)
            if epoch%100==0:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,metadata=vars(a)),out.with_suffix('.checkpoint.pt'))
        model.eval();tl,tm=ev('test',a.epochs);meta['final']=dict(epoch=a.epochs,test_loss=tl,test_metric=tm,val_metric=meta['history'][-1]['val_metric'])
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=a.epochs,metadata=vars(a)),out.with_suffix('.checkpoint.pt'))
        model.load_state_dict(best_weights);model.eval();bl,bm=ev('test',best_epoch)
        meta['validation_selected']=dict(epoch=best_epoch,val_metric=best,test_loss=bl,test_metric=bm)
        meta.update(status='complete',wall_seconds=time.perf_counter()-start,peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else None)
    except Exception as e:
        meta.update(status='failed',error=repr(e),traceback=traceback.format_exc());print(meta['traceback'],flush=True)
    atomic(out,clean(meta))
    if meta['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
