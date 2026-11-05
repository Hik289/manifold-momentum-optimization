"""Prepare the six fixed new tasks; preserve raw source hashes and disjoint splits."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import sklearn
from scipy.io import loadmat
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'data/new_tasks';RAW=DEST/'raw';RAW.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(url,name,expected=None):
 p=RAW/name
 if not p.exists():
  res=requests.get(url,timeout=(20,180));res.raise_for_status();q=p.with_suffix(p.suffix+'.download');q.write_bytes(res.content);q.replace(p)
 if expected:assert sha(p)==expected,(name,'source checksum')
 return p,{'url':url,'sha256':sha(p),'bytes':p.stat().st_size}
def write(task,arrays,meta):
 for key,arr in arrays.items():
  assert np.isfinite(arr).all(),(task,key,'nonfinite input')
 p=DEST/f'{task}.npz';np.savez_compressed(p,**arrays)
 meta.update(task=task,prepared_sha256=sha(p),arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()},split_seed=42,sklearn_version=sklearn.__version__)
 (DEST/f'{task}.json').write_text(json.dumps(meta,indent=2)+'\n');print(task,{k:list(v.shape) for k,v in arrays.items() if k.endswith('_x') or k.endswith('_series')},flush=True)
def standardize(parts):
 axes=(0,1) if parts[0].ndim==3 else (0,);mean=parts[0].mean(axis=axes,keepdims=True);std=parts[0].std(axis=axes,keepdims=True).clip(1e-6)
 return [((v-mean)/std).astype('float32') for v in parts],{'mean':mean.reshape(-1).tolist(),'std':std.reshape(-1).tolist(),'fitted_on':'training only'}
def ett(task):
 name='ETTh1' if task=='etth1' else 'ETTh2';revision='1d16c8f4f943005d613b5bc962e9eeb06058cf07';p,source=fetch(f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{revision}/ETT-small/{name}.csv',name+'.csv');df=pd.read_csv(p);x=df.drop(columns='date').to_numpy(dtype='float32');assert x.shape[1]==7 and len(x)>=14400
 mean=x[:8640].mean(0);std=x[:8640].std(0).clip(1e-6);z=(x-mean)/std;L=96
 arrays={'train_series':z[:8640].astype('float32'),'val_series':z[8640-L:11520].astype('float32'),'test_series':z[11520-L:14400].astype('float32')}
 write(task,arrays,{'family':'forecasting','sources':[source],'revision':revision,'columns':list(df.columns[1:]),'input_length':L,'horizon':24,'input_dim':7,'output_dim':7,'target_ranges':{'train':[96,8640],'validation':[8640,11520],'test':[11520,14400]},'evaluation_target_stride':24,'normalization':{'mean':mean.tolist(),'std':std.tolist(),'fit_rows':[0,8640]},'scope':'first 20 benchmark months; seven-variable 24-hour forecasting, with train-only scaling and nonoverlapping evaluation targets'})
def har():
 p,source=fetch('https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip','uci_har.zip');z=zipfile.ZipFile(p)
 if not any(n.endswith('train/Inertial Signals/body_acc_x_train.txt') for n in z.namelist()):
  inner=next(n for n in z.namelist() if n.lower().endswith('.zip'));z=zipfile.ZipFile(io.BytesIO(z.read(inner)))
 def file(suffix):return io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(suffix))))
 names=['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z'];parts={}
 for split in ['train','test']:
  xx=np.stack([np.loadtxt(file(f'{split}/Inertial Signals/{n}_{split}.txt'),dtype='float32') for n in names],axis=-1);yy=np.loadtxt(file(f'{split}/y_{split}.txt'),dtype='int64')-1;subjects=np.loadtxt(file(f'{split}/subject_{split}.txt'),dtype='int64');parts[split]=(xx,yy,subjects)
 x,y,subjects=parts['train'];te,ty,ts=parts['test'];ids=np.random.default_rng(42).permutation(np.unique(subjects));vsub=ids[:4];vm=np.isin(subjects,vsub);xs,norm=standardize([x[~vm],x[vm],te]);sets=[set(subjects[~vm]),set(subjects[vm]),set(ts)];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
 arrays={}
 for name,xx,yy,ss in [('train',xs[0],y[~vm],subjects[~vm]),('val',xs[1],y[vm],subjects[vm]),('test',xs[2],ty,ts)]:arrays.update({name+'_x':xx,name+'_y':yy,name+'_subjects':ss})
 write('uci_har',arrays,{'family':'sequence_classification','sources':[source],'input_dim':9,'classes':6,'normalization':norm,'subjects':{k:sorted(int(v) for v in group) for k,group in zip(['train','validation','test'],sets)},'scope':'native 128-step nine-channel inertial sequences; official subject-held-out test, four official training subjects held out for validation'})
def forda():
 p,source=fetch('https://timeseriesclassification.com/aeon-toolkit/FordA.zip','FordA.zip');z=zipfile.ZipFile(p)
 def parse(split):
  candidates=[n for n in z.namelist() if ('_'+split+'.') in n.upper()]
  def rank(n):return next((i for i,e in enumerate(['.tsv','.txt','.ts','.arff']) if n.lower().endswith(e)),99)
  name=sorted(candidates,key=rank)[0];text=z.read(name).decode('utf-8-sig');ext=Path(name).suffix.lower()
  if ext in ('.txt','.tsv'):
   a=np.loadtxt(io.StringIO(text));y=a[:,0];x=a[:,1:]
  elif ext=='.ts':
   rows=[];ys=[];active=False
   for l in text.splitlines():
    l=l.strip()
    if l.lower()=='@data':active=True;continue
    if not active or not l or l.startswith('#'):continue
    a,b=l.rsplit(':',1);rows.append([float(v) for v in a.split(',')]);ys.append(float(b))
   x=np.array(rows);y=np.array(ys)
  else:
   from scipy.io import arff
   a,_=arff.loadarff(io.StringIO(text));x=np.array([list(v)[:-1] for v in a],dtype='float32');y=np.array([float(v[-1]) for v in a])
  return x.astype('float32')[:,:,None],(y>0).astype('int64')
 x,y=parse('TRAIN');te,ty=parse('TEST');assert x.shape==(3601,500,1) and te.shape==(1320,500,1)
 tr,va=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42);xs,norm=standardize([x[tr],x[va],te]);arrays={}
 for name,xx,yy in [('train',xs[0],y[tr]),('val',xs[1],y[va]),('test',xs[2],ty)]:arrays.update({name+'_x':xx,name+'_y':yy})
 arrays.update(train_indices=tr,val_indices=va);assert not set(tr)&set(va)
 write('forda',arrays,{'family':'sequence_classification','sources':[source],'input_dim':1,'classes':2,'normalization':norm,'scope':'native 500-step series; official train/test retained, stratified 20% training holdout for validation'})
def denoise():
 obj=load_digits();x=(obj.data/16).astype('float32');y=obj.target;tr,te=train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=42);tr,va=train_test_split(tr,test_size=.125,stratify=y[tr],random_state=42)
 cases=[('digits_denoise',x,tr,va,te,{'family':'denoising','sources':[{'url':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html','content_sha256':hashlib.sha256(obj.data.tobytes()).hexdigest()}],'scope':'8x8 digits with a fixed 70/10/20 split and pixel scale /16; labels used for stratification only'})]
 p,source=fetch('https://ndownloader.figshare.com/files/5976027','olivettifaces.mat','b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794');raw=loadmat(p)['faces'].T.astype('float32');assert raw.shape==(400,4096) and raw.min()>=0 and 1<raw.max()<=255
 images=(raw.reshape(400,64,64).transpose(0,2,1)/255).reshape(400,16,4,16,4).mean((2,4));xx=images.reshape(400,256);sub=np.repeat(np.arange(40),10);order=np.random.default_rng(42).permutation(40);groups=[order[:28],order[28:32],order[32:]];indices=[np.flatnonzero(np.isin(sub,g)) for g in groups]
 cases.append(('olivetti_denoise',xx,*indices,{'family':'denoising','sources':[source],'subject_ids':{k:sorted(g.tolist()) for k,g in zip(['train','validation','test'],groups)},'scope':'raw fixed /255 pixel scaling; 4x4 average pool to16x16; disjoint 28/4/8 identity groups, reconstruction only'}))
 for task,clean,tr,va,te,meta in cases:
  arrays={};rng=np.random.default_rng(20261102)
  for name,idx in [('train',tr),('val',va),('test',te)]:
   c=clean[idx].astype('float32');arrays[name+'_clean']=c;arrays[name+'_indices']=idx
   if name!='train':arrays[name+'_x']=np.clip(c+rng.normal(0,.15,c.shape),0,1).astype('float32')
  assert all(not set(a)&set(b) for a,b in [(tr,va),(tr,te),(va,te)]);meta.update(input_dim=clean.shape[1],noise_sigma=.15,heldout_noise_seed=20261102,pixel_scaling='fixed constants, not fitted on held-out images');write(task,arrays,meta)
if __name__=='__main__':
 for task in ['etth1','etth2']:ett(task)
 har();forda();denoise()
 manifests={p.stem:json.loads(p.read_text()) for p in DEST.glob('*.json')};(ROOT/'analysis/NEW_TASK_DATA_PROVENANCE.json').write_text(json.dumps(manifests,indent=2)+'\n');print('All six data bundles prepared.',flush=True)
