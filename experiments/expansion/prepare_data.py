"""Fetch official split files; train-only word vocabulary; persist exact hashes and scope."""
import collections, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np
import requests
ROOT=Path(__file__).resolve().parents[2];root=ROOT/'data/expansion';root.mkdir(parents=True,exist_ok=True)
revision='b08601e04326c79dfdd32d625aee71d232d685c3'
def fetch(url,path):
    if not path.exists():
        r=requests.get(url,timeout=180);r.raise_for_status();tmp=path.with_suffix('.download');tmp.write_bytes(r.content);tmp.replace(path)
    return path
for color in ['red','white']:
    fetch(f'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{color}.csv',root/f'wine_{color}.csv')
print('Wine Quality ready',flush=True)
if '--wine-only' in sys.argv:sys.exit(0)
import pyarrow.parquet as pq
for task,config,cap in [('wikitext2','wikitext-2-raw-v1',None),('wikitext103','wikitext-103-raw-v1',2000000)]:
    dest=root/'word';dest.mkdir(exist_ok=True)
    if (dest/f'{task}.npz').exists() and (dest/f'{task}.json').exists():
        old=json.loads((dest/f'{task}.json').read_text())
        if old.get('revision')==revision and all(old.get('tokens',{}).get(k,0)>33 for k in ['train','validation','test']):continue
    splits={};sources={}
    for split in ['train','validation','test']:
        files=requests.get(f'https://huggingface.co/api/datasets/Salesforce/wikitext/tree/{revision}/{config}',timeout=90);files.raise_for_status()
        items=sorted(x['path'] for x in files.json() if x['path'].split('/')[-1].startswith(split+'-') and x['path'].endswith('.parquet'))
        assert items,(task,split,'official split files missing at pinned revision')
        tokens=[];limit=cap if split=='train' else None;sources[split]=[]
        for name in items:
            path=fetch(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/{name}',dest/Path(name).name.replace('.parquet',f'-{task}-{revision[:8]}.parquet'))
            sources[split].append(dict(url=f'https://huggingface.co/datasets/Salesforce/wikitext/blob/{revision}/{name}',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['text']):
                for text in batch.column(0).to_pylist():tokens.extend(text.split()+['<eos>'])
                if limit and len(tokens)>=limit:break
            if limit and len(tokens)>=limit:break
        splits[split]=tokens[:limit] if limit else tokens
    counter=collections.Counter(splits['train']);vocab=['<unk>']+[w for w,n in counter.most_common(8191) if w!='<unk>'];lookup={w:i for i,w in enumerate(vocab)}
    arrays={k:np.fromiter((lookup.get(w,0) for w in tok),dtype=np.int32) for k,tok in splits.items()}
    assert all(len(v)>33 for v in arrays.values()),(task,'empty train/validation/test split')
    tmp=dest/f'{task}.building.npz';np.savez_compressed(tmp,**arrays);tmp.replace(dest/f'{task}.npz')
    meta=dict(source='official Salesforce WikiText split files',revision=revision,tokenizer='whitespace + <eos>; top8191 train-only vocabulary + <unk>',vocab_size=len(vocab),tokens={k:len(v) for k,v in arrays.items()},oov_fraction={k:float((v==0).mean()) for k,v in arrays.items()},sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},sources=sources,training_cap=cap,scope='full WikiText-2 splits' if cap is None else 'first 2M training word tokens; full official validation/test; not full WikiText-103')
    (dest/f'{task}.json').write_text(json.dumps(meta,indent=2,allow_nan=False));(dest/f'{task}_vocab.json').write_text(json.dumps(vocab));print(task,meta['tokens'],flush=True)
