"""Save real corpus excerpts with disjoint documents and immutable hashes.
These byte-level subsets serve screening, never stand in for a GPT-2 benchmark.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,requests
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
corpora={'wikitext103':('Salesforce/wikitext','wikitext-103-raw-v1'),
'openwebtext':('Skylion007/openwebtext','plain_text'),'c4':('allenai/c4','en')}
for name,(repo,config) in corpora.items():
    try:
        docs=[];chunks=[]
        for off in range(0,1000,100):
            r=requests.get('https://datasets-server.huggingface.co/rows',params=dict(dataset=repo,config=config,split='train',offset=off,length=100),timeout=90)
            for retry in range(6):
                if r.status_code==200:break
                time.sleep(min(15,2**retry));r=requests.get(r.url,timeout=90)
            r.raise_for_status();rows=r.json()['rows']
            docs.extend([x['row']['text'] for x in rows if x['row']['text'].strip()])
        # Whole documents are assigned before concatenation. No shared token windows.
        splits={'train':docs[100:],'validation':docs[:50],'test':docs[50:100]}
        arrays={k:np.frombuffer(('\n'.join(v)).encode('utf-8'),dtype=np.uint8) for k,v in splits.items()}
        assert all(len(v)>128 for v in arrays.values())
        np.savez_compressed(root/f'{name}.npz',**arrays)
        (root/f'{name}.json').write_text(json.dumps(dict(repo=repo,config=config,source='datasets-server /rows, train offsets 0:1000',documents={k:len(v) for k,v in splits.items()},bytes={k:len(v) for k,v in arrays.items()},sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}),indent=2))
        print(name,{k:len(v) for k,v in arrays.items()},flush=True)
    except Exception as exc:print(name,'FAILED',repr(exc),flush=True)
