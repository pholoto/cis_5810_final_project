"""Expand CompEval with balanced pairs, original bytes, and a repository-wide image budget."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
from pathlib import Path
import random
import time
import warnings
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif'}

def image_bytes():
    return sum(p.stat().st_size for directory in ['data','samples','reports','assets'] for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)

def fetch(index):
    # Viewer quota is much tighter than HF file downloads. Throttle globally
    # (one worker), and retry the SAME row on 429 rather than skipping it.
    for attempt in range(12):
        time.sleep(3.5)
        try:
            r=requests.get('https://datasets-server.huggingface.co/rows',params={'dataset':'OwensLab/CommunityForensics-Eval','config':'default','split':'CompEval','offset':index,'length':1},timeout=(15,45))
            if r.status_code==429:
                retry=r.headers.get('Retry-After','120')
                delay=max(60,min(int(retry) if retry.isdigit() else 120,900))
                print('RATE_LIMIT row',index,'retry_in_seconds',delay,'attempt',attempt+1,flush=True)
                time.sleep(delay)
                continue
            if r.status_code!=200:return index,None,f'HTTP {r.status_code}'
            item=r.json()['rows'][0]
            if item.get('truncated_cells'):return index,None,'truncated viewer cells'
            return index,item['row'],None
        except requests.RequestException as exc:
            if attempt>=2:return index,None,str(exc)
            time.sleep(10)
    raise RuntimeError('Persistent HF rate limit; selected images retained, evaluation not started')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--max-image-bytes',type=int,default=950_000_000)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--scan-limit',type=int,default=5000)
    p.add_argument('--output',type=Path,default=ROOT/'data/community_1gb')
    a=p.parse_args()
    if a.count<24 or a.count%2:p.error('count must be even and at least 24')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'sampling.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=a.output/'manifest.jsonl'
    seed_manifest=ROOT/'data/community_smoke24/manifest.jsonl'
    if not manifest.exists():manifest.write_text(seed_manifest.read_text())
    records=[json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(records)%2==0
    labels=Counter(r['label'] for r in records)
    assert labels[0]==labels[1]
    seen={r['sha256'] for r in records};indices_seen={r['row_index'] for r in records}
    groups=Counter((r['label'],r['group']) for r in records)
    group_bytes=Counter()
    for row in records:group_bytes[(row['label'],row['group'])]+=row['bytes']
    stored=image_bytes()
    if len(records)>=a.count:raise RuntimeError('Target already reached; use the existing manifest')
    if stored>=a.max_image_bytes:raise RuntimeError('Existing image storage already meets the budget')
    revision=records[0]['dataset_revision']
    candidates=random.Random(a.seed).sample(range(51836),a.scan_limit)
    pending={0:[],1:[]};errors=Counter();scanned=0
    stop_reason='scan_limit'
    config={'target':a.count,'max_image_bytes':a.max_image_bytes,'seed':a.seed,'scan_limit':a.scan_limit,'dataset_revision':revision,'sampling':'seed 24 reused by path; seeded random rows; balanced pairs; max 40 per generator, 150 per real source, 100 MB per group; unavailable rows skipped; budget-limited, not a uniform benchmark'}
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('START',json.dumps(config),'existing_image_bytes',stored,flush=True)
    with ThreadPoolExecutor(max_workers=1) as pool, manifest.open('a') as dest, (a.output/'skipped.jsonl').open('a') as skips:
        for start in range(0,len(candidates),8):
            batch=[i for i in candidates[start:start+8] if i not in indices_seen]
            for index,row,error in pool.map(fetch,batch):
                scanned+=1
                if error:
                    errors[error]+=1;skips.write(json.dumps({'row_index':index,'reason':error})+'\n');skips.flush();continue
                label=int(row['label']);group=str(row['model_name'] if label else row['real_source'])
                if row['split']!='test' or len(pending[label])>=4:continue
                limit=40 if label else 150
                if groups[(label,group)]+sum(r[0]['group']==group for r in pending[label])>=limit:continue
                try:
                    blob=base64.b64decode(row['image_data'],validate=True)
                    if group_bytes[(label,group)]+len(blob)+sum(len(r[1]) for r in pending[label] if r[0]['group']==group)>100_000_000:continue
                    digest=hashlib.sha256(blob).hexdigest()
                    if digest in seen or any(digest==r[0]['sha256'] for queue in pending.values() for r in queue):continue
                    with warnings.catch_warnings(record=True) as caught, Image.open(io.BytesIO(blob)) as image:
                        image.load();width,height=image.size;ext=image.format.lower()
                        if min(width,height)<32:continue
                    record={k:v for k,v in row.items() if k!='image_data'}
                    record.update(sha256=digest,label=label,group=group,dataset='OwensLab/CommunityForensics-Eval',dataset_revision=revision,dataset_split='CompEval',row_index=index,width=width,height=height,bytes=len(blob),decode_warnings=[str(w.message) for w in caught])
                    pending[label].append((record,blob,ext))
                except Exception as exc:
                    skips.write(json.dumps({'row_index':index,'reason':str(exc)})+'\n');continue
                if not pending[0] or not pending[1]:continue
                pair=[pending[0][0],pending[1][0]]
                pair_bytes=sum(len(item[1]) for item in pair)
                if stored+pair_bytes>a.max_image_bytes:
                    stop_reason='image_byte_budget';break
                # Never store an unbalanced selected pair, and preserve original bytes.
                for record,blob,ext in pair:
                    path=a.output/f'{record["sha256"]}.{ext}'
                    if path.exists():
                        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
                        stored-=len(blob)  # orphan from an interrupted manifest append
                    else:
                        path.write_bytes(blob)
                    record['path']=str(path.relative_to(ROOT))
                    records.append(record);seen.add(record['sha256']);labels[record['label']]+=1
                    groups[(record['label'],record['group'])]+=1;group_bytes[(record['label'],record['group'])]+=len(blob)
                dest.write(''.join(json.dumps(item[0])+'\n' for item in pair));dest.flush()
                pending[0].pop(0);pending[1].pop(0);stored+=pair_bytes
                print('SAVED',len(records),'labels',dict(labels),'total_image_MB',round(stored/1e6,2),'scanned',scanned,flush=True)
                if len(records)>=a.count:stop_reason='target_count';break
            if stop_reason in ('target_count','image_byte_budget'):break
    actual=image_bytes()
    assert actual<=a.max_image_bytes,(actual,a.max_image_bytes)
    summary={**config,'count':len(records),'labels':dict(labels),'unique_images':len(seen),'total_repository_image_bytes':actual,'selected_original_bytes':sum(r['bytes'] for r in records),'stop_reason':stop_reason,'scanned':scanned,'http_errors':dict(errors),'groups':[{'label':k[0],'group':k[1],'count':v,'bytes':group_bytes[k]} for k,v in groups.items()],'balanced':labels[0]==labels[1]}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DONE',json.dumps(summary),flush=True)
    if len(records)<=24:raise RuntimeError('No expansion obtained')

if __name__=='__main__':main()
