#!/usr/bin/env python3
"""Resolve frozen input protections and snapshot file hashes/tree counts."""
import json,os,sqlite3,stat,sys,struct
from pathlib import Path
from safety import R,T,sha,nodes,now,under
W=T.parent.parent

def setup():
 ctx=json.loads((R/'context.json').read_text());pro=dict(ctx['protected_paths']);hashes=set(ctx['old_hash_files']);trees=set();absent={}
 # Read index directly; no Git command or metadata mutation.
 gitdir=Path((W/'.git').read_text().split(': ',1)[1].strip());raw=(gitdir/'index').read_bytes();version,count=struct.unpack('>II',raw[4:12])
 if raw[:4]!=b'DIRC' or version not in [2,3]:raise RuntimeError('unsupported Git index format')
 offset=12;tracked=[]
 for _ in range(count):
  start=offset;flags=struct.unpack('>H',raw[offset+60:offset+62])[0];offset+=62
  if version==3 and flags&0x4000:offset+=2
  end=raw.index(b'\0',offset);relative=os.fsdecode(raw[offset:end]);offset=start+((end+1-start+7)//8)*8
  target=str(W/relative);tracked.append(target);pro[target]='Git-tracked path from current worktree index'
  if os.path.isfile(target):hashes.add(target)
  elif not os.path.exists(target):absent[target]='Git-tracked path already absent before cleanup'
 (R/'tracked_paths.json').write_text(json.dumps({'index_version':version,'entry_count':count,'paths':tracked},indent=2)+'\n')
 receipt=T/'h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json'
 r5=W/'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r5.py'
 for p in [receipt,r5]:pro[str(p)]='Explicit promotion receipt or R5 reader';hashes.add(str(p))
 def strings(x):
  if isinstance(x,str):yield x
  elif isinstance(x,list):
   for v in x:yield from strings(v)
  elif isinstance(x,dict):
   for v in x.values():yield from strings(v)
 for s in strings(json.loads(receipt.read_text())):
  if s.startswith('/data/ycfeng/'):
   p=s.split('#',1)[0];pro[p]='Promotion receipt bound input';
   if os.path.isfile(p):hashes.add(p)
 for folder in [T/'h200_mainline/p2_64_formal_retry05/package_01',T/'h200_mainline/p2_64_formal_retry06/package_01',T/'h200_mainline/p2_64_reference/package_01',T/'h3o_operator_costs/formal_p2_20261102/promotion']:
  if folder.exists():
   for p,s in nodes(folder):
    if stat.S_ISREG(s.st_mode):hashes.add(p)
 # R5 helpers, current source pin and harness code used by the frozen package.
 for p in r5.parent.glob('*.py'):hashes.add(str(p))
 for p,why in list(pro.items()):
  if os.path.isfile(p):hashes.add(p)
  elif os.path.isdir(p):
   if p in ctx['worktrees'] and p not in [str(W),'/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness','/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination']:continue
   trees.add(p)
  else:absent[p]=why
 for folder in ['/data/ycfeng/tmp/glm53_dsv4/h200/monitor','/data/ycfeng/tmp/glm53_dsv4/h200/ds4_resume']:
  if os.path.isdir(folder):
   for p,s in nodes(folder):
    if stat.S_ISREG(s.st_mode) and Path(p).suffix in ['.py','.sh']:hashes.add(p)
 # Unique raw measurement roots use inventory existence + regular-file count.
 manifest=json.loads((R/'deletion_manifest.json').read_text())
 c=sqlite3.connect(R/'inventory.sqlite3')
 for (p,) in c.execute("SELECT path FROM directories WHERE path LIKE '/data/ycfeng/tmp/glm53_dsv4/h200/%/download' OR path LIKE '/data/ycfeng/tmp/glm53_dsv4/h200/%/downloads' OR path='/data/ycfeng/tmp/glm53_dsv4/h200/downloads'"):
  if any(under(p,e['path']) for e in manifest['entries']):continue
  trees.add(p);pro[p]='Unique native measurements and download evidence; never delete'
 manifest=json.loads((R/'deletion_manifest.json').read_text())
 # Every surviving profiling CSV in deletion roots stays byte-identical.
 for (p,) in c.execute("SELECT path FROM nodes WHERE kind='file' AND path LIKE '%.csv' AND (path LIKE '%/profiling/%' OR path LIKE '%/raw/%')"):
  if any(under(p,e['path']) for e in manifest['entries']):continue
  pro[p]='Profiling table or raw measurement CSV';hashes.add(p)
 manifest=json.loads((R/'deletion_manifest.json').read_text())
 for e in manifest['entries']:
  for p in e['retained']:
   if os.path.isfile(p):hashes.add(p)
  parent=Path(e['path']).parent
  for p in parent.iterdir():
   if p.is_file() and not p.is_symlink() and p.stat().st_size<1<<20 and p.suffix in ['.json','.jsonl','.yaml','.yml','.csv','.log','.xml','.sh','.py','.txt'] and str(p)!=e['path']:hashes.add(str(p))
 # Protect any new labels created by the lead since the inventory.
 for label in Path('/data/ycfeng/tmp/ds4_runs').iterdir():
  if label.is_dir() and not label.is_symlink():
   st=label.stat()
   if max(st.st_mtime_ns,st.st_ctime_ns)>=ctx['cutoff_ns']:
    pro[str(label)]='New/recent Pi label; entire tree protected by 05:50Z cutoff';trees.add(str(label))
 # Credentials stay outside deletion roots; digest only, never contents.
 for p in list(hashes):
  if not os.path.isfile(p):absent[p]='Previously protected file already absent before this cleanup';hashes.remove(p)
 for e in manifest['entries']:
  overlaps=[p for p in set(pro)|hashes|trees if under(p,e['path']) or under(e['path'],p)]
  if overlaps:raise RuntimeError('selected entry overlaps newly resolved protection: '+e['path']+' '+str(overlaps[:5]))
 out={'protected_paths':pro,'hash_files':sorted(hashes),'count_trees':sorted(trees),'worktree_existence':ctx['worktrees'],'absent_before':absent,'note':'All registered worktrees guarded by deletion-root exclusion and existence. File counts cover relevant source/harness, environments, current labels, baseline, formal replay, unique native measurement roots; counts include regular files without following symlinks.'}
 (R/'protection_plan.json').write_text(json.dumps(out,indent=2)+'\n');print('protection plan',len(hashes),'hashes',len(trees),'tree counts',len(absent),'pre-existing absent')

def snapshot(name):
 plan=json.loads((R/'protection_plan.json').read_text());out={'observed_utc':now(),'files':{},'trees':{},'worktrees':{},'errors':[]}
 for p in plan['hash_files']:
  try:out['files'][p]={'sha256':sha(p),'size':os.stat(p).st_size}
  except OSError as e:out['errors'].append({'path':p,'error':str(e)})
 # Avoid repeated walks of nested protected trees by collecting parent counts once.
 maximal=[p for p in plan['count_trees'] if not any(p!=q and under(p,q) for q in plan['count_trees'])]
 counts={p:0 for p in plan['count_trees']};exists={p:os.path.isdir(p) for p in counts}
 for root in maximal:
  try:
   for p,s in nodes(root):
    if str(R)==p:continue
    if stat.S_ISREG(s.st_mode):
     if not under(p,str(R)):
      ancestor=os.path.dirname(p)
      while ancestor!='/':
       if ancestor in counts:counts[ancestor]+=1
       ancestor=os.path.dirname(ancestor)
  except OSError as e:out['errors'].append({'path':root,'error':str(e)})
 for p,n in counts.items():out['trees'][p]={'exists':exists[p],'regular_file_count':n}
 for p in plan['worktree_existence']:out['worktrees'][p]={'exists':os.path.isdir(p)}
 (R/name).write_text(json.dumps(out,indent=2)+'\n');print(name,'hashed',len(out['files']),'counted',len(out['trees']),'errors',len(out['errors']),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='setup':setup()
 else:snapshot(sys.argv[1])
