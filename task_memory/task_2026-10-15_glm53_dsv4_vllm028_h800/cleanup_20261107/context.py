#!/usr/bin/env python3
"""Read task requirements and retained evidence; catalog task files once."""
import json,os,re,sqlite3,stat,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;T=R.parent;W=T.parent.parent
CUTOFF=1794030600000000000  # overridden by ISO conversion below
from datetime import datetime,timezone
CUTOFF=int(datetime.fromisoformat('2026-11-07T05:50:00+00:00').timestamp()*1e9)
def main():
 c=sqlite3.connect(R/'inventory.sqlite3');c.execute('CREATE TABLE task_files(path TEXT PRIMARY KEY,size INTEGER,basename TEXT)')
 docs={};posts=[];errors=[]
 def visit(root):
  try:
   with os.scandir(root) as it:
    for e in it:
     if e.is_symlink(): continue
     if e.is_dir(follow_symlinks=False):
      if e.name not in ['cleanup_20261103','cleanup_20261107','codex-implement-86673e2f-3b87-4663-a948-1a9ec9eb03a4']:visit(e.path)
     elif e.is_file(follow_symlinks=False):
      s=e.stat(follow_symlinks=False);c.execute('INSERT INTO task_files VALUES(?,?,?)',(e.path,s.st_size,e.name))
      if e.name=='post_run_command.json':posts.append(e.path)
  except OSError as e:errors.append({'path':str(root),'error':str(e)})
 visit(T);c.execute('CREATE INDEX task_size ON task_files(size)');c.commit()
 for p in [T/x for x in ['requirements.md','plan.md','progress.md','summary.md','issues.md']]+list((T/'tickets').glob('*.md')):
  text=p.read_text();matches=[]
  for i,l in enumerate(text.splitlines(),1):
   if (p.name=='plan.md' and ('→' in l or 'dependency' in l.lower()) and i>110) or ('2026-11-07' in l or 'd505eece' in l or 'a06a72a4' in l or 'post_run_command' in l):matches.append({'line':i,'text':l})
  docs[str(p)]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode()),'relevant_lines':matches}
 post_data={p:json.loads(Path(p).read_text()) for p in posts}
 old=json.loads((T/'cleanup_20261103/protected_before.json').read_text())
 protected={p:'November 3 retained input/active root' for p in old['protected_paths'] if '/h200/' not in p}
 for p in [T/'h200_mainline/p2_64_formal_retry05',T/'h200_mainline/p2_64_formal_retry06',T/'h200_mainline/build_package.py',T/'h200_mainline/terminal_receipt.py',Path('/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry05.sh'),Path('/data/ycfeng/tmp/glm53_dsv4/h200/monitor'),Path('/data/ycfeng/tmp/glm53_dsv4/h200/ds4_resume'),Path('/data/ycfeng/tmp/ds4_runs/p64-formal2-1102n/scratch/poll_once.py'),Path('/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline'),W/'.cortexkit',W/'.pi',W/'launcher.pid.txt',W/'remaining_measurements']:
  protected[str(p)]='Explicit assignment protection (retry06 added conservatively due concurrent mainline)'
 for p,d in post_data.items():
  def strings(x):
   if isinstance(x,str):yield x
   elif isinstance(x,list):
    for y in x:yield from strings(y)
   elif isinstance(x,dict):
    for y in x.values():yield from strings(y)
  for s in strings(d):
   if s.startswith('/data/ycfeng/'):protected[s]='Referenced by '+p
 # Current promotion/source paths in package build/acceptance; record entire documents.
 sources={}
 for name in ['h200_mainline/p2_64_formal_retry05/package_01_build.json','h200_mainline/p2_64_formal_retry05/lead_package_acceptance.json']:
  p=T/name;d=json.loads(p.read_text());sources[str(p)]=d
  for k in ['source','harness','package','case']:
   if isinstance(d.get(k),str) and d[k].startswith('/data/ycfeng/'):protected[d[k]]='Frozen build input '+str(p)
 # Read registered worktree metadata directly, without any Git command.
 common=Path('/data/ycfeng/Frontier/.git');worktrees={str(Path('/data/ycfeng/Frontier'))}
 for p in (common/'worktrees').glob('*/gitdir'):
  worktrees.add(str(Path(p.read_text().strip()).parent))
 for p in worktrees:protected[p]='Registered Git worktree; excluded from deletion roots'
 labels=[]
 for path,m,ct in c.execute("SELECT path,max_mtime_ns,max_ctime_ns FROM directories WHERE path IN(SELECT path FROM nodes WHERE parent='/data/ycfeng/tmp/ds4_runs')"):
  cwd=Path(path)/'cwd';ownership=False;ev=[]
  if cwd.is_file():
   content=cwd.read_text().strip()
   if content==str(W):ownership=True;ev.append({'path':str(cwd),'cwd':content})
  for p in [Path(path)/'prompt.md',Path('/data/ycfeng/tmp/ds4_runs/prompts')/(Path(path).name+'.md')]:
   if p.is_file():
    txt=p.read_text()
    if str(W) in txt or T.name in txt:ownership=True;ev.append({'path':str(p),'sha256':hashlib.sha256(txt.encode()).hexdigest(),'reference':str(W) if str(W) in txt else T.name})
  recent=max(m,ct)>=CUTOFF
  if recent:protected[path]='Pi label created/modified at or after 2026-11-07T05:50Z (mtime OR ctime of any descendant)'
  if Path(path).name in ['p64-formal5-submit-1107a','p64-formal5-submit-1107d','provider-probe-1107c']:protected[path]='I-010 evidence; label and node cache conservatively retained'
  labels.append({'path':path,'owned':ownership,'evidence':ev,'recent':recent,'max_mtime_ns':m,'max_ctime_ns':ct})
 # Credentials: record existence/hash without contents later; never delete.
 for p in list(Path('/data/ycfeng/envs').glob('*'))+[Path('/data/ycfeng/tmp/stepmind-env')]+list(Path('/data/ycfeng/tmp').glob('brainctl-auth-*')):
  protected[str(p)]='Required environment or credential'
 out={'documents':docs,'post_run_commands':post_data,'package_sources':sources,'protected_paths':protected,'old_hash_files':list(old['files']),'worktrees':sorted(worktrees),'labels':labels,'catalog_errors':errors,'cutoff_ns':CUTOFF,'task_file_count':c.execute('SELECT COUNT(*) FROM task_files').fetchone()[0]}
 (R/'context.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'task_files':out['task_file_count'],'task_catalog_errors':len(errors),'worktrees':len(worktrees),'protected_roots':len(protected),'labels':len(labels),'owned_labels':sum(x['owned'] for x in labels),'recent_labels':sum(x['recent'] for x in labels)},indent=2))
if __name__=='__main__':main()
