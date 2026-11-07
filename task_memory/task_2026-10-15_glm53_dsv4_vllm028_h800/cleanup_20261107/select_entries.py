#!/usr/bin/env python3
"""Build explicit candidates from inventory and already retained task evidence."""
import json,sqlite3,hashlib,re,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;T=R.parent;ROOT='/data/ycfeng/tmp/glm53_dsv4';PI='/data/ycfeng/tmp/ds4_runs'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 c=sqlite3.connect(R/'inventory.sqlite3');c.row_factory=sqlite3.Row;ctx=json.loads((R/'context.json').read_text());pro=ctx['protected_paths'];labels={x['path']:x for x in ctx['labels']};entries=[];uncertain=[]
 def protected(p):return [q for q in pro if p==q or p.startswith(q+'/') or q.startswith(p+'/')]
 def add(p,cat,reason,evidence,retained=()):
  if protected(p):uncertain.append({'path':p,'reason':'protected input or protected Pi label','evidence':protected(p)});return
  if any(p==x['path'] or p.startswith(x['path']+'/') for x in entries):return
  markers=c.execute("SELECT path FROM nodes WHERE (path=? OR path LIKE ?) AND (path LIKE '%/.git' OR path LIKE '%/.git/%') LIMIT 1",(p,p+'/%')).fetchone()
  if markers:uncertain.append({'path':p,'reason':'contains a Git repository marker; preserve all tracked fixtures conservatively','evidence':markers[0]});return
  st=c.execute('SELECT * FROM nodes WHERE path=?',(p,)).fetchone()
  if st['kind']=='dir':
   d=c.execute('SELECT * FROM directories WHERE path=?',(p,)).fetchone()
   if json.loads(d['errors']) or d['special']:uncertain.append({'path':p,'reason':'inventory incomplete or nonregular contents','evidence':dict(d)});return
   stats={'regular_file_count':d['files'],'allocated_bytes':d['allocated'],'logical_bytes':d['logical'],'directory_count':d['dirs'],'symlink_count':d['symlinks']}
  else:stats={'regular_file_count':1,'allocated_bytes':st['allocated'],'logical_bytes':st['size'],'directory_count':0,'symlink_count':0}
  entries.append({'path':p,'action':'remove_directory' if st['kind']=='dir' else 'remove_file','category':cat,'reason':reason,'evidence':evidence,'retained':list(retained),'identity':{k:st[k] for k in ['dev','ino','mtime_ns','ctime_ns']},**stats})
 # Pytest fixture roots only: directory naming plus actual test-case children.
 for d in c.execute('SELECT * FROM directories ORDER BY length(path),path'):
  p=d['path'];name=Path(p).name
  if not d['files'] or not (name=='pytest' or name.startswith(('pytest_','pytest-','dummy_basetemp','placement_basetemp','held_basetemp')) or 'basetemp' in name):continue
  if any(p.startswith(x['path']+'/') for x in entries):continue
  if p.startswith(PI+'/'):
   label=PI+'/'+p[len(PI)+1:].split('/')[0]
   if not labels.get(label,{}).get('owned'):continue
  children=[Path(x[0]).name for x in c.execute('SELECT path FROM nodes WHERE parent=?',(p,))]
  fixtures=[n for n in children if n.startswith('test_') or re.fullmatch(r'pytest-\d+',n)]
  if not fixtures:continue
  add(p,'pytest_temporary','Disposable pytest basetemp fixtures; retained source, measurement inputs, run logs and assertions remain outside this exact tree.',{'directory_name':name,'test_fixture_children':fixtures,'precedent':str(T/'test_report_2026-11-03_task_cleanup.md')})
 # Rebuildable estimator/lock-only directories, never CSV profiling inputs.
 for d in c.execute('SELECT * FROM directories WHERE files>0 AND symlinks=0 AND special=0 ORDER BY length(path),path'):
  p=d['path'];name=Path(p).name
  if not (name.startswith('fold_') or 'cache' in name):continue
  if p.startswith(PI+'/'):continue
  if set(json.loads(d['extensions']))<={'.pkl','.file'}:
   add(p,'sklearn_cache','Regenerable sklearn estimator/prediction caches and lock files only; no profiling CSV or config inside the selected directory.',{'extensions':json.loads(d['extensions']),'precedent':str(T/'test_report_2026-11-03_task_cleanup.md'),'rebuild_source':'frontier/execution_time_predictor/sklearn_execution_time_predictor.py'})
 # Completed historical fidelity case outputs; archive summaries before any deletion.
 for (p,) in c.execute("SELECT path FROM directories WHERE path LIKE '%/cases' ORDER BY path"):
  if 'fidelity' not in p:continue
  parent=Path(p).parent;m=parent/'manifest.json';res=parent/'results.jsonl'
  if protected(p):continue
  if not(m.is_file() and res.is_file()):continue
  try:
   manifest=json.loads(m.read_text());records=[json.loads(l) for l in res.read_text().splitlines() if l.strip()]
   if len(records)<50 or any(x.get('returncode')!=0 or x.get('error') is not None for x in records):
    uncertain.append({'path':p,'reason':'fidelity outcome not fully successful or fewer than 50 cases','evidence':{'results':str(res),'records':len(records)}});continue
   if any(not x.get('source_revision') or not x.get('harness_revision') or not x.get('script') or not x.get('artifacts') for x in records):raise ValueError('incomplete per-case provenance/artifact records')
   if any(not x.get('artifact_dir','').startswith('cases/') or any(not re.fullmatch('[0-9a-f]{64}',a.get('sha256','')) for a in x['artifacts']) for x in records):raise ValueError('invalid artifact paths/hashes')
   dest=R/'evidence/fidelity'/hashlib.sha256(str(parent).encode()).hexdigest()[:16];dest.mkdir(parents=True,exist_ok=True)
   copied=[]
   for source in [m,res]:
    target=dest/source.name;shutil.copyfile(source,target)
    if sha(source)!=sha(target):raise RuntimeError('evidence copy differs')
    copied.append({'original':str(source),'retained_copy':str(target),'sha256':sha(target)})
   add(p,'historical_fidelity_outputs','Completed superseded CPU fidelity case outputs; all successful per-case outcomes, source/harness revisions, arguments and artifact hashes retained in both original and task-record copies. November 3 baseline remains protected.',{'case_count':len(records),'returncodes':{'0':len(records)},'summary_copies':copied,'precedent':str(T/'test_report_2026-11-03_task_cleanup.md')},[str(m),str(res)]+[x['retained_copy'] for x in copied])
  except (ValueError,OSError,KeyError) as e:uncertain.append({'path':p,'reason':'fidelity evidence failed validation','evidence':str(e)})
 # Old owned Pi node caches. All other contents of each label remain intact.
 for d in c.execute("SELECT * FROM directories WHERE path LIKE '%/node-compile-cache' ORDER BY path"):
  p=d['path'];label=PI+'/'+p[len(PI)+1:].split('/')[0];info=labels.get(label,{})
  if not info.get('owned'):uncertain.append({'path':p,'reason':'Pi label ownership not proven','evidence':info.get('evidence',[])});continue
  if not (Path(label)/'exit').is_file():uncertain.append({'path':p,'reason':'No recorded terminal Pi exit; retain possible live cache','evidence':info});continue
  info=dict(info);info['terminal_exit']={'path':str(Path(label)/'exit'),'value':(Path(label)/'exit').read_text().strip()}
  if info.get('recent'):uncertain.append({'path':p,'reason':'entire label protected by 05:50Z cutoff','evidence':info});continue
  add(p,'node_compile_cache','Regenerable Node compilation cache inside an old task-owned Pi scratch; cwd/prompt prove ownership; pre-cutoff label.',info, [x['path'] for x in info['evidence']])
 # Hash-identical simulator traces. Unique native measurements are never selected.
 dup=json.loads((R/'duplicate_evidence.json').read_text())
 for x in dup['duplicates']:
  if any(x['path']==e['path'] or x['path'].startswith(e['path']+'/') for e in entries):continue
  if any(x['retained_copy']==e['path'] or x['retained_copy'].startswith(e['path']+'/') for e in entries):
   uncertain.append({'path':x['path'],'reason':'duplicate keeper is covered by another candidate; keep this trace','evidence':x});continue
  add(x['path'],'duplicate_simulator_trace','Redundant generated simulator operator trace; full SHA256 equals an existing retained trace. Only this file is deleted; configs, ledgers, validation, summaries and logs remain.',x,[x['retained_copy']])
 # Retain uncertain large native and accepted diagnostic outputs; no speculative raw cleanup.
 inv=json.loads((R/'inventory.json').read_text())
 for x in inv['errors']:uncertain.append({'path':x['path'],'reason':'permission denied; invalidity and contents unavailable','evidence':x['error']})
 for x in inv['top_level']:
  p=x['path']
  if not p.startswith(ROOT+'/') or x['allocated']<1<<30:continue
  uncertain.append({'path':p,'reason':'Retain all unselected contents: unique raw measurements/profiling tables or historical accepted evidence without a verified duplicate; only individually manifested descendants may go.','evidence':{'allocated_bytes':x['allocated'],'regular_file_count':x['files'],'selected_descendants':[e['path'] for e in entries if e['path'].startswith(p+'/')]}})
 out={'authorization':'requirements.md 2026-11-07 prompt cleanup and 2026-11-03 task artifact cleanup; explicit rm authorized by caller','roots':[ROOT,PI],'entries':entries,'uncertain_retained':uncertain,'method':'exact paths; non-following inventory; immutable retention checks; process scan immediately before removal'}
 (R/'deletion_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
 cats={}
 for e in entries:
  a=cats.setdefault(e['category'],{'entries':0,'regular_files':0,'allocated_bytes':0});a['entries']+=1;a['regular_files']+=e['regular_file_count'];a['allocated_bytes']+=e['allocated_bytes']
 print(json.dumps({'categories':cats,'uncertain_entries':len(uncertain)},indent=2))
if __name__=='__main__':main()
