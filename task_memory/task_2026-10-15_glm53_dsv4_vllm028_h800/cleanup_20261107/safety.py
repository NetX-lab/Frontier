#!/usr/bin/env python3
"""Adapted Nov 3 exact-path validation and /proc scan; no shell deletion."""
import datetime,hashlib,json,os,re,sqlite3,stat,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;T=R.parent;ROOTS=['/data/ycfeng/tmp/glm53_dsv4','/data/ycfeng/tmp/ds4_runs']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def under(p,q):return p==q or p.startswith(q+'/')
def nodes(path):
 """Non-following generator that fails closed on inaccessible entries."""
 s=os.lstat(path);yield str(path),s
 if stat.S_ISDIR(s.st_mode):
  with os.scandir(path) as it:
   for e in it:yield from nodes(e.path)
def measure(path):
 out={'regular_file_count':0,'allocated_bytes':0,'logical_bytes':0,'directory_count':0,'symlink_count':0,'special':0};h=hashlib.sha256();max_time=0
 for p,s in nodes(path):
  max_time=max(max_time,s.st_mtime_ns,s.st_ctime_ns)
  kind='file' if stat.S_ISREG(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'link' if stat.S_ISLNK(s.st_mode) else 'special'
  if kind=='file':out['regular_file_count']+=1;out['allocated_bytes']+=s.st_blocks*512;out['logical_bytes']+=s.st_size
  elif kind=='dir':out['directory_count']+=1
  elif kind=='link':out['symlink_count']+=1
  else:out['special']+=1
  h.update(json.dumps([p,kind,s.st_dev,s.st_ino,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,os.readlink(p) if kind=='link' else None],separators=(',',':')).encode());h.update(b'\n')
 out['metadata_sha256']=h.hexdigest();out['max_time_ns']=max_time;return out
def mountpoints():
 return [re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),l.split()[4]) for l in Path('/proc/self/mountinfo').read_text().splitlines()]
def proc_scan(targets):
 own=os.getpid();hits=[];denied={};scanned=0
 def hit(pid,kind,path):
  path=path.removesuffix(' (deleted)')
  for t in targets:
   if under(path,t):hits.append({'pid':pid,'kind':kind,'path':path,'target':t})
 def read(p):
  try:return Path(p).read_bytes()
  except (FileNotFoundError,ProcessLookupError):return b''
  except PermissionError:denied[str(p)]=True;return b''
 for pid in os.listdir('/proc'):
  if not pid.isdigit() or int(pid)==own:continue
  scanned+=1;base='/proc/'+pid
  for n in ['cwd','root','exe']:
   try:hit(int(pid),n,os.readlink(base+'/'+n))
   except (FileNotFoundError,ProcessLookupError):pass
   except PermissionError:denied[base+'/'+n]=True
  try:fds=os.listdir(base+'/fd')
  except (FileNotFoundError,ProcessLookupError):fds=[]
  except PermissionError:denied[base+'/fd']=True;fds=[]
  for fd in fds:
   try:hit(int(pid),'fd',os.readlink(base+'/fd/'+fd))
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  for l in read(base+'/maps').decode(errors='replace').splitlines():
   parts=l.split(None,5)
   if len(parts)==6:hit(int(pid),'maps',parts[5])
  cmd=read(base+'/cmdline').replace(b'\0',b' ').decode(errors='replace')
  for t in targets:
   if t in cmd:hits.append({'pid':int(pid),'kind':'cmdline_reference','path':t,'target':t})
 return {'observed_utc':now(),'scanned_processes':scanned,'hits':hits,'unreadable_count':len(denied),'unreadable_paths':sorted(denied),'note':'No cmdline contents retained (credentials excluded). Permission limits do not establish that a path is unused.'}
def validate(entry,protected):
 p=entry['path'];errors=[]
 if not any(p.startswith(x+'/') for x in ROOTS) or os.path.normpath(p)!=p:errors.append('outside exact deletion roots or noncanonical path')
 if os.path.realpath(p)!=p:errors.append('symlink ancestor or entry')
 for q,why in protected.items():
  if under(p,q) or under(q,p):errors.append('protected overlap '+q)
 if any(under(m,p) for m in mountpoints()):errors.append('nested mount boundary')
 current='/';components=[]
 for piece in p.strip('/').split('/'):
  current=os.path.join(current,piece);components.append(current)
  try:
   st=os.lstat(current)
   if stat.S_ISLNK(st.st_mode):errors.append('symlink component '+current)
  except OSError as e:errors.append(str(e));break
 if errors:return {'path':p,'errors':errors}
 s=os.lstat(p)
 if s.st_dev!=os.lstat(os.path.dirname(p)).st_dev:errors.append('cross-device entry')
 if (s.st_dev,s.st_ino)!=(entry['identity']['dev'],entry['identity']['ino']):errors.append('entry identity changed')
 try:
  obs=measure(p)
  for k in ['regular_file_count','allocated_bytes','logical_bytes','directory_count','symlink_count']:
   if obs[k]!=entry[k]:errors.append(f'{k} changed: {obs[k]} != {entry[k]}')
  if obs['special']:errors.append('nonregular files')
  if entry['category']=='sklearn_cache':
   for q,st in nodes(p):
    if stat.S_ISREG(st.st_mode) and Path(q).suffix not in ['.file','.pkl']:errors.append('unexpected cache extension '+q)
  if entry['category']=='node_compile_cache':
   label=ROOTS[1]+'/'+p[len(ROOTS[1])+1:].split('/')[0]
   if measure(label)['max_time_ns']>=json.loads((R/'context.json').read_text())['cutoff_ns']:errors.append('Pi label now recent; full protection')
  for q,st in nodes(p):
   if Path(q).name=='.git':errors.append('Git marker inside candidate '+q)
  if entry['category']=='duplicate_simulator_trace':
   keeper=entry['evidence']['retained_copy'];wanted=entry['evidence']['sha256']
   if sha(p)!=wanted or sha(keeper)!=wanted:errors.append('duplicate hashes no longer equal')
  return {'path':p,'errors':errors,'observed':obs}
 except OSError as e:return {'path':p,'errors':errors+[str(e)]}
def df():
 p=subprocess.run(['df','-B1','/data/ycfeng'],capture_output=True,text=True)
 return {'observed_utc':now(),'argv':['df','-B1','/data/ycfeng'],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
