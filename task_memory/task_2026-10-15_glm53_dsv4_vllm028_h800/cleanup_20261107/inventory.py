#!/usr/bin/env python3
"""One bounded, non-following inventory of both authorized deletion roots."""
import datetime, json, os, sqlite3, stat, subprocess, time
from pathlib import Path
RECORD=Path(__file__).resolve().parent
ROOTS=['/data/ycfeng/tmp/glm53_dsv4','/data/ycfeng/tmp/ds4_runs']
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    db=sqlite3.connect(RECORD/'inventory.sqlite3')
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('CREATE TABLE nodes(path TEXT PRIMARY KEY,parent TEXT,kind TEXT,size INTEGER,allocated INTEGER,dev INTEGER,ino INTEGER,mtime_ns INTEGER,ctime_ns INTEGER,nlink INTEGER,target TEXT)')
    db.execute('CREATE TABLE directories(path TEXT PRIMARY KEY,files INTEGER,allocated INTEGER,logical INTEGER,dirs INTEGER,symlinks INTEGER,special INTEGER,max_mtime_ns INTEGER,max_ctime_ns INTEGER,extensions TEXT,errors TEXT)')
    started=now(); errors=[]; count=0
    def visit(path):
        nonlocal count
        s=os.lstat(path); kind='dir' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'link' if stat.S_ISLNK(s.st_mode) else 'special'
        db.execute('INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)',(path,os.path.dirname(path),kind,s.st_size,s.st_blocks*512,s.st_dev,s.st_ino,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,os.readlink(path) if kind=='link' else None))
        count+=1
        a={'files':int(kind=='file'),'allocated':s.st_blocks*512 if kind=='file' else 0,'logical':s.st_size if kind=='file' else 0,'dirs':int(kind=='dir'),'symlinks':int(kind=='link'),'special':int(kind=='special'),'max_mtime_ns':s.st_mtime_ns,'max_ctime_ns':s.st_ctime_ns,'extensions':{},'errors':[]}
        if kind=='file': a['extensions'][Path(path).suffix or '<none>']=1
        if kind=='dir':
            with os.scandir(path) as it:
                for item in it:
                    try: child=visit(item.path)
                    except OSError as e:
                        msg={'path':item.path,'error':str(e)};errors.append(msg);a['errors'].append(msg);continue
                    for k in ['files','allocated','logical','dirs','symlinks','special']: a[k]+=child[k]
                    for k in ['max_mtime_ns','max_ctime_ns']: a[k]=max(a[k],child[k])
                    for k,v in child['extensions'].items(): a['extensions'][k]=a['extensions'].get(k,0)+v
                    a['errors']+=child['errors']
            db.execute('INSERT INTO directories VALUES(?,?,?,?,?,?,?,?,?,?,?)',(path,*[a[k] for k in ['files','allocated','logical','dirs','symlinks','special','max_mtime_ns','max_ctime_ns']],json.dumps(a['extensions']),json.dumps(a['errors'])))
        if count%50000==0: db.commit(); print('inventoried',count,flush=True)
        return a
    totals={r:visit(r) for r in ROOTS};db.commit()
    db.execute('CREATE INDEX node_parent ON nodes(parent)');db.commit()
    top=[dict(zip(['path','files','allocated','logical','dirs','symlinks','special','max_mtime_ns','max_ctime_ns','extensions','errors'],r)) for r in db.execute('SELECT * FROM directories WHERE path IN (SELECT path FROM nodes WHERE parent IN (?,?))',ROOTS)]
    out={'started_utc':started,'finished_utc':now(),'interpreter':subprocess.check_output(['python3','--version'],text=True).strip(),'nodes':count,'roots':totals,'top_level':sorted(top,key=lambda r:-r['allocated']),'errors':errors,'method':'lstat/scandir; symlink targets never traversed; regular allocation is st_blocks*512 (hardlink counts per path)'}
    (RECORD/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'nodes':count,'errors':len(errors),'roots':{k:{x:v[x] for x in ['files','allocated','dirs']} for k,v in totals.items()}},indent=2))
if __name__=='__main__':main()
