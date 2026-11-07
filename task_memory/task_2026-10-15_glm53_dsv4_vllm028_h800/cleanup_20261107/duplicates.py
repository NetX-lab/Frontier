#!/usr/bin/env python3
"""Prove duplicate simulator traces by SHA256; never classify native measurements."""
import hashlib,json,sqlite3,time,os
from pathlib import Path
R=Path(__file__).resolve().parent

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=sqlite3.connect(R/'inventory.sqlite3');ctx=json.loads((R/'context.json').read_text());protected=ctx['protected_paths']
 rows=list(c.execute("SELECT path,size,mtime_ns,ctime_ns,dev,ino FROM nodes WHERE kind='file' AND path LIKE '%/op_traces.jsonl' AND size>=67108864"));sizes={}
 for x in rows:sizes.setdefault(x[1],[]).append(x)
 hashes=[];duplicates=[]
 for size,group in sorted(sizes.items(),reverse=True):
  if len(group)<2:continue
  seen={}
  for p,s,m,ct,dev,ino in sorted(group,key=lambda x:(not any(x[0]==q or x[0].startswith(q+'/') for q in protected),x[0])):
   st=os.lstat(p)
   if (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_dev,st.st_ino)!=(s,m,ct,dev,ino):raise RuntimeError('changed trace '+p)
   digest=sha256(p);hashes.append({'path':p,'sha256':digest,'size':s})
   print('hashed',p,flush=True)
   if digest in seen and not any(p==q or p.startswith(q+'/') for q in protected):duplicates.append({'path':p,'retained_copy':seen[digest],'sha256':digest,'size':s})
   else:seen.setdefault(digest,p)
 (R/'duplicate_evidence.json').write_text(json.dumps({'hashes':hashes,'duplicates':duplicates},indent=2)+'\n')
 print(json.dumps({'hashed':len(hashes),'duplicates':len(duplicates),'duplicate_bytes':sum(x['size'] for x in duplicates)},indent=2))
if __name__=='__main__':main()
