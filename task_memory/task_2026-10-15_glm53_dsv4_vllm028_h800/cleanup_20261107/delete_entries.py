#!/usr/bin/env python3
"""Remove each frozen exact manifest path once; persist every deletion receipt."""
import json,os,subprocess,time
from safety import R,validate,proc_scan,now,sha

def main():
 receipt_path=R/'deletion_receipt.json'
 if receipt_path.exists():raise RuntimeError('Receipt exists: no deletion retry or replay allowed')
 before=json.loads((R/'protected_hash_before.json').read_text())
 if before['errors']:raise RuntimeError('protected snapshot incomplete; do not delete')
 m=json.loads((R/'deletion_manifest.json').read_text());plan=json.loads((R/'protection_plan.json').read_text());pro=plan['protected_paths']
 receipt={'started_utc':now(),'manifest_sha256_before':sha(R/'deletion_manifest.json'),'results':[],'status':'IN_PROGRESS'}
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 scan=proc_scan([e['path'] for e in m['entries']]);(R/'proc_scan_immediate.json').write_text(json.dumps(scan,indent=2)+'\n');live={h['target'] for h in scan['hits']}
 started=time.monotonic()
 for e in m['entries']:
  p=e['path'];row={'path':p,'category':e['category'],'started_utc':now(),'command_argv':['rm','-rf','--',p]}
  if p in live:
   row.update(outcome='retained',error='live process reference immediately before deletion');receipt['results'].append(row);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');continue
  check=validate(e,pro)
  if check['errors'] or check.get('observed',{}).get('metadata_sha256')!=e['validated_metadata_sha256']:
   row.update(outcome='retained',error='pre-removal revalidation failed',validation=check)
  else:
   row['observed_before']=check['observed']
   # No shell, glob expansion, symlink traversal, or rm retry.
   proc=subprocess.run(row['command_argv'],capture_output=True,text=True)
   row.update(rm_returncode=proc.returncode,rm_stdout=proc.stdout,rm_stderr=proc.stderr,absent_after=not os.path.lexists(p))
   row['outcome']='deleted' if proc.returncode==0 and row['absent_after'] and not proc.stderr else 'error'
   row['error']=None if row['outcome']=='deleted' else 'rm error or remaining path'
  row['finished_utc']=now();receipt['results'].append(row);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(row['outcome'],p,flush=True)
 receipt['finished_utc']=now();receipt['runtime_seconds']=time.monotonic()-started
 receipt['status']='PASS' if all(x['outcome']=='deleted' for x in receipt['results']) else 'PARTIAL'
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 # Receipts embedded in manifest make every entry self-contained.
 for e,row in zip(m['entries'],receipt['results']):e['deletion_receipt']=row
 (R/'deletion_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print('receipt',receipt['status'],len(receipt['results']),'entries',flush=True)
if __name__=='__main__':main()
