#!/usr/bin/env python3
"""Readback receipts, protected hashes/counts and exact original inventory paths."""
import json,os,sqlite3,stat
from safety import R,sha,now,under,df

def main():
 m=json.loads((R/'deletion_manifest.json').read_text());receipt=json.loads((R/('deletion_receipt_final.json' if (R/'deletion_receipt_final.json').exists() else 'deletion_receipt.json')).read_text());before=json.loads((R/'protected_hash_before.json').read_text());after=json.loads((R/'protected_hash_after.json').read_text());plan=json.loads((R/'protection_plan.json').read_text())
 deleted=[e['path'] for e in m['entries'] if e['deletion_receipt']['outcome']=='deleted'];errors=[]
 for e in m['entries']:
  row=e['deletion_receipt']
  if row['outcome']=='deleted' and (os.path.lexists(e['path']) or row['rm_returncode']!=0 or row['rm_stderr'] or not row['absent_after']):errors.append({'path':e['path'],'error':'deletion receipt or absence failed'})
 unchanged=[];changed=[]
 for p,v in before['files'].items():
  if after['files'].get(p)==v:unchanged.append(p)
  else:changed.append({'path':p,'before':v,'after':after['files'].get(p)})
 tree_changes=[{'path':p,'before':v,'after':after['trees'].get(p)} for p,v in before['trees'].items() if after['trees'].get(p)!=v]
 missing_worktrees=[p for p,v in before['worktrees'].items() if v['exists'] and not after['worktrees'].get(p,{}).get('exists')]
 # No fresh full-tree inventory: readback original named paths only.
 c=sqlite3.connect(R/'inventory.sqlite3');unexpected=[];identity_changes=[];surviving=0;covered=0
 for p,k,dev,ino,target in c.execute('SELECT path,kind,dev,ino,target FROM nodes'):
  if any(under(p,q) for q in deleted):covered+=1;continue
  try:
   s=os.lstat(p);surviving+=1
   if (s.st_dev,s.st_ino)!=(dev,ino):identity_changes.append({'path':p,'before':[dev,ino],'after':[s.st_dev,s.st_ino]})
   if k=='link' and os.readlink(p)!=target:identity_changes.append({'path':p,'error':'symlink target changed'})
  except FileNotFoundError:unexpected.append(p)
  except OSError as e:errors.append({'path':p,'error':str(e)})
 (R/'df_after.json').write_text(json.dumps(df(),indent=2)+'\n')
 package=R.parent/'h200_mainline/p2_64_formal_retry05/package_01/manifest.json';promotion=R.parent/'h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json'
 seals={'generation05_manifest_sha256':sha(package),'generation05_manifest_expected':'23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c','promotion_receipt_sha256':sha(promotion),'promotion_receipt_expected':'d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596'}
 out={'observed_utc':now(),'receipt_status':receipt['status'],'manifest_entries':len(m['entries']),'deleted_entries':len(deleted),'protected_files_unchanged':len(unchanged),'protected_files_changed':changed,'protected_tree_count_changes':tree_changes,'missing_registered_worktrees':missing_worktrees,'unlisted_missing_paths':unexpected,'unlisted_identity_changes':identity_changes,'surviving_inventory_nodes':surviving,'manifest_covered_inventory_nodes':covered,'snapshot_errors':before['errors']+after['errors'],'errors':errors,'seals':seals,'preexisting_absent_protections':plan['absent_before']}
 seal_ok=seals['generation05_manifest_sha256']==seals['generation05_manifest_expected'] and seals['promotion_receipt_sha256']==seals['promotion_receipt_expected']
 out['status']='PASS' if not any([changed,tree_changes,missing_worktrees,unexpected,identity_changes,out['snapshot_errors'],errors]) and receipt['status']=='PASS' and seal_ok else 'PARTIAL'
 (R/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['preexisting_absent_protections']},indent=2))
if __name__=='__main__':main()
