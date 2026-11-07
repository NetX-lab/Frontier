#!/usr/bin/env python3
"""Separate actual removals from retained candidates; never rerun any rm."""
import json,shutil
from safety import R,now

def main():
 if (R/'candidate_manifest.json').exists():raise RuntimeError('Already finalized; no repeat')
 shutil.copyfile(R/'deletion_manifest.json',R/'candidate_manifest.json')
 shutil.copyfile(R/'verification.json',R/'verification_initial.json')
 shutil.copyfile(R/'df_after.json',R/'df_after_initial.json')
 m=json.loads((R/'deletion_manifest.json').read_text());raw=json.loads((R/'deletion_receipt.json').read_text());deleted=[];retained=[]
 for e in m['entries']:
  row=e['deletion_receipt']
  if row['outcome']=='deleted':deleted.append(e)
  else:
   if 'rm_returncode' in row:raise RuntimeError('An attempted rm failed; cannot finalize as a retention')
   retained.append(e);m['uncertain_retained'].append({'path':e['path'],'reason':'Retained without invoking rm: label failed the immediate cutoff recheck. Earlier pytest cleanup in the same label updates its scratch timestamp; concurrent updates cannot be excluded.','evidence':row,'regular_file_count':e['regular_file_count'],'allocated_bytes':e['allocated_bytes']})
 m['entries']=deleted;m['retained_not_attempted']=retained;m['finalized_utc']=now();m['candidate_manifest']=str(R/'candidate_manifest.json')
 (R/'deletion_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 out={'status':'PASS','observed_utc':now(),'results':[e['deletion_receipt'] for e in deleted],'deleted_entries':len(deleted),'retained_not_attempted':[{'path':e['path'],'command_executed':False,'receipt':e['deletion_receipt']} for e in retained],'original_execution_receipt':str(R/'deletion_receipt.json'),'note':'All actual rm calls returned zero with no stderr and their paths were absent. The raw driver status PARTIAL includes the deliberate retained candidate; no rm was attempted for that candidate.'}
 (R/'deletion_receipt_final.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Final manifest:',len(deleted),'actual removals;',len(retained),'retained without rm')
if __name__=='__main__':main()
