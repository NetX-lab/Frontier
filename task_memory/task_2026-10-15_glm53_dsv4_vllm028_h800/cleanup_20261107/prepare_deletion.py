#!/usr/bin/env python3
"""Validate candidates, drop live/uncertain paths, then freeze exact manifest."""
import json
from safety import R,validate,proc_scan,now

def main():
 m=json.loads((R/'deletion_manifest.json').read_text());plan=json.loads((R/'protection_plan.json').read_text());pro=plan['protected_paths']
 rows=[]
 for e in m['entries']:
  row=validate(e,pro);rows.append(row);print('validation', 'PASS' if not row['errors'] else 'RETAIN',e['path'],flush=True)
 scan=proc_scan([e['path'] for e in m['entries']]);(R/'proc_scan_pre.json').write_text(json.dumps(scan,indent=2)+'\n')
 live={h['target'] for h in scan['hits']};keep=[]
 for e,row in zip(m['entries'],rows):
  if row['errors'] or e['path'] in live:
   m['uncertain_retained'].append({'path':e['path'],'reason':'pre-deletion validation or live process reference','evidence':{'errors':row['errors'],'live_references':[h for h in scan['hits'] if h['target']==e['path']]}})
  else:e['validated_metadata_sha256']=row['observed']['metadata_sha256'];keep.append(e)
 (R/'validation_pre.json').write_text(json.dumps({'observed_utc':now(),'rows':rows,'candidate_count':len(rows),'passed_count':len(keep)},indent=2)+'\n')
 m['entries']=keep;m['frozen_utc']=now();(R/'deletion_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'frozen_entries':len(keep),'retained_after_validation':len(rows)-len(keep),'proc_hits':len(scan['hits']),'proc_unreadable':scan['unreadable_count']},indent=2))
if __name__=='__main__':main()
