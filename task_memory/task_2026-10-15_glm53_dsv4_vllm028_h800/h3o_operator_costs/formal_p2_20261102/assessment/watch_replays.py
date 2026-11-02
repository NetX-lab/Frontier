"""Read commands once per ten minutes; wait without touching replay state."""
import datetime
import json
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
records=HERE/'scratch/replay_polls.jsonl'
while True:
    now=datetime.datetime.now(datetime.timezone.utc)
    cases={}
    for case in ['p2_32','p2_64']:
        d=json.loads((HERE.parent/f'{case}_command.json').read_text())
        cases[case]={'status':d['status'],'simulator_status':d.get('simulator_status'),'stage': 'readings' if d.get('readings') else 'gate' if d.get('gate') else 'simulator','run_root':d['run_root']}
    row={'observed_utc':now.isoformat(),'cases':cases}
    with records.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if all(d['status'] in ['FINISHED','FAILED'] for d in cases.values()):break
    if now.hour>=9:print('09:00Z cutoff reached',flush=True);break
    for _ in range(12):time.sleep(50)
