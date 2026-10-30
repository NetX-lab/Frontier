"""List worker records between a prefill execute_model and the next forward on the same lane."""
import json, sys, statistics
from pathlib import Path
run = Path(sys.argv[1])
def rows(p): return [json.loads(l) for l in open(p) if l.strip()]
shown = 0
gaps = []
for p in sorted(run.glob('nodes/*/worker_dp_wave/*.jsonl')):
    r = rows(p); s = next(x for x in r if x['event'] == 'worker_wave_start')
    if s['role'] != 'prefill': continue
    for i, x in enumerate(r):
        if x['event'] != 'worker_execute_model' or x.get('tokens') != 6144: continue
        if x.get('calls', {}).get('dp_padding_sync', {}).get('stage') != 'serving': continue
        j = i + 1
        while j < len(r) and not r[j].get('calls', {}).get('dp_cpu_all_reduce'): j += 1
        if j == len(r): continue
        between = r[i+1:j]
        gaps.append(1000 * (r[j]['t'] - x['end_t']))
        if shown < 3:
            shown += 1
            print('prefill exec', round(1000*(x['end_t']-x['t']), 1), 'ms; next forward after', round(gaps[-1], 1), 'ms')
            for y in between:
                print('   ', y['event'], 'start +%.1f' % (1000*(y['t']-x['end_t'])), 'dur %.1f' % (1000*(y['end_t']-y['t'])), 'tokens', y.get('tokens'), list(y.get('calls', {})))
print('exec end -> next forward start: median', round(statistics.median(gaps), 1), 'p10', round(sorted(gaps)[len(gaps)//10], 1), 'p90', round(sorted(gaps)[9*len(gaps)//10], 1), 'n', len(gaps))
