"""Scratch case for analyze_lifecycle.py: the CPU-check observer records plus one proxy/client request."""
import json, sys
from pathlib import Path
check, native06, out = (Path(a) for a in sys.argv[1:4])
case = out / 'case'
run = case / 'runs/groundtruth_clean/check'
(case / 'inputs').mkdir(parents=True)
(case / 'inputs/engine.json').write_text(json.dumps(dict(data_parallel_size=1)))
for role, source in (('decode', 'remote_prefill_observed'), ('prefill', 'remote_decode_observed')):
    target = run / f'nodes/host-{role}/scheduler_lifecycle'
    target.mkdir(parents=True)
    for path in (check / source / 'scheduler_lifecycle').glob('*.jsonl'):
        # The D API server names a proxied request cmpl-<client id>-<index>-<suffix>.
        (target / path.name).write_text(path.read_text().replace('"id-1"', '"cmpl-r1-0-00000000"'))
rows = [json.loads(l) for l in (check / 'remote_prefill_observed/scheduler_lifecycle').glob('*.jsonl').__next__().read_text().splitlines()]
add = next(r for r in rows if r['event'] == 'add')
first = next(r for r in rows if r['event'] == 'update' and 'first_token' in r)
response_wall = add['arrival_time'] - 0.002
first_byte_wall = first['t'] + first['dur_ms'] / 1e3 + 0.003
proxy = dict(request_id='r1', prefill_response_time=response_wall, prefill_response_monotonic=100.0,
             kv_transfer_params=dict(remote_request_id=add['remote_request_id']),
             decode_first_byte_time=first_byte_wall,
             decode_first_byte_monotonic=100.0 + first_byte_wall - response_wall)
(run / 'run').mkdir()
(run / 'run/proxy_requests.jsonl').write_text(json.dumps(proxy) + '\n')
(run / 'run/client_requests.jsonl').write_text(json.dumps(dict(request_id='r1', role='formal', repeat=1)) + '\n')
canonical = json.loads((native06 / 'run/canonical_repetitions.json').read_text())
(run / 'run/canonical_repetitions.json').write_text(json.dumps(dict(repetitions=canonical['repetitions'][:1])))
print(run)
