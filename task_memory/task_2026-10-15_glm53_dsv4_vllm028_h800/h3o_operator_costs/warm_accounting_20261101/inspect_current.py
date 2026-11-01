"""Bounded stdlib reads of the existing six replays; no prediction or FIFO run."""
import collections
import gzip
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
TMP = Path('/data/ycfeng/tmp/glm53_dsv4')
result = {}
for case in ('p2_32', 'p2_64'):
    result[case] = {}
    for gen in ('B29', 'pass1_all', 'pass2_all'):
        root = TMP / (f'h3o_b/stage4/a10/{case}_warm_k19b29' if gen == 'B29' else f'a6_pass_replays_20261101/runs/{gen}_{case}')
        config_path = next(root.glob('metrics/**/config.json'))
        config = json.load(open(config_path))
        requests = []
        with gzip.open(root / 'forwards.jsonl.gz', 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r['role'] == 'prefill' and r['step_class'] == 'prefill':
                    requests.append(r)
        assert len(requests) == 128
        segments = {k: statistics.mean(r['segment_ms'].get(k, 0) for r in requests) for k in {k for r in requests for k in r['segment_ms']}}
        scopes = {k: statistics.mean(r['scope_ms'].get(k, 0) for r in requests) for k in {k for r in requests for k in r['scope_ms']}}
        assert max(abs(sum(r['segment_ms'].values()) - r['composed_ms']) for r in requests) < 1e-7
        ledger = [json.loads(l) for l in open(config_path.parent / 'frontier_transfer_ledger.jsonl')]
        sizes = sorted({r['bytes'] for r in ledger if r['transfer_kind'] == 'kv_cache'})
        assert len(sizes) == 1
        transfer = config['kv_cache_transfer_config']
        rate = transfer['network_bandwidth_gbps'] * 1e9 / 8000
        result[case][gen] = dict(root=str(root), n=128, segments=segments, scope_ms=scopes,
            step_ms=statistics.mean(r['composed_ms'] for r in requests),
            dp0=requests[0], bytes=sizes[0], transfer_config=transfer,
            structural_wire_ms=transfer['network_latency_ms'] + sizes[0] / rate,
            packed_wire_ms=transfer['network_latency_ms'] + 46103040 / rate,
            extra_wire_ms=(46103040 - sizes[0]) / rate,
            model_config=config['cluster_config']['prefill_replica_config']['model_config'])
json.dump(result, open(OUT / 'current.json', 'w'), indent=2)
for case, gens in result.items():
    for gen, r in gens.items():
        print(case, gen, 'step', r['step_ms'], 'segments', r['segments'], 'bytes', r['bytes'], 'extra_wire_ms', r['extra_wire_ms'])
