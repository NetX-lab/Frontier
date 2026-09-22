"""Synthetic inputs for compare_lanes: ideal vLLM pairing, a matching Frontier
'after' set, the real P0 base G7 cases, and one vLLM round shifted by a dummy."""
import json, shutil, sys
from pathlib import Path
from tests.comparison.stage_admission_pp import compare_lanes

S = Path(sys.argv[1]); BASE = Path('/data/ycfeng/tmp/stage_admission_ordering/base')
D = 0.12  # forward duration
def ideal(n, shift_round=None, rnd=0):
    """Stage 0: lane pair k runs [k*D, (k+1)*D); stage 1 one slot later."""
    fw = []
    for i in range(n):
        lane, k = i % 2, i // 2
        s0 = k * D + (D if (shift_round == rnd and lane == 1) else 0.0)
        fw.append((lane, 0, s0, s0 + D, i)); fw.append((lane, 1, s0 + D, s0 + 2 * D, i))
    return fw
def write_vllm(model, shift):
    d = S / 'vllm' / 'runs' / model; (d / 'dp_placement').mkdir(parents=True)
    reqs, pp, rounds, place, t0 = [], [], [], {0: [], 1: []}, 100.0
    for label, rnd, n in [('warmup', 0, 4)] + [(f'b{b}-r{r}', r, b) for b in (8, 16) for r in range(3)]:
        off = 1.7e9
        for lane, stage, s, e, i in (ideal(n, shift, rnd) if label != 'warmup' else ideal(n)):
            rid = f'{label}-q{i}'
            rec = {'request_ids': [rid], 'pp_rank': stage, 'is_last_rank': stage == 1,
                   'forward_start_ts': t0 + s, 'send_start_ts': None if stage else t0 + e,
                   'timestamp': off + t0 + e}
            pp.append(rec)
            if stage == 0: place[lane].append({'kind': 'engine_iteration', 'engine': lane, 'scheduled_new_req_ids': [rid]})
        for i in range(n):
            reqs.append({'request_id': f'{label}-q{i}', 'burst': label, 'round': rnd, 'index': i, 'rank': i % 2,
                         'num_output_tokens': 1})
        rounds.append({'label': label, 'wall_minus_monotonic_before': off, 'wall_minus_monotonic_after': off})
        t0 += 10.0
    (d / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reqs))
    (d / 'pp_boundary.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pp))
    (d / 'summary.json').write_text(json.dumps({'rounds': rounds}))
    for e, rows in place.items():
        (d / 'dp_placement' / f'dp_placement_{e}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
def write_after(model, n):
    cid = f'G7-{model}-dp2-pp2-n{n}'; d = S / 'frontier' / 'after' / cid; (d / 'metrics' / 'x').mkdir(parents=True)
    (d / 'run.json').write_text(json.dumps({'outcome': 'success'}))
    shutil.copy(BASE / cid / 'case.json', d / 'case.json')
    rows = [{'execution_scope': 'ATTN_DP_LANE', 'replica_local_id': l, 'stage_id': st, 'stage_start_ts': s,
             'stage_end_ts': e, 'request_ids': [str(i)]} for l, st, s, e, i in ideal(n)]
    (d / 'metrics' / 'x' / 'frontier_stage_batch_ledger.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (S / 'frontier' / 'base').mkdir(parents=True, exist_ok=True)
    (S / 'frontier' / 'base' / cid).symlink_to(BASE / cid)
write_vllm('moe', shift=None); write_vllm('dense', shift=1)
for m in ('moe', 'dense'):
    for n in (8, 16): write_after(m, n)
rows, details = compare_lanes.compare(S / 'vllm', S / 'frontier', 'base', 'after')
bad = [(r['check'], r['model'], r['burst'], r['round']) for r in rows if r['status'] != 'MATCH']
print('rows', len(rows)); print('mismatch', bad)
print('dense base n8 stage0', {k: v for k, v in details['runs']['G7-dense-dp2-pp2-n8']['frontier_base']['stage0'].items() if k != 'M3_pairing'})
print('placement', {m: (p['ok'], len(p['unseen'])) for m, p in details['placement'].items()})
