import json, sys, glob, collections
A = sys.argv[1]
for p in sorted(glob.glob(A + '/nodes/*/worker_dp_wave/*.jsonl')):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    s = next(x for x in rows if x['event'] == 'worker_wave_start')
    c = collections.Counter()
    for x in rows:
        if x['event'] in ('worker_dummy_batch', 'worker_execute_model'):
            sy = x.get('calls', {}).get('dp_padding_sync')
            if sy:
                c[(x['event'][7:11], sy.get('stage'), sy.get('desired_graph_mode'), sy.get('graph_mode'),
                   sy.get('padded_num_tokens'), sy.get('dp_size'))] += 1
    print(s['role'], s['dp_rank'], s['host'][:13], dict(c))
