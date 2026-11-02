"""Split the H4b-(a) D-side handoff into parts a-g (design.md sections 1, 4 and 8).

    python3 analyze_lifecycle.py <run_root> <reference_run_root> <output_dir>

Both run roots are local copies of ``case/runs/groundtruth_clean/<run tag>``;
the engine config is read from the case inputs three levels above. The
reference (native06 on H800, P2-32r on H200) is used for acceptance item 3
and the proxy handoff.

Per formal request, joined through ``kv_transfer_params.remote_request_id``:
- c, d, e, f on the D engine host's monotonic clock: ``add`` -> schedule that
  requests the NIXL READ -> update reporting it in ``finished_recving`` ->
  schedule that promotes the request -> end of the update with its first token;
- the clock-free remainder a + b + g: the proxy's P response -> first D byte
  interval minus the engine interval c + d + e + f;
- a, b, g from wall clocks on different hosts (proxy, D API server, D engine core).
"""

import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy

PARTS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'engine_ms', 'remainder_ms', 'proxy_handoff_ms', 'iterations',
         'd_iterations', 'read_transfer_max_ms')
CLIENT_METRICS = ('ttft_ms', 'tpot_ms', 'e2e_ms')


def jsonl(path, problems):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A process killed during a buffered write leaves a partial last line.
            problems.append(f'{path.name}:{number} is not JSON')
    return rows


def returned(record):
    """Monotonic time at which the scheduler call returned."""
    return record['mono'] + record['dur_ms'] / 1e3


def load_cores(run_root, problems):
    cores = {}
    for path in sorted(run_root.glob('nodes/*/scheduler_lifecycle/dp*_pid*.jsonl')):
        rows = jsonl(path, problems)
        starts = [row for row in rows if row['event'] == 'start']
        if len(starts) != 1:
            problems.append(f'{path}: {len(starts)} start records')
            continue
        key = (starts[0]['role'], starts[0]['dp_rank'])
        if key in cores:
            problems.append(f'{key}: more than one lifecycle file (engine core restarted)')
            continue
        stops = [row for row in rows if row['event'] == 'stop']
        errors = max([sum(row['event'] == 'error' for row in rows)] + [stop['errors'] for stop in stops])
        cores[key] = dict(path=str(path.relative_to(run_root)), host=starts[0]['host'], rows=rows,
                          errors=errors, stopped=bool(stops))
    return cores


def decode_timelines(rows):
    """Per D request: its add, READ, finished_recving, promotion and first-token records.

    The worker-side NIXL stats are not keyed by request. Each READ collects the stats of every update in its window
    (READ schedule, finished_recving update]; ``read_window_shared`` marks a window that overlaps another READ.
    """
    requests, active, reading, schedule_monos, decoding = {}, set(), set(), [], False
    for row in rows:
        event = row['event']
        if event == 'add':
            # The lane decodes when its last step before the arrival ran tokens.
            requests[row['request_id']] = dict(add=row, active_at_add=len(active), decoding=decoding)
            active.add(row['request_id'])
        elif event == 'schedule':
            schedule_monos.append(row['mono'])
            decoding = 'tokens' in row
            for request_id in row.get('recv', ()):
                if 'read' not in requests[request_id]:
                    requests[request_id].update(read=row, read_stats=[], read_window_shared=False)
                    reading.add(request_id)
            for request_id, computed, scheduled in row.get('new', ()):
                if request_id in requests and 'promotion' not in requests[request_id]:
                    requests[request_id].update(promotion=row, promotion_tokens=[computed, scheduled])
            active.difference_update(row.get('finished', ()))
        elif event == 'update':
            if 'kv_stats' in row:
                for request_id in reading:
                    requests[request_id]['read_stats'].append(row['kv_stats'])
                    requests[request_id]['read_window_shared'] |= len(reading) > 1
            for request_id in row.get('finished_recving', ()):
                requests[request_id].setdefault('received', row)
                reading.discard(request_id)
            for request_id in row.get('first_token', ()):
                requests[request_id].setdefault('first', row)
            active.difference_update(row.get('done', ()))
    for timeline in requests.values():
        if 'promotion' in timeline:
            # Engine iterations from the arrival through the promotion step.
            timeline['iterations'] = (bisect_right(schedule_monos, timeline['promotion']['mono'])
                                      - bisect_left(schedule_monos, timeline['add']['mono']))
        if 'received' in timeline:
            # Engine iterations scheduled after the READ step, up to the finished_recving update.
            timeline['d_iterations'] = (bisect_right(schedule_monos, timeline['received']['mono'])
                                        - bisect_right(schedule_monos, timeline['read']['mono']))
    return requests


def request_row(client, proxy, timeline, core_key, core):
    row = dict(request_id=client['request_id'], repeat=client['repeat'],
               proxy_handoff_ms=(proxy['decode_first_byte_monotonic'] - proxy['prefill_response_monotonic']) * 1e3)
    if timeline is None:
        return row | dict(missing=['add'])
    missing = [name for name in ('read', 'received', 'promotion', 'first') if name not in timeline]
    add = timeline['add']
    row.update(d_dp_rank=core_key[1], d_host=core['host'], active_at_add=timeline['active_at_add'],
               # The D API server derives its request id from the proxy's X-Request-Id.
               d_request_id_matches=add['request_id'].startswith(f"cmpl-{client['request_id']}-"),
               lane_state='decoding' if timeline['decoding'] else 'idle', missing=missing)
    if missing:
        return row
    read, received, promotion, first = (timeline[name] for name in ('read', 'received', 'promotion', 'first'))
    first_returned_wall = first['t'] + first['dur_ms'] / 1e3
    transfers = [value * 1e3 for stats in timeline['read_stats'] for value in stats.get('transfer_duration', ())]
    row.update(
        a=(add['arrival_time'] - proxy['prefill_response_time']) * 1e3,
        b=(add['t'] - add['arrival_time']) * 1e3,
        c=(read['mono'] - add['mono']) * 1e3,
        d=(received['mono'] - read['mono']) * 1e3,
        e=(promotion['mono'] - received['mono']) * 1e3,
        f=(returned(first) - promotion['mono']) * 1e3,
        g=(proxy['decode_first_byte_time'] - first_returned_wall) * 1e3,
        engine_ms=(returned(first) - add['mono']) * 1e3,
        iterations=timeline['iterations'],
        d_iterations=timeline['d_iterations'],
        read_waiting_remote_kv=add['request_id'] in read.get('waiting_remote_kv', ()),
        promotion_tokens=timeline['promotion_tokens'],
        received_with=len(received['finished_recving']),
        read_window_shared=timeline['read_window_shared'],
        read_transfer_duration_ms=transfers,
        read_transfer_max_ms=max(transfers, default=None),
        read_bytes=sum(value for stats in timeline['read_stats'] for value in stats.get('bytes_transferred', ())))
    row['remainder_ms'] = row['proxy_handoff_ms'] - row['engine_ms']
    return row


def summary(rows):
    out = {}
    for name in PARTS:
        values = [row[name] for row in rows if row.get(name) is not None]
        if values:
            out[name] = dict(n=len(values), mean=float(numpy.mean(values)),
                             p50=float(numpy.percentile(values, 50)), p95=float(numpy.percentile(values, 95)),
                             min=float(min(values)), max=float(max(values)))
    return out


def client_metrics(run_root):
    """Per repetition: the harness's client metrics and the mean proxy handoff of formal requests."""
    repetitions = json.loads((run_root / 'run/canonical_repetitions.json').read_text())['repetitions']
    clients = {row['request_id']: row for row in map(json.loads, (run_root / 'run/client_requests.jsonl')
                                                      .read_text().splitlines())}
    handoff = {}
    for proxy in map(json.loads, (run_root / 'run/proxy_requests.jsonl').read_text().splitlines()):
        client = clients[proxy['request_id']]
        if client['role'] == 'formal':
            handoff.setdefault(client['repeat'], []).append(
                (proxy['decode_first_byte_monotonic'] - proxy['prefill_response_monotonic']) * 1e3)
    return {repetition['repeat']: {name: repetition['metrics'][name]['mean'] for name in CLIENT_METRICS}
            | dict(total_thpt_tokens_s=repetition['metrics']['total_thpt_tokens_s'],
                   proxy_handoff_ms=float(numpy.mean(handoff[repetition['repeat']])))
            for repetition in repetitions}


def main(run_root, reference_root, output_dir):
    problems = []
    engine = json.loads((run_root.parents[2] / 'inputs/engine.json').read_text())
    expected = {(role, rank) for role in ('prefill', 'decode') for rank in range(engine['data_parallel_size'])}
    cores = load_cores(run_root, problems)
    clients = [json.loads(line) for line in (run_root / 'run/client_requests.jsonl').read_text().splitlines()]
    proxies = {row['request_id']: row for row in
               map(json.loads, (run_root / 'run/proxy_requests.jsonl').read_text().splitlines())}
    owners = {}
    for key, core in cores.items():
        if key[0] == 'decode':
            for request_id, timeline in decode_timelines(core['rows']).items():
                owners[timeline['add'].get('remote_request_id')] = (key, core, timeline)
    rows = []
    for client in (row for row in clients if row['role'] == 'formal'):
        proxy = proxies[client['request_id']]
        key, core, timeline = owners.get(proxy['kv_transfer_params']['remote_request_id'], (None, None, None))
        rows.append(request_row(client, proxy, timeline, key, core))

    observed, reference = client_metrics(run_root), client_metrics(reference_root)
    ranges = {name: [min(values[name] for values in reference.values()),
                     max(values[name] for values in reference.values())]
              for name in next(iter(reference.values()))}
    complete = [row for row in rows if 'missing' in row and not row['missing']]
    acceptance_2 = dict(
        cores_present=sorted(map(list, cores)), cores_expected=sorted(map(list, expected)),
        all_cores=set(cores) == expected,
        no_observer_errors=all(core['errors'] == 0 for core in cores.values()),
        stop_records=sum(core['stopped'] for core in cores.values()),
        formal_requests=len(rows), complete_timelines=len(complete),
        d_request_ids_match=sum(row.get('d_request_id_matches', False) for row in rows),
        read_in_waiting_remote_kv=all(row['read_waiting_remote_kv'] for row in complete),
        promotion_k1=all(row['promotion_tokens'][1] == 1 for row in complete),
        incomplete=[[row['request_id'], row['missing']] for row in rows if row.get('missing')][:20],
        problems=problems)
    acceptance_2['passed'] = (acceptance_2['all_cores'] and acceptance_2['no_observer_errors']
                              and len(complete) == len(rows) > 0 and acceptance_2['read_in_waiting_remote_kv']
                              and acceptance_2['d_request_ids_match'] == len(rows)
                              and not problems)
    result = dict(
        run_root=str(run_root), reference_root=str(reference_root),
        acceptance_2_observer_records=acceptance_2,
        acceptance_3_client_metrics=dict(
            run=observed, reference=reference, reference_range=ranges,
            outside_reference_range={repeat: [name for name, value in values.items()
                                              if not ranges[name][0] <= value <= ranges[name][1]]
                                     for repeat, values in observed.items()}),
        acceptance_4_parts={state: summary([row for row in complete if state in ('all', row['lane_state'])])
                            for state in ('all', 'idle', 'decoding')}
                           | dict(by_d_dp_rank={rank: summary([row for row in complete if row['d_dp_rank'] == rank])
                                                for rank in sorted({row['d_dp_rank'] for row in complete})},
                                  by_repeat={repeat: summary([row for row in complete if row['repeat'] == repeat])
                                             for repeat in sorted({row['repeat'] for row in complete})},
                                  proxy_handoff_all_formal=summary(rows)['proxy_handoff_ms']))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'lifecycle_requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (output_dir / 'lifecycle_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(acceptance_2=acceptance_2['passed'], complete=len(complete), formal=len(rows),
                          outside_reference_range=result['acceptance_3_client_metrics']['outside_reference_range']),
                     indent=2))


if __name__ == '__main__':
    main(*(Path(argument).resolve() for argument in sys.argv[1:4]))
