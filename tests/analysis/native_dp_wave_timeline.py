"""Reconstruct formal P-side wave timelines from a persisted native run copy.

python native_dp_wave_timeline.py RUN_COPY OUTPUT_DIRECTORY

Wall timestamps join hosts. Monotonic timestamps remain local evidence; clock
synchronization is not measured. Missing spans are reported, never inferred as
observed events. The historical grouping is the p_blackout.py classification.
"""
import argparse
import bisect
import collections
import json
from pathlib import Path

PROMPT_TOKENS = 6144
COMPUTE_MS = 300.0
EARLY_MS = 260.0
LATE_MS = 760.0
SERVED_SLACK_MS = 100.0
PREVIOUS_LIMIT_S = 2.0


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def classification(schedule, update, previous_end, compute_ms=COMPUTE_MS):
    duration = (update - schedule) * 1000
    if previous_end is None or schedule - previous_end > PREVIOUS_LIMIT_S:
        return 'ungrouped'
    gap = (schedule - previous_end) * 1000
    if gap < EARLY_MS:
        return 'served_at_once' if duration < compute_ms + SERVED_SLACK_MS else 'held'
    if gap < LATE_MS:
        return '260_760_ms'
    return 'after_760_ms'


class Records:
    def __init__(self, rows):
        self.events = collections.defaultdict(list)
        self.requests = collections.defaultdict(dict)
        self.times = {}
        expanded = list(rows)
        for row in rows:
            for event, span in row.get('calls', {}).items():
                expanded.append(dict(event=event,
                    t=row['t']+(span['mono']-row['mono']),
                    end_t=row['t']+(span['end_mono']-row['mono']), **span))
        for row in sorted(expanded, key=lambda r: r.get('t', 0)):
            event = row['event']
            self.events[event].append(row)
            if 'request_id' in row:
                self.requests[event][row['request_id']] = row
        for event, values in self.events.items():
            self.times[event] = [r.get('t', 0) for r in values]

    def between(self, event, start, end):
        values, times = self.events[event], self.times.get(event, [])
        return values[bisect.bisect_left(times, start):bisect.bisect_right(times, end)]

    def previous(self, event, when):
        index = bisect.bisect_right(self.times.get(event, []), when) - 1
        return self.events[event][index] if index >= 0 else None


def prefills(rows, rank):
    pending, adds, steps = {}, {}, []
    for row in rows:
        if row['event'] == 'add':
            adds[row['request_id']] = row
        elif row['event'] == 'schedule' and row.get('tokens') == PROMPT_TOKENS:
            pending[row['step']] = row
        elif row['event'] == 'update' and row.get('step') in pending:
            schedule = pending.pop(row['step'])
            for request_id, _, tokens in schedule.get('new', []):
                if tokens != PROMPT_TOKENS:
                    continue
                steps.append(dict(lane=rank, request_id=request_id,
                    warmup='warmup' in request_id, schedule=schedule, update=row,
                    add=adds.get(request_id)))
    return steps


def matching_sync(records, request_id):
    executes = [r for r in records.events['worker_execute_model']
                if request_id in r.get('request_ids', [])]
    for execute in executes:
        syncs = records.between('dp_padding_sync', execute['t'], execute['end_t'])
        for sync in syncs:
            if sync.get('num_tokens') == PROMPT_TOKENS:
                reduces = [r for r in records.between('dp_cpu_all_reduce', sync['t'], sync['end_t'])
                           if r.get('forward') == sync['forward']]
                if len(reduces) == 1:
                    return execute, sync, reduces[0]
    return None, None, None


def candidate_intervals(schedule_t, update_end, lanes, real_reduce, compute_ms, real_lane=None):
    """Follow the lane that joins last; separate rendezvous from reduction time."""
    if real_reduce is None or any(lane.get('reduce') is None for lane in lanes):
        return None, 'unavailable'
    all_lanes = lanes + ([real_lane] if real_lane else [])
    critical = max(all_lanes, key=lambda lane: lane['reduce']['t'])
    if any(critical.get(key) is None for key in ('enqueue', 'handle', 'dummy')):
        return None, 'unavailable'
    enqueue, handle, dummy, reduce = (critical[k] for k in ('enqueue', 'handle', 'dummy', 'reduce'))
    boundaries = [schedule_t, enqueue['t'], handle['t'], handle['end_t'],
                  dummy['t'], reduce['t'], max(lane['reduce']['end_t'] for lane in all_lanes)]
    # A broadcast can precede the real lane's schedule. That interval has zero wait.
    boundaries[1:5] = [max(schedule_t, t) for t in boundaries[1:5]]
    if any(b < a for a, b in zip(boundaries, boundaries[1:])):
        return None, 'clock_order_unverified'
    spans = dict(message=(boundaries[1] - boundaries[0]) * 1000,
        input_queue=(boundaries[2] - boundaries[1]) * 1000,
        start_wave_handling=(boundaries[3] - boundaries[2]) * 1000,
        idle_engine=(boundaries[4] - boundaries[3]) * 1000,
        worker_dispatch=(boundaries[5] - boundaries[4]) * 1000,
        all_reduce=(boundaries[6] - boundaries[5]) * 1000,
        after_all_reduce=max(0, (update_end - real_reduce['end_t']) * 1000 - compute_ms))
    spans['message'] += spans.pop('start_wave_handling')
    labels = dict(message='start_wave_message', input_queue='start_wave_message',
        idle_engine='idle_engine', worker_dispatch='worker_join',
        all_reduce='all_reduce', after_all_reduce='after_all_reduce')
    winner = max(spans, key=spans.get)
    return dict(critical_lane=critical['lane'], milliseconds=spans,
        after_all_reduce_total_ms=(update_end - real_reduce['end_t']) * 1000,
        compute_baseline_ms=compute_ms), labels[winner]


def timeline(step, previous_end, engines, workers, proxy, compute_ms):
    rank, rid = step['lane'], step['request_id']
    schedule, update = step['schedule'], step['update']
    s, e = schedule['t'], update['t']
    engine, worker = engines[rank], workers.get(rank, Records([]))
    add = step['add']
    wave_end = engines[min(engines)].previous('wave_complete', s)
    engine_add = engine.requests['engine_add'].get(rid)
    execute, sync, real_reduce = matching_sync(worker, rid)
    lanes = []
    wave = engine_add.get('current_wave_after') if engine_add else None
    for peer in sorted(engines):
        if peer == rank:
            continue
        messages = [r for r in engines[peer].between('start_dp_wave_enqueue',
                    wave_end['t'] if wave_end else s - PREVIOUS_LIMIT_S, e)
                    if r.get('wave') == wave]
        enqueue = messages[-1] if messages else None
        handled = [r for r in engines[peer].between('handle_client_request',
                   enqueue['t'] if enqueue else s, e)
                   if r.get('request_type') == 'START_DP_WAVE' and r.get('wave') == wave]
        handle = handled[0] if handled else None
        dummies = engines[peer].between('engine_dummy_batch', handle['end_t'] if handle else s, e)
        dummy = dummies[0] if dummies else None
        peer_worker = workers.get(peer, Records([]))
        peer_sync = [r for r in peer_worker.events['dp_padding_sync']
                     if sync and r.get('forward') == sync['forward'] and
                     r.get('stage') == sync.get('stage') and
                     (wave_end is None or r['t'] >= wave_end['t']) and r['t'] <= sync['end_t']]
        peer_reduce = [r for r in peer_worker.events['dp_cpu_all_reduce']
                       if len(peer_sync)==1 and r.get('forward')==sync['forward']
                       and peer_sync[0]['t'] <= r['t'] <= peer_sync[0]['end_t']]
        lanes.append(dict(lane=peer, enqueue=enqueue, handle=handle, dummy=dummy,
            sync=peer_sync[0] if len(peer_sync) == 1 else None,
            reduce=peer_reduce[0] if len(peer_reduce) == 1 else None))
    step_calls = [r for r in engine.between('process_engine_step', s-PREVIOUS_LIMIT_S, s)
                  if r['t'] <= s <= r['end_t']]
    real_lane = dict(lane=rank, enqueue=engine.requests['add_enqueue'].get(rid),
                    handle=engine_add, dummy=step_calls[-1] if step_calls else execute, reduce=real_reduce)
    intervals, placement = candidate_intervals(s, e + update.get('dur_ms', 0) / 1000,
        lanes, real_reduce, compute_ms, real_lane)
    rendezvous_verified = bool(sync and real_reduce and
        len(engines) == sync['dp_size'] and len(workers) == sync['dp_size'] and
        real_reduce.get('reduced_values') is not None and
        all(lane['reduce'] and lane['reduce'].get('reduced_values') == real_reduce['reduced_values']
            for lane in lanes))
    if intervals is not None and not rendezvous_verified:
        intervals, placement = None, 'rendezvous_unverified'
    proxy_row = proxy.get(rid, {})
    complete = all((add, engine.requests['add_enqueue'].get(rid), engine_add, wave_end,
                    execute, sync, real_reduce, rid in update.get('first_token', [])))
    complete &= all(all(lane.get(k) for k in ('enqueue', 'handle', 'dummy', 'sync', 'reduce')) for lane in lanes)
    complete &= intervals is not None and rendezvous_verified
    arrival = proxy_row.get('proxy_arrival_time', add.get('arrival_time') if add else None)
    complete &= arrival is not None
    return dict(request_id=rid, lane=rank,
        classification=classification(s, e, previous_end, compute_ms),
        previous_prefill_end=previous_end, previous_wave_complete=wave_end,
        arrival=arrival, rendezvous_verified=rendezvous_verified,
        enqueue=engine.requests['add_enqueue'].get(rid), engine_add=engine_add,
        scheduler_add=add, schedule=schedule, worker_execute=execute,
        sync=sync, reduce=real_reduce, other_lanes=lanes, first_token_update=update,
        intervals=intervals, delay_candidate=placement, complete=bool(complete),
        step_duration_ms=(e-s)*1000, wait_ms=max(0, (e-s)*1000-compute_ms))


def analyze(run, compute_ms=COMPUTE_MS):
    engines, workers, steps, errors = {}, {}, [], []
    for path in sorted((run / 'nodes').glob('*/scheduler_lifecycle/*.jsonl')):
        rows = read_rows(path)
        start = next(r for r in rows if r['event'] == 'start')
        errors.extend(dict(path=str(path), **r) for r in rows if r['event'] == 'error')
        if start.get('role') == 'prefill':
            rank = start['dp_rank']
            if rank in engines:
                raise ValueError(f'Duplicate prefill core for dp{rank}')
            engines[rank] = Records(rows)
            steps.extend(prefills(rows, rank))
    for path in sorted((run / 'nodes').glob('*/worker_dp_wave/*.jsonl')):
        rows = read_rows(path)
        start = next(r for r in rows if r['event'] == 'worker_wave_start')
        errors.extend(dict(path=str(path), **r) for r in rows if r['event'] == 'error')
        if start.get('role') == 'prefill':
            rank = start['dp_rank']
            if rank in workers:
                raise ValueError(f'Duplicate TP0 worker for dp{rank}')
            workers[rank] = Records(rows)
    proxy = {}
    for path in (run / 'run/proxy_requests.jsonl', run / 'proxy_requests.jsonl'):
        if path.exists():
            for row in read_rows(path):
                rid = row.get('kv_transfer_params', {}).get('remote_request_id')
                if rid:
                    proxy[rid] = row
    steps.sort(key=lambda step: step['schedule']['t'])
    timelines = [timeline(step, steps[i-1]['update']['t'] if i else None,
                 engines, workers, proxy, compute_ms)
                 for i, step in enumerate(steps) if not step['warmup']]
    counts = collections.Counter(row['classification'] for row in timelines)
    summary = dict(formal_prefills=len(timelines), classification=dict(counts),
        complete_timelines=sum(row['complete'] for row in timelines),
        candidates=dict(collections.Counter(row['delay_candidate'] for row in timelines)),
        errors=errors, clock_contract='Wall timestamps across hosts; clock synchronization unmeasured',
        coordinator_coverage='unavailable; source to engine enqueue remains combined',
        interval_semantics='CPU spans; after_all_reduce includes device work and output/update dispatch')
    return timelines, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compute-ms', type=float, default=COMPUTE_MS)
    args = parser.parse_args()
    timelines, summary = analyze(args.run, args.compute_ms)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'prefill_timelines.jsonl').open('w') as stream:
        for row in timelines:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
