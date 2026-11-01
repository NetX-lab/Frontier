"""Bind NIXL compatibility events to first-contact requests, without latency cuts.

Usage: identify_openers.py --run LABEL=ARTIFACTS [--run ...] --output NEW_JSON
Decode ownership comes from the retained serving inventory, not scheduler
lifecycle records. Lifecycle records are read only afterwards as ground truth.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

UTC = dt.timezone.utc
STAMP = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)")
HANDSHAKE = re.compile(r"Worker_DP(\d+)_TP(\d+).*NIXL compatibility check passed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(request):
    return request.split("cmpl-", 1)[1].rsplit("-0-", 1)[0]


def iso(t):
    return dt.datetime.fromtimestamp(t, UTC).isoformat()


def identify(run):
    sources = []

    def load(path):
        sources.append(path)
        return json.loads(path.read_text())

    manifest = load(run / "run_manifest.json")
    year = manifest["written_before_launch_utc"][:4]
    phases = load(run / "run/canonical_repetitions.json")["repetitions"]
    sources.extend((run / "run/client_requests.jsonl", run / "run/proxy_requests.jsonl"))
    client = {r["request_id"]: r for r in rows(sources[-2])}
    proxy = {r["request_id"]: r for r in rows(sources[-1])}
    require(set(client) == set(proxy), "Client/proxy identities differ")
    owners, lane_paths, logs = {}, {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(run.glob("nodes/*/serving/rank0.json")):
        record = load(path)
        if record["role"] != "decode":
            continue
        lane = record["inventory"]["request_owner"]
        require(lane not in lane_paths and record["counters"]["dropped_rows"] == 0, "Bad decode inventory")
        lane_paths[lane] = path
        for group in record["counters"]["request_ids"]:
            for request in group["request_ids"]:
                rid = key(request)
                require(rid not in owners, "Duplicate decode request owner")
                owners[rid] = lane
    require(set(owners) == set(proxy), "Incomplete decode ownership")
    anchors = []
    for path in sorted(run.glob("nodes/*/server_ready.json")):
        record = load(path)
        log_path = path.with_name("server.log")
        starts = []
        for number, line in enumerate(log_path.open(), 1):
            if "Starting vLLM server on " in line and (stamp := STAMP.search(line)):
                t = dt.datetime.strptime(year + "-" + stamp[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
                starts.append(dict(line=number, local_stamp=stamp[1], utc_second=t))
        require(starts, f"Missing server clock anchor: {path}")
        ready = dt.datetime.fromisoformat(record["ready_at_utc"]).timestamp()
        delta = ready - max(s["utc_second"] for s in starts)
        require(0 <= delta < 5, f"Server clock/readiness inconsistent: {path}")
        anchors.append(dict(path=str(path), ready_at_utc=record["ready_at_utc"], log=str(log_path),
                            starts=starts, ready_minus_last_start_second_s=delta))
    windows, offsets = [], []
    for repeat in phases:
        for phase, origin in repeat["phases"].items():
            offset = origin["origin_wall"] - origin["origin_monotonic"]
            offsets.append(offset)
            selected = [r for r in client.values() if r["repeat"] == repeat["repeat"] and r["segment"] == phase]
            require(selected, "Empty phase")
            windows.append(dict(repeat=repeat["repeat"], phase=phase, origin_wall=origin["origin_wall"],
                                origin_monotonic=origin["origin_monotonic"], wall_minus_monotonic_s=offset,
                                end_wall=max(r["last_token_monotonic"] for r in selected) + offset))
    proxy_offsets = [r[f"{field}_time"] - r[f"{field}_monotonic"] for r in proxy.values()
                     for field in ("proxy_arrival", "prefill_response", "decode_first_byte", "decode_response")]
    require(max([*offsets, *proxy_offsets]) - min([*offsets, *proxy_offsets]) < 0.001, "Client/proxy clock drift")
    first = {}
    for record in sorted(proxy.values(), key=lambda r: r["prefill_response_time"]):
        pair = (record["kv_transfer_params"]["remote_engine_id"], owners[record["request_id"]])
        first.setdefault(pair, record["request_id"])
    for path in sorted(run.glob("nodes/*/server.log")):
        sources.append(path)
        for number, line in enumerate(path.open(), 1):
            if match := HANDSHAKE.search(line):
                lane, tp = map(int, match.group(1, 2))
                stamp = STAMP.search(line)[1]
                t = dt.datetime.strptime(year + "-" + stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
                require(lane in lane_paths and lane_paths[lane].parent.parent == path.parent, "Log/owner lane mismatch")
                logs[lane][tp].append(dict(path=str(path), line=number, local_stamp=stamp, utc_second=t))
    events, formal = [], []
    for lane, workers in sorted(logs.items()):
        require(set(workers) == set(range(8)), "Expected all eight TP workers")
        expected = sum(d == lane for _, d in first)
        require(all(len(records) == expected for records in workers.values()), "Handshake/pair count differs")
        for ordinal in range(expected):
            records = [workers[tp][ordinal] for tp in range(8)]
            times = [r["utc_second"] for r in records]
            matching = [w for w in windows if min(times) >= w["origin_wall"] and max(times) + 1 <= w["end_wall"]]
            require(len(matching) == 1, "Handshake phase/window ambiguous")
            window = matching[0]
            candidates = [rid for rid, r in proxy.items() if owners[rid] == lane and all(
                t < r["decode_first_byte_time"] and t + 1 > r["prefill_response_time"] for t in times)]
            bound = None
            if window["phase"] == "formal":
                require(len(candidates) == 1, f"Ambiguous formal event D{lane}: {candidates}")
                bound = candidates[0]
                pair = (proxy[bound]["kv_transfer_params"]["remote_engine_id"], lane)
                require(first[pair] == bound, f"Formal event does not bind pair opener: {bound}")
                require(client[bound]["repeat"] == window["repeat"] and client[bound]["segment"] == "formal", "Phase mismatch")
                formal.append(bound)
            event = dict(decode_lane=lane, ordinal_per_tp=ordinal, repeat=window["repeat"], phase=window["phase"],
                         interval_utc=[iso(min(times)), iso(max(times) + 1)], tp_records=records,
                         candidate_request_ids=sorted(candidates), bound_request_id=bound,
                         binding="UNIQUE_FORMAL_PAIR_OPENER" if bound else "WARMUP_WINDOW_ONLY_NO_EXCLUSION")
            if bound:
                r = proxy[bound]
                event.update(remote_engine_id=r["kv_transfer_params"]["remote_engine_id"],
                             prefill_lane=int(r["kv_transfer_params"]["remote_engine_id"].rsplit("_dp", 1)[1]),
                             prefill_response_utc=iso(r["prefill_response_time"]), decode_first_byte_utc=iso(r["decode_first_byte_time"]),
                             proxy_handoff_ms=(r["decode_first_byte_monotonic"] - r["prefill_response_monotonic"]) * 1000)
            events.append(event)
    expected_formal = sorted(rid for rid in first.values() if client[rid]["segment"] == "formal")
    require(sorted(formal) == expected_formal, "Formal event coverage differs from first-contact pairs")
    validation = dict(status="NOT_APPLICABLE_CLEAN_NO_LIFECYCLE")
    lifecycle_paths = sorted(run.glob("nodes/*/scheduler_lifecycle/*.jsonl"))
    if lifecycle_paths:
        p_lanes, adds = {}, []
        for path in lifecycle_paths:
            sources.append(path)
            records = rows(path)
            start = next(r for r in records if r["event"] == "start")
            for record in records:
                if record["event"] == "add":
                    if start["role"] == "prefill":
                        p_lanes[record["request_id"]] = start["dp_rank"]
                    else:
                        adds.append((record["t"], record["request_id"], record["remote_request_id"], start["dp_rank"]))
        seen, truth = set(), []
        for _, request, remote, lane in sorted(adds):
            pair = (p_lanes[remote], lane)
            require(owners[key(request)] == lane, "Lifecycle/serving owner mismatch")
            if pair not in seen:
                seen.add(pair)
                if client[key(request)]["segment"] == "formal":
                    truth.append(key(request))
        require(sorted(truth) == sorted(formal), "Lifecycle ground truth differs")
        validation = dict(status="PASS_EXACT", lifecycle_formal_openers=sorted(truth), derived_formal_openers=sorted(formal))
    return dict(run=str(run), mode=manifest["mode"], native_commit=manifest["groundtruth_commit"],
                clock=dict(server_local_to_utc_offset_s=0, resolution_s=1, anchors=anchors,
                           client_proxy_wall_minus_monotonic_range_s=[min([*offsets, *proxy_offsets]), max([*offsets, *proxy_offsets])],
                           rule="UTC civil seconds established by server start/readiness anchors; client/proxy phase epoch anchors map monotonic seconds."),
                windows=windows, events=events, first_contact_pairs=[dict(remote_engine_id=p, decode_lane=d, request_id=rid)
                    for (p, d), rid in sorted(first.items())], formal_openers=sorted(formal), validation=validation,
                counts=dict(handshake_log_lines=sum(len(rs) for w in logs.values() for rs in w.values()),
                            events=len(events), warmup_events=sum(e["phase"] == "warmup" for e in events), formal_events=len(formal)),
                sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(sources))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {label: identify(Path(path)) for label, path in (s.split("=", 1) for s in args.run)}
    with args.output.open("x") as stream:
        stream.write(json.dumps(dict(decision="D-WARM-TARGET", runs=runs,
             rule="Every formal eight-TP compatibility cohort must overlap exactly one same-lane proxy handoff, "
                  "which must be the first request from that remote engine. No latency threshold. "
                  "Warmup cohorts retain candidate lists, not unproved per-engine bindings."), indent=2) + "\n")
    for label, result in runs.items():
        print(label, result["counts"], result["validation"]["status"], result["formal_openers"])


if __name__ == "__main__":
    main()
