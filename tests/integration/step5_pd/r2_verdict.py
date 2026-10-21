"""Offline, diagnostic R2 checks, not a ticket-08 acceptance or a job controller.

Contract (JSONL in evidence/nodes/<host>/pd_state/):

* Every record has event, role (prefill/decode), host, pid, dp_rank,
  monotonic_ns (Linux CLOCK_MONOTONIC), request_id (native), and engine_id.
  Worker events also have tp_rank in 0..7. Frontend selections may omit engine_id.
* target_content identifies its source by remote_engine_id/remote_request_id.
  Other D events may repeat these fields; contradictory identities are failures.
  Content events have group_id, layer, component (raw_kv/summary_mean), flat
  local_block_ids, block_hashes (SHA-256 of each COMPLETE physical native row),
  and block_bytes (bytes per row). target_content additionally has flat
  remote_block_ids in the same wire order. Source content may be a superset.
* source_hold/source_free and target_reserve/target_free are scheduler events,
  with block_ids = [[physical IDs for group 0], ...]. read_submit is a D worker
  event with nested local_block_ids/remote_block_ids: the actual submitted wire
  mapping, including any SWA suffix selection. Target content must match it,
  NOT a prefix or equal P/D page IDs. Free events include integer
  pool_free_before/pool_free_after and counts_scope="request_release". With
  engine.enable_prefix_caching=false, the increase must equal the number of
  unique non-null physical pool IDs released across all groups (block 0 is
  null; aliases are counted once). Optional released_block_count must agree.
  Missing/aggregate count scope or unknown prefix-caching mode is UNVERIFIED,
  not evidence of per-request conservation.
* D workers emit read_submit/read_network_done/read_gpu_visible, then one or
  more target_compute events. Each compute gives positions (zero-based token
  positions), or num_computed_tokens and num_scheduled_tokens (a contiguous
  range). If both forms are present they must agree. target_reserve additionally
  gives num_prompt_tokens, num_external_tokens, client_request_id and wave.
   source_hold.num_computed_tokens is the actual P computed prompt N, NOT the
   loaded prefix M. Its kv_xfer_params gives remote_num_tokens=N and advertised
   remote_block_ids. M = B * floor((N-B)/B), with engine.block_size B. An explicit
   num_matched_tokens, if supplied, describes M separately. Each D read_metadata
   gives remote_num_tokens, advertised_remote_block_ids, reserved_local_block_ids,
   and local_physical_block_ids. Missing source metadata is never reconstructed
   from the actual READ or its hashes.
* P workers emit source_notified for the native P request, one per TP rank.
  This request identity is the native notification correlation across hosts.
  No ordering comparison ever uses clocks from different hosts.
* frontend_select includes client_request_id, wave, dp_rank (the selected
  lane), and load_snapshot (nonempty list/dict of observed native loads).
  Its request_id is the selected role's native request ID.
* config.json pd_state_observation.expected_layers is a complete independent
  census: [{"group_id": 0, "layer": "name", "components": ["raw_kv", ...]}].
  A raw-only layer is explicit. Alternatively each worker emits
  state_registration with complete=true and expected_layers in this format.
  Registration is an engine-wide census, so request_id may be absent there.
  region_size and page_size are explicit pd_state_observation integers;
  page_size may instead come from engine.block_size. No layer census or region
  size is inferred from the content samples. trace/*/request_ids.json is the
  independent request census, using the existing stage format.
  The runtime collector's spellings are also understood: blocks,
  pool_free_blocks_before/after, selected_client_request_id, and lb_engines.
  Its registration event supplies independent expected_raw_layers (name->group)
  and expected_summary_layers (registration name->main layer); the frozen
  expected_layer_count/expected_summary_layer_count must match those censuses.
  Native num_external_computed_tokens supplies the loaded prefix. Client joins
   supply the frozen wave/prompt length, not an invented observed prompt count.
   Frozen or complete worker registration group_specs entries give group_id
   (transfer ordinal), native_group_id (scheduler table ordinal), block_size,
   and explicit sliding_window (null for full history, positive token count for
   SWA). Alternatively transfer_group_ids supplies the scheduler ordinals. A
   missing map is UNVERIFIED, not an assumed identity. Physical IDs are resolved
   in the independent complete scheduler tables without compacting null block 0.
   Full-history coverage is [0,M/B); current SWA coverage starts at
   floor(max(0,N-sliding_window)/B). Unsupported/missing group semantics remain
   UNVERIFIED; exact coverage is not inferred from the submitted descriptors.
* native_read_ready is emitted on native get_transfer_results.finished_recving,
   before any diagnostic fence. It requires observer_fence=false and a boundary
   string naming the native completion. read_gpu_visible is diagnostic, with
   observer_fence=true, and cannot substitute for this native witness.
   Explicit content_client_request_ids/readiness_only_client_request_ids must
   partition selected_client_request_ids and the frozen trace. Only the former
   require hashes; every request still needs eight-rank mapping/token/lifetime
   evidence. Readiness-only requests must have no pre-forward diagnostic fence
   or CPU readback, and must cover both frozen sequential probes plus a native
   multi-request wave on both P/D lanes and D endpoints. Legacy mask-free data
   still requires native_read_ready; it cannot establish no-fence wave coverage.
* native_error is an explicit runtime failure (message); observation_error is
   an instrumentation gap. Unknown events/invalid JSON are gaps, not success.
   observer_started/hook_installed are lifecycle metadata, not worker witnesses.
* Cancellation is planned by trace cancel_after_first_content=true. The matching
   evidence/waves/<wave>/client_requests.jsonl row must have client_cancelled=true
   (the historical evidence/replay/<wave> path is also read)
  and must not claim stream_completed=true. The D target_free must explicitly
  report request_status, finish_reason, or status as FINISHED_ABORTED (also
  accepting RequestStatus.FINISHED_ABORTED). Missing/unknown native abort status
  is UNVERIFIED; an explicit different FINISHED_* status is a failed cancellation.
  Cancelled cases retain every content/rank/READ/free check and local prompt-tail
  scheduling check, but need not show subsequent decode or a region crossing.
  No usage or output-token total is synthesized for a cancelled request.

PASS covers only the reported normal/cancelled-transfer checks. Unsupported pressure,
reset/reuse, full cancellation/abort conservation and actual cross-host transfer overlap remain
listed as unverified even when scope_pass is true. R1 must be judged separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


TP_RANKS = set(range(8))
COMPONENTS = {"raw_kv", "summary_mean"}
WORKER_EVENTS = {
    "source_content", "target_content", "read_submit", "read_network_done",
    "read_gpu_visible", "native_read_ready", "read_metadata", "tail_state",
    "target_compute", "source_notified", "state_registration",
}
SCHEDULER_EVENTS = {"source_hold", "source_free", "target_reserve", "target_free", "block_allocate"}
EVENTS = WORKER_EVENTS | SCHEDULER_EVENTS | {"frontend_select"}
AUXILIARY_EVENTS = {"source_compute", "pressure", "reuse", "reset", "abort", "free_deferred"}
METADATA_EVENTS = {"observer_started", "hook_installed", "provenance", "runner_created",
                   "runner_abi", "capture_elapsed"}
UNVERIFIED_CRITERIA = [
    "Open-tail accumulator reset/continuity and local output equivalence beyond scheduled token positions.",
    "Page reset/reuse with explicit component and alias ownership.",
    "Target-capacity pressure, resident/wire accounting, and full pool conservation.",
    "Actual concurrent transfer overlap and reuse races (an offered wave is not proof).",
    "Cancellation and abort resource/token/transfer conservation.",
    "Full ticket-08 nested G0-G2, reference equivalence, fidelity and overhead budgets.",
]


class Check:
    def __init__(self):
        self.failures = []
        self.missing = []
        self.details = {}

    def require(self, condition, message):
        if not condition:
            self.missing.append(message)

    def expect(self, condition, message):
        if not condition:
            self.failures.append(message)

    def result(self):
        return {"status": "FAIL" if self.failures else "UNVERIFIED" if self.missing else "PASS",
                "failures": self.failures, "missing": self.missing, **self.details}


def integer(value):
    return type(value) is int and value >= 0


def identity(row):
    return row.get("engine_id"), row.get("request_id")


def label(row):
    return f"{row.get('engine_id')}/{row.get('request_id')}/TP{row.get('tp_rank', 'scheduler')}"


def layer_keys(layers):
    """Read an independent complete layer census; never guess summary ownership."""
    if not isinstance(layers, list) or not layers:
        raise ValueError("expected_layers must be a nonempty complete census")
    keys = set()
    for row in layers:
        if (not isinstance(row, dict) or not integer(row.get("group_id"))
                or not isinstance(row.get("layer"), str) or not row["layer"]
                or not isinstance(row.get("components"), list) or not row["components"]
                or not set(row["components"]) <= COMPONENTS):
            raise ValueError("expected_layers entries require group_id, layer, explicit components")
        for component in row["components"]:
            key = row["group_id"], row["layer"], component
            if key in keys:
                raise ValueError(f"Duplicate registered layer/component: {key}")
            keys.add(key)
    if {key[2] for key in keys} != COMPONENTS:
        raise ValueError("The Step-5 census must include raw KV and summary-mean state")
    return keys


def registration_record(row, config, check):
    """Translate the collector's native expected inventories, not sampled rows."""
    raw, side = row.get("expected_raw_layers"), row.get("expected_summary_layers")
    observed = row.get("registrations")
    if not isinstance(raw, dict) or not isinstance(side, dict) or not isinstance(observed, list):
        raise ValueError("Native registration lacks independent raw/summary inventories")
    if (not all(isinstance(name, str) and integer(group) for name, group in raw.items())
            or not all(isinstance(name, str) and isinstance(main, str) and main in raw for name, main in side.items())):
        raise ValueError("Invalid native expected raw/summary layer mapping")
    observation = config.get("pd_state_observation", {})
    counts = (observation.get("expected_layer_count"), observation.get("expected_summary_layer_count"))
    if not all(integer(value) and value > 0 for value in counts):
        raise ValueError("Native registration needs frozen expected layer and summary counts")
    check.expect((len(raw), len(side)) == counts, f"{label(row)}: registration counts disagree with frozen model")
    summary_layers = set(side.values())
    layers = [{"group_id": group, "layer": name,
               "components": ["raw_kv", "summary_mean"] if name in summary_layers else ["raw_kv"]}
              for name, group in raw.items()]
    expected = layer_keys(layers)
    actual = {(r["group_id"], r["layer"], r["component"]) for r in observed}
    check.require(expected <= actual, f"{label(row)}: missing registered native layer/component")
    check.expect(actual <= expected, f"{label(row)}: unexpected native registration")
    return {**row, "event": "state_registration", "complete": (len(raw), len(side)) == counts,
            "expected_layers": layers}


def normalize(row, expected):
    """Resolve only explicitly emitted equivalent fields and frozen client joins."""
    row = dict(row)
    for native, canonical in (("blocks", "block_ids"), ("pool_free_blocks_before", "pool_free_before"),
                              ("pool_free_blocks_after", "pool_free_after"),
                              ("selected_client_request_id", "client_request_id"),
                              ("num_external_computed_tokens", "num_external_tokens"),
                              ("lb_engines", "load_snapshot")):
        if canonical not in row and native in row:
            row[canonical] = row[native]
    client = row.get("client_request_id")
    trace = expected.get(client) if isinstance(client, str) else None
    if trace:
        row.setdefault("wave", trace["wave"])
        if row.get("event") == "target_reserve" and "num_prompt_tokens" not in row:
            row["num_prompt_tokens"] = trace["num_prefill_tokens"]
            row["prompt_count_source"] = "frozen_client_trace"
    return row


def load(stage, instrumentation, config, expected, content_check):
    records = []
    native_errors = []
    paths = sorted((stage / "evidence" / "nodes").glob("*/pd_state/*.jsonl"))
    instrumentation.require(paths, "No pd_state JSONL evidence")
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeError) as error:
            instrumentation.missing.append(f"{path}: {error}")
            continue
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Record is not an object")
                if row.get("event") == "native_error":
                    native_errors.append({"location": where, **row})
                    continue
                if row.get("event") in ("observation_error", "observer_error", "observer_budget_exceeded", "release_unverified"):
                    raise ValueError(f"Observer error: {row.get('message', row.get('error', row.get('reason')))}")
                if row.get("event") in AUXILIARY_EVENTS | METADATA_EVENTS:
                    # These events do not close the unsupported criteria listed
                    # in UNVERIFIED_CRITERIA. Keep allocation snapshots separately.
                    continue
                row = normalize(row, expected)
                if row.get("event") == "registration":
                    row = registration_record(row, config, content_check)
                if row.get("event") not in EVENTS:
                    raise ValueError(f"Unknown event: {row.get('event')!r}")
                if (row.get("role") not in ("prefill", "decode")
                        or not isinstance(row.get("host"), str) or not row["host"]
                        or not all(integer(row.get(key)) for key in ("pid", "dp_rank", "monotonic_ns"))):
                    raise ValueError("Missing/invalid role, host, pid, dp_rank or monotonic_ns")
                event = row["event"]
                if event != "state_registration" and (not isinstance(row.get("request_id"), str) or not row["request_id"]):
                    raise ValueError("Missing native request_id")
                if event != "frontend_select" and (not isinstance(row.get("engine_id"), str) or not row["engine_id"]):
                    raise ValueError("Missing native engine_id")
                if event == "frontend_select" and row.get("engine_id") is not None and not isinstance(row["engine_id"], str):
                    raise ValueError("Invalid frontend engine_id")
                for key in ("remote_engine_id", "remote_request_id", "wave"):
                    if key in row and not isinstance(row[key], str):
                        raise ValueError(f"Invalid {key}")
                if event in WORKER_EVENTS and not integer(row.get("tp_rank")):
                    raise ValueError("Worker record is missing tp_rank")
                instrumentation.expect(row["host"] == path.parents[1].name,
                                       f"{where}: host disagrees with node directory")
                if event in WORKER_EVENTS:
                    instrumentation.expect(row["tp_rank"] in TP_RANKS,
                                           f"{where}: TP rank is outside 0..7")
                if event not in ("state_registration", "frontend_select", "block_allocate", "tail_state"):
                    expected_role = "prefill" if event.startswith("source_") else "decode"
                    instrumentation.expect(row["role"] == expected_role,
                                           f"{where}: event belongs to {expected_role}")
                records.append(row)
            except (ValueError, TypeError, KeyError) as error:
                instrumentation.missing.append(f"{where}: {error}")
    for path in sorted((stage / "evidence" / "nodes").glob("*/result.json")):
        try:
            result = json.loads(path.read_text())
            if not isinstance(result, dict):
                raise ValueError("Node result is not an object")
            instrumentation.require(not result.get("observation_errors"),
                                    f"{path}: observation_errors={result.get('observation_errors')}")
        except (OSError, ValueError) as error:
            instrumentation.missing.append(f"{path}: {error}")
    instrumentation.require(records, "No valid observation records")
    instrumentation.details.update(files=len(paths), records=len(records))
    return records, native_errors


def one(rows, event, check, name):
    selected = [row for row in rows if row["event"] == event]
    check.require(selected, f"{name}: missing {event}")
    check.expect(len(selected) <= 1, f"{name}: duplicate {event} / ambiguous ownership")
    return selected[0] if selected else None


def blocks(row, field, check):
    value = row.get(field)
    if (not isinstance(value, list) or not value
            or any(not isinstance(group, list) or not all(integer(v) for v in group) for group in value)):
        check.missing.append(f"{label(row)} {row['event']}: missing/invalid {field} group lists")
        return None
    if field != "block_ids":
        for group in value:
            physical = [v for v in group if v]
            check.expect(len(physical) == len(set(physical)), f"{label(row)}: duplicate IDs in {field}")
            if row["event"] == "read_submit":
                check.expect(len(physical) == len(group), f"{label(row)}: READ includes null physical block 0")
    return value


def ordered(before, after, check, reason):
    if before is None or after is None:
        return
    check.require(before["host"] == after["host"], f"{reason}: different hosts; clocks not compared")
    if before["host"] == after["host"]:
        check.expect(before["monotonic_ns"] <= after["monotonic_ns"], reason)


def content_rows(rows, check):
    result = {}
    for row in rows:
        if row["event"] not in ("source_content", "target_content"):
            continue
        name = label(row)
        ids, hashes = row.get("local_block_ids"), row.get("block_hashes")
        if (not integer(row.get("group_id")) or not isinstance(row.get("layer"), str)
                or not row["layer"] or not isinstance(row.get("component"), str) or row["component"] not in COMPONENTS
                or not isinstance(ids, list) or not all(integer(v) for v in ids)
                or not isinstance(hashes, list) or len(ids) != len(hashes)
                or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v) for v in hashes)
                or not integer(row.get("block_bytes")) or row["block_bytes"] == 0):
            check.missing.append(f"{name}: incomplete physical-row content record")
            continue
        check.expect(len(ids) == len(set(ids)), f"{name}: duplicate content page IDs")
        key = row["tp_rank"], row["group_id"], row["layer"], row["component"]
        check.expect(key not in result, f"{name}: duplicate content snapshot {key}")
        result[key] = row
    return result


def expected_keys(config, registrations, engine, rank, check):
    frozen = config.get("pd_state_observation", {}).get("expected_layers")
    observed = [row for row in registrations if row.get("engine_id") == engine and row["tp_rank"] == rank]
    keys = None
    try:
        if frozen is not None:
            keys = layer_keys(frozen)
        elif not observed:
            check.missing.append(f"{engine}/TP{rank}: no frozen or complete registered layer census")
        for row in observed:
            if row.get("complete") is not True:
                check.missing.append(f"{engine}/TP{rank}: registration census is not complete")
                continue
            census = layer_keys(row.get("expected_layers"))
            if keys is None:
                keys = census
            else:
                check.expect(keys == census, f"{engine}/TP{rank}: registration disagrees with layer census")
    except (ValueError, TypeError) as error:
        check.missing.append(f"{engine}/TP{rank}: {error}")
    return keys


def check_content(source, target, config, registrations, check):
    p_content, d_content = content_rows(source, check), content_rows(target, check)
    p_engine = source[0]["engine_id"] if source else None
    d_engine = target[0]["engine_id"]
    compared = 0
    differing_ids = False
    for rank in sorted(TP_RANKS):
        p_keys = expected_keys(config, registrations, p_engine, rank, check)
        d_keys = expected_keys(config, registrations, d_engine, rank, check)
        if p_keys is not None and d_keys is not None:
            check.expect(p_keys == d_keys, f"{d_engine}/TP{rank}: P/D registered state differs")
        expected = (p_keys or set()) | (d_keys or set())
        for side, content in (("P", p_content), ("D", d_content)):
            actual = {key[1:] for key in content if key[0] == rank}
            check.require(expected <= actual, f"{d_engine}/TP{rank}: missing {side} content {sorted(expected - actual)}")
            if expected:
                check.expect(actual <= expected, f"{d_engine}/TP{rank}: content outside registered layer census")
        read = one([r for r in target if r.get("tp_rank") == rank], "read_submit", check,
                   f"{d_engine}/TP{rank}")
        if read is not None and "remote_tp_rank" in read:
            check.expect(type(read["remote_tp_rank"]) is int and read["remote_tp_rank"] == rank,
                         f"{d_engine}/TP{rank}: actual READ uses the wrong source TP rank")
        local = blocks(read, "local_block_ids", check) if read else None
        remote = blocks(read, "remote_block_ids", check) if read else None
        if local is not None and remote is not None:
            check.expect(len(local) == len(remote) and all(len(a) == len(b) for a, b in zip(local, remote)),
                         f"{d_engine}/TP{rank}: submitted source/target group lengths differ")
        for key, d_row in d_content.items():
            if key[0] != rank:
                continue
            p_row = p_content.get(key)
            wire = d_row.get("remote_block_ids")
            if "remote_tp_rank" in d_row:
                check.expect(type(d_row["remote_tp_rank"]) is int and d_row["remote_tp_rank"] == rank,
                             f"{label(d_row)}: content uses the wrong source TP rank")
            check.require(all(isinstance(d_row.get(field), str) and d_row[field]
                              for field in ("remote_engine_id", "remote_request_id")),
                          f"{label(d_row)}: content lacks native source identity")
            if not isinstance(wire, list) or not all(integer(v) for v in wire) or len(wire) != len(d_row["local_block_ids"]):
                check.missing.append(f"{label(d_row)}: incomplete remote physical-row mapping")
                continue
            check.expect(len(wire) == len(set(wire)), f"{label(d_row)}: duplicate remote page IDs")
            group = key[1]
            if local is not None and remote is not None:
                in_range = group < len(local) and group < len(remote)
                check.expect(in_range, f"{label(d_row)}: content group absent from actual READ")
                if in_range:
                    check.expect((d_row["local_block_ids"], wire) == (local[group], remote[group]),
                                 f"{label(d_row)}: target content disagrees with actual READ mapping")
            if p_row is None:
                check.missing.append(f"{label(d_row)}: no source content for {key[1:]}")
                continue
            p_hashes = dict(zip(p_row["local_block_ids"], p_row["block_hashes"]))
            for source_id, target_id, digest in zip(wire, d_row["local_block_ids"], d_row["block_hashes"]):
                check.require(source_id in p_hashes, f"{label(d_row)}: source row {source_id} not hashed")
                if source_id in p_hashes:
                    check.expect(p_hashes[source_id].lower() == digest.lower(),
                                 f"{label(d_row)} {key[1:]}: content mismatch P{source_id}->D{target_id}")
                    compared += 1
                    differing_ids |= source_id != target_id
            check.expect(p_row["block_bytes"] == d_row["block_bytes"], f"{label(d_row)}: physical row bytes differ")
    check.require(compared > 0, f"{d_engine}: no complete physical rows compared")
    return {"compared_rows": compared, "different_local_page_ids": differing_ids}


def registered_groups(config, registrations, engine, rank, check):
    """Read explicit native group ownership and semantics, not descriptor shapes."""
    observation = config.get("pd_state_observation", {})
    frozen = observation.get("group_specs")
    observed = [r for r in registrations if r.get("engine_id") == engine and r["tp_rank"] == rank
                and r.get("group_specs") is not None]
    candidates = observed or [{"group_specs": frozen, "transfer_group_ids": observation.get("transfer_group_ids")}]
    result = None
    for row in candidates:
        specs, mapping = row.get("group_specs"), row.get("transfer_group_ids")
        if not isinstance(specs, list) or not specs:
            check.missing.append(f"{engine}/TP{rank}: no independent group specs/native group mapping")
            continue
        current = {}
        for spec in specs:
            if not isinstance(spec, dict) or not integer(spec.get("group_id")):
                check.missing.append(f"{engine}/TP{rank}: invalid registered transfer group")
                continue
            group = spec["group_id"]
            native = spec.get("native_group_id")
            if native is None and isinstance(mapping, list) and group < len(mapping):
                native = mapping[group]
            size, window = spec.get("block_size"), spec.get("sliding_window")
            if (not integer(native) or not integer(size) or size == 0 or "sliding_window" not in spec
                    or (window is not None and (not integer(window) or window == 0))):
                check.missing.append(f"{engine}/TP{rank}/group{group}: missing exact native group/block/window semantics")
                continue
            check.expect(group not in current, f"{engine}/TP{rank}: duplicate registered transfer group {group}")
            current[group] = (native, size, window)
        if len(current) != len(specs):
            continue
        if result is None:
            result = current
        else:
            check.expect(current == result, f"{engine}/TP{rank}: contradictory native group registrations")
    return result or {}


def token_attribution(source, target, reserve, config, check, *, frozen_prompt=None):
    hold = next((r for r in source if r["event"] == "source_hold"), None)
    computed = hold.get("num_computed_tokens") if hold else None
    prompt = reserve.get("num_prompt_tokens") if reserve else None
    external = reserve.get("num_external_tokens") if reserve else None
    size = config.get("engine", {}).get("block_size")
    name = label(reserve or (target[0] if target else {}))
    for value, description in ((computed, "actual P computed prompt"), (prompt, "frozen/observed prompt"),
                               (external, "D externally loaded prefix")):
        check.require(integer(value), f"{name}: missing {description} token attribution")
    check.require(integer(size) and size > 0, f"{name}: no frozen native block size for matched-token rule")
    if integer(computed) and integer(external):
        check.expect(computed >= external, f"{name}: D loaded more tokens than P actually computed")
        if integer(size) and size > 0 and computed >= size:
            expected = size * ((computed - size) // size)
            check.expect(external == expected, f"{name}: loaded prefix {external} != native backoff {expected}")
    if integer(computed) and integer(prompt):
        check.expect(computed == prompt, f"{name}: actual P computed prompt differs from frozen/observed prompt")
    if integer(frozen_prompt):
        if integer(computed):
            check.expect(computed == frozen_prompt, f"{name}: actual P computation differs from independent frozen prompt")
        if integer(prompt):
            check.expect(prompt == frozen_prompt, f"{name}: target prompt differs from independent frozen prompt")
    if integer(prompt) and integer(external):
        check.expect(0 < external < prompt, f"{name}: no positive locally computed prompt tail")
        if integer(size) and size > 0:
            check.expect(size <= prompt - external < 2 * size, f"{name}: prompt tail is outside native block backoff")
    if hold and "num_matched_tokens" in hold:
        matched = hold["num_matched_tokens"]
        check.require(integer(matched), f"{name}: invalid explicit source num_matched_tokens")
        if integer(matched) and integer(external):
            check.expect(matched == external, f"{name}: explicit source matched tokens differ from D loaded prefix")
    params = hold.get("kv_xfer_params") if hold else None
    check.require(isinstance(params, dict), f"{name}: missing independent source transfer metadata")
    if isinstance(params, dict):
        advertised = params.get("remote_num_tokens")
        check.require(integer(advertised), f"{name}: source metadata lacks advertised remote_num_tokens")
        if integer(advertised) and integer(computed):
            check.expect(advertised == computed, f"{name}: source advertised tokens differ from actual P computation")
        for field, actual in (("remote_engine_id", hold["engine_id"]), ("remote_request_id", hold["request_id"])):
            if field in params:
                check.expect(params[field] == actual, f"{name}: source advertised wrong {field}")
    for rank in TP_RANKS:
        metadata = one([r for r in target if r.get("tp_rank") == rank], "read_metadata", check, f"{name}/TP{rank}")
        if metadata:
            remote = metadata.get("remote_num_tokens")
            check.require(integer(remote), f"{name}/TP{rank}: no received remote_num_tokens")
            if integer(remote) and integer(computed):
                check.expect(remote == computed, f"{name}/TP{rank}: received remote tokens differ from actual P computation")
    return {"source_computed_tokens": computed, "prompt_tokens": prompt, "external_tokens": external,
            "frozen_prompt_tokens": frozen_prompt,
            "prompt_count_source": reserve.get("prompt_count_source", "observed_target_reservation") if reserve else None,
            "source_transfer_metadata": params}


def logical_indexes(ids, table, check, name):
    """Resolve physical IDs against a full scheduler table, retaining HMA null slots."""
    owners = defaultdict(list)
    for index, block in enumerate(table):
        if block:
            owners[block].append(index)
    indexes = []
    for block in ids:
        locations = owners.get(block, [])
        check.expect(block != 0 and len(locations) == 1,
                     f"{name}: page {block} has no unique non-null scheduler logical owner")
        if block and len(locations) == 1:
            indexes.append(locations[0])
    check.expect(all(a < b for a, b in zip(indexes, indexes[1:])),
                 f"{name}: duplicate, reversed or out-of-order logical pages")
    return indexes


def check_mapping(source, target, reserve, config, registrations, tokens, check):
    hold = next((r for r in source if r["event"] == "source_hold"), None)
    source_tables = blocks(hold, "block_ids", check) if hold else None
    target_tables = blocks(reserve, "block_ids", check) if reserve else None
    check.require(source_tables is not None and target_tables is not None, "No independent scheduler tables for READ mapping")
    params = tokens["source_transfer_metadata"]
    advertised = None
    if isinstance(params, dict):
        advertised = blocks({**hold, "remote_block_ids": params.get("remote_block_ids")}, "remote_block_ids", check)
    else:
        check.missing.append("No independently advertised source block tables")
    computed, external = tokens["source_computed_tokens"], tokens["external_tokens"]
    for rank in sorted(TP_RANKS):
        name = f"{label(target[0])}/TP{rank}"
        p_groups = registered_groups(config, registrations, hold["engine_id"] if hold else None, rank, check)
        d_groups = registered_groups(config, registrations, target[0]["engine_id"], rank, check)
        rows = [r for r in target if r.get("tp_rank") == rank]
        read = one(rows, "read_submit", check, name)
        metadata = one(rows, "read_metadata", check, name)
        if not read:
            continue
        if "remote_tp_rank" in read:
            check.expect(type(read["remote_tp_rank"]) is int and read["remote_tp_rank"] == rank,
                         f"{name}: READ targets wrong source TP rank")
        local, remote = blocks(read, "local_block_ids", check), blocks(read, "remote_block_ids", check)
        if local is None or remote is None:
            continue
        check.expect(len(local) == len(remote), f"{name}: READ source/target group counts differ")
        if p_groups and d_groups:
            check.expect(set(p_groups) == set(d_groups) == set(range(len(remote))),
                         f"{name}: READ omits or adds independently registered transfer groups")
        if advertised is not None:
            check.expect(len(advertised) == len(remote), f"{name}: source advertised group count differs from READ")
        meta = {}
        if metadata:
            ordered(metadata, read, check, f"{name}: READ preceded metadata reception")
            for field in ("advertised_remote_block_ids", "reserved_local_block_ids", "local_physical_block_ids"):
                meta[field] = blocks(metadata, field, check)
                if meta[field] is not None:
                    check.expect(len(meta[field]) == len(remote), f"{name}: {field} group count differs from READ")
            received = meta.get("advertised_remote_block_ids")
            if advertised is not None and received is not None:
                check.expect(advertised == received, f"{name}: independently advertised and received source block IDs differ")
        for group, (p_wire, d_wire) in enumerate(zip(remote, local)):
            check.expect(len(p_wire) == len(d_wire), f"{name}/group{group}: paired descriptor lengths differ")
            p_spec, d_spec = p_groups.get(group), d_groups.get(group)
            check.require(p_spec is not None and d_spec is not None, f"{name}/group{group}: no explicit scheduler/transfer group map")
            if p_spec is None or d_spec is None or source_tables is None or target_tables is None:
                continue
            p_native, size, window = p_spec
            d_native, d_size, d_window = d_spec
            check.expect((size, window) == (d_size, d_window), f"{name}/group{group}: P/D group semantics differ")
            valid = p_native < len(source_tables) and d_native < len(target_tables)
            check.expect(valid, f"{name}/group{group}: mapped scheduler group is absent")
            if not valid:
                continue
            p_table, d_table = source_tables[p_native], target_tables[d_native]
            p_indexes = logical_indexes(p_wire, p_table, check, f"{name}/group{group}/P READ")
            d_indexes = logical_indexes(d_wire, d_table, check, f"{name}/group{group}/D READ")
            check.expect(p_indexes == d_indexes and len(p_indexes) == len(p_wire) == len(d_indexes),
                         f"{name}/group{group}: paired physical rows represent different logical pages")
            if advertised is not None and group < len(advertised):
                logical_indexes([v for v in advertised[group] if v], p_table, check, f"{name}/group{group}/P advertised")
                check.expect(set(p_wire) <= set(advertised[group]), f"{name}/group{group}: READ outside independently advertised source pages")
            for field in ("reserved_local_block_ids", "local_physical_block_ids"):
                tables = meta.get(field)
                if tables is not None and group < len(tables):
                    logical_indexes([v for v in tables[group] if v], d_table, check, f"{name}/group{group}/{field}")
                    check.expect(set(d_wire) <= set(tables[group]), f"{name}/group{group}: READ outside independently received {field}")
            if not integer(external):
                check.missing.append(f"{name}/group{group}: no external token range for exact READ coverage")
                continue
            check.require(external % size == 0, f"{name}/group{group}: unsupported partial physical-page external range")
            if external % size:
                continue
            end = external // size
            check.expect(all(index < end for index in p_indexes + d_indexes),
                         f"{name}/group{group}: READ at or beyond externally loaded full-page range")
            if window is not None and not integer(computed):
                check.missing.append(f"{name}/group{group}: no actual P token boundary for current SWA coverage")
                continue
            # The first locally recomputed query is at external, not at P's
            # later prompt endpoint. Its oldest SWA key must also be available.
            start = max(0, (external - window + 1) // size) if window is not None else 0
            required = list(range(start, end))
            check.expect(p_indexes == d_indexes == required,
                         f"{name}/group{group}: READ does not cover required full-prefix/current-SWA logical pages {required}")


def check_native_readiness(source, target, check, *, no_fence):
    local = Check()
    for rank in sorted(TP_RANKS):
        rows = [r for r in target if r.get("tp_rank") == rank]
        name = f"{label(target[0])}/TP{rank}"
        ready = one(rows, "native_read_ready", local, name)
        network = one(rows, "read_network_done", local, name)
        compute = sorted((r for r in rows if r["event"] == "target_compute"), key=lambda r: r["monotonic_ns"])
        local.require(compute, f"{name}: no first target forward for native readiness")
        first = compute[0] if compute else None
        if ready:
            local.require(type(ready.get("observer_fence")) is bool, f"{name}: native readiness lacks explicit observer_fence flag")
            local.expect(ready.get("observer_fence") is not True, f"{name}: observer fence is not a native readiness witness")
            boundary = ready.get("boundary")
            local.require(boundary == "native_get_transfer_results_finished_recving",
                          f"{name}: no explicit native finished_recving completion boundary")
        ordered(network, ready, local, f"{name}: native completion preceded network completion")
        ordered(ready, first, local, f"{name}: target computed before native finished_recving readiness")
        visible = next((r for r in rows if r["event"] == "read_gpu_visible"), None)
        ordered(ready, visible, local, f"{name}: native readiness was observed only after diagnostic GPU fence")
        if no_fence:
            for row in rows:
                diagnostic = row["event"] in ("source_content", "target_content", "tail_state") or row.get("observer_fence") is True
                if first and row["host"] == first["host"] and row["monotonic_ns"] <= first["monotonic_ns"]:
                    local.expect(not diagnostic, f"{name}: diagnostic fence/CPU readback before first target forward")
                    if row["event"] == "read_gpu_visible":
                        local.require(row.get("observer_fence") is False, f"{name}: pre-forward GPU event lacks no-fence proof")
            p = [r for r in source if r.get("tp_rank") == rank]
            local.expect(not any(r["event"] in ("source_content", "tail_state") or r.get("observer_fence") is True for r in p),
                         f"{name}: readiness-only source performed unselected diagnostic readback/fence")
    outcome = local.result()
    check.failures.extend(outcome["failures"])
    check.missing.extend(outcome["missing"])
    return outcome["status"] == "PASS"


def check_free(held, freed, check, *, prefix_caching_disabled):
    if freed is None:
        return
    if freed['event'] == 'target_free':
        defer = freed.get('defer_block_free')
        check.require(type(defer) is bool, f"{label(freed)}: missing native release guard")
        if defer is True:
            last, processed = freed.get('last_sched_seq'), freed.get('processed_step_seq')
            check.require(integer(last) and integer(processed),
                          f"{label(freed)}: missing native GPU-step completion fence")
            if integer(last) and integer(processed):
                check.expect(last <= processed, f"{label(freed)}: native last GPU step was not processed before free")
            if 'deferred_fence_seq' in freed:
                fence = freed['deferred_fence_seq']
                check.require(integer(fence), f"{label(freed)}: invalid native deferred release fence")
                if integer(fence) and integer(processed):
                    check.expect(fence <= processed, f"{label(freed)}: native deferred free preceded its completion fence")
    allocated = blocks(held, "block_ids", check) if held else None
    released = blocks(freed, "block_ids", check)
    check.require(allocated is not None, f"{label(freed)}: no ownership snapshot for release accounting")
    if allocated is not None and released is not None:
        check.expect([set(g) - {0} for g in allocated] == [set(g) - {0} for g in released],
                     f"{label(freed)}: freed pages differ from held/reserved pages")
    before, after = freed.get("pool_free_before"), freed.get("pool_free_after")
    check.require(integer(before) and integer(after), f"{label(freed)}: missing pool free counts")
    exact_scope = freed.get("counts_scope") == "request_release"
    check.require(exact_scope, f"{label(freed)}: pool counts are not attributed to this request release")
    check.require(prefix_caching_disabled, f"{label(freed)}: exact pool accounting requires known disabled prefix caching")
    if (allocated is not None and released is not None and integer(before) and integer(after)
            and exact_scope and prefix_caching_disabled):
        expected = len({block for group in released for block in group} - {0})
        delta = after - before
        check.expect(delta == expected,
                     f"{label(freed)}: pool increase {delta} != {expected} unique non-null released blocks (leak or over-release)")
        count = freed.get("released_block_count")
        if count is not None:
            check.require(integer(count), f"{label(freed)}: invalid released_block_count")
            if integer(count):
                check.expect(count == delta == expected, f"{label(freed)}: release count/pool delta/owned pages disagree")


def check_lifetime(source, target, check, *, prefix_caching_disabled, content_selected=True, explicit_mask=False):
    name = label(target[0])
    hold = one(source, "source_hold", check, name)
    p_free = one(source, "source_free", check, name)
    reserve = one(target, "target_reserve", check, name)
    d_free = one(target, "target_free", check, name)
    check_free(hold, p_free, check, prefix_caching_disabled=prefix_caching_disabled)
    allocations = [r for r in target if r["event"] == "block_allocate"]
    if d_free:
        allocations = [r for r in allocations if r["host"] == d_free["host"]
                       and r["monotonic_ns"] <= d_free["monotonic_ns"]]
    final_allocation = max(allocations, key=lambda r: r["monotonic_ns"]) if allocations else reserve
    check_free(final_allocation, d_free, check, prefix_caching_disabled=prefix_caching_disabled)
    for rank in sorted(TP_RANKS):
        p = [row for row in source if row.get("tp_rank") == rank]
        d = [row for row in target if row.get("tp_rank") == rank]
        notice = one(p, "source_notified", check, f"{name}/TP{rank}")
        submit = one(d, "read_submit", check, f"{name}/TP{rank}")
        network = one(d, "read_network_done", check, f"{name}/TP{rank}")
        visible = one(d, "read_gpu_visible", check, f"{name}/TP{rank}") if content_selected else None
        if visible and explicit_mask:
            check.require(type(visible.get("observer_fence")) is bool, f"{name}/TP{rank}: diagnostic GPU visibility lacks fence label")
            check.expect(visible.get("observer_fence") is not False, f"{name}/TP{rank}: content GPU visibility falsely claims no observer fence")
        compute = sorted((r for r in d if r["event"] == "target_compute"), key=lambda r: r["monotonic_ns"])
        check.require(compute, f"{name}/TP{rank}: missing target_compute")
        for a, b, reason in (
            (hold, notice, "source notification preceded hold"),
            (notice, p_free, "source pages freed before all TP notifications"),
            (reserve, submit, "READ preceded target reservation"),
            (submit, network, "network completion preceded READ"),
            (network, visible, "GPU readiness preceded network completion"),
            (visible, compute[0] if compute else None, "target computed before GPU-visible readiness"),
            (compute[-1] if compute else None, d_free, "target pages freed before compute finished"),
            (visible, d_free, "target pages freed before GPU-visible readiness"),
        ):
            ordered(a, b, check, f"{name}/TP{rank}: {reason}")
        for row in p:
            if row["event"] == "source_content":
                ordered(row, p_free, check, f"{name}/TP{rank}: source content sampled after free")
        for row in d:
            if row["event"] == "target_content" and content_selected:
                ordered(visible, row, check, f"{name}/TP{rank}: target content sampled before GPU readiness")
                ordered(row, compute[0] if compute else None, check,
                        f"{name}/TP{rank}: transfer snapshot sampled after target compute")
    return reserve


def positions(row, check):
    start, count = row.get("num_computed_tokens"), row.get("num_scheduled_tokens")
    values = row.get("positions")
    if values is None:
        if not integer(start) or not integer(count) or count == 0:
            check.missing.append(f"{label(row)}: compute has no positions or valid scheduled interval")
            return None
        return start, start + count
    if not isinstance(values, list) or not values or not all(integer(v) for v in values):
        check.missing.append(f"{label(row)}: invalid compute positions")
        return None
    first, end = values[0], values[-1] + 1
    check.expect(all(b == a + 1 for a, b in zip(values, values[1:])), f"{label(row)}: noncontiguous compute positions")
    if start is not None or count is not None:
        check.expect(integer(start) and integer(count) and (start, count) == (first, len(values)),
                     f"{label(row)}: compute positions/counts disagree")
    return first, end


def check_tail(source, target, reserve, region_size, check, *, cancellation_expected=False):
    if reserve is None:
        check.missing.append("No target reservation for tail attribution")
        return
    prompt, external = reserve.get("num_prompt_tokens"), reserve.get("num_external_tokens")
    if not integer(prompt) or not integer(external):
        check.missing.append(f"{label(reserve)}: missing prompt/external token attribution")
        return
    check.expect(0 < external < prompt, f"{label(reserve)}: no positive locally computed prompt tail")
    for rank in sorted(TP_RANKS):
        compute = sorted((r for r in target if r.get("tp_rank") == rank and r["event"] == "target_compute"),
                         key=lambda r: r["monotonic_ns"])
        end, crossing = external, False
        check.require(compute, f"{label(reserve)}/TP{rank}: no tail compute")
        for row in compute:
            interval = positions(row, check)
            if interval is None:
                continue
            start, next_end = interval
            check.require(start <= end, f"{label(row)}: missing tail/decode scheduling interval")
            check.expect(start >= end, f"{label(row)}: duplicate or already externally loaded compute position")
            if integer(region_size) and region_size > 0:
                boundary = ((max(start, prompt) + region_size - 1) // region_size) * region_size
                crossing |= boundary < next_end and boundary > external
            end = next_end
        check.require(end >= prompt, f"{label(reserve)}/TP{rank}: locally computed tail is incomplete")
        if not cancellation_expected:
            check.require(end > prompt, f"{label(reserve)}/TP{rank}: no subsequent decode compute")
            if integer(region_size) and region_size > 0:
                check.require(crossing, f"{label(reserve)}/TP{rank}: no observed decode crossing a region")


def trace_requests(stage, check):
    expected = {}
    paths = sorted((stage / "trace").glob("*/request_ids.json"))
    check.require(paths, "Missing independent trace request census")
    for path in paths:
        try:
            for row in json.loads(path.read_text())["rows"]:
                if (not isinstance(row, dict) or not isinstance(row.get("request_id"), str) or not row["request_id"]
                        or not integer(row.get("num_prefill_tokens")) or row["num_prefill_tokens"] == 0):
                    raise ValueError("Trace rows require client request_id and num_prefill_tokens")
                rid = row["request_id"]
                if "cancel_after_first_content" in row and type(row["cancel_after_first_content"]) is not bool:
                    raise ValueError("cancel_after_first_content must be a boolean")
                check.expect(rid not in expected, f"Duplicate trace request {rid}")
                expected[rid] = {**row, "wave": path.parent.name}
        except (OSError, ValueError, TypeError, KeyError) as error:
            check.missing.append(f"{path}: {error}")
    return expected


def client_records(stage, check):
    """Load captured client facts without manufacturing final usage on cancellation."""
    records = defaultdict(list)
    paths = sorted(path for root in ("waves", "replay")
                   for path in (stage / "evidence" / root).glob("*/client_requests.jsonl"))
    for path in paths:
        try:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("request_id"), str) or not row["request_id"]:
                    raise ValueError(f"line {number}: missing client request_id")
                records[path.parent.name, row["request_id"]].append(row)
        except (OSError, ValueError, TypeError) as error:
            check.missing.append(f"{path}: {error}")
    return records


def request_disposition(target, reserve, expected, clients, check):
    local = Check()
    client = reserve.get("client_request_id") if reserve else None
    trace = expected.get(client) if isinstance(client, str) else None
    local.require(trace is not None, f"{client}: no frozen client request for outcome attribution")
    planned = bool(trace and trace.get("cancel_after_first_content") is True)
    captured = clients.get((trace["wave"], client), []) if trace else []
    local.expect(len(captured) <= 1, f"{client}: duplicate captured client outcomes")
    observed = captured[0] if captured else {}
    terminal = []
    for row in target:
        if row["event"] == "target_free":
            for field in ("request_status", "finish_reason", "status"):
                if field in row:
                    terminal.append({"field": field, "value": row[field]})
    labels = {r["value"].removeprefix("RequestStatus.") for r in terminal if isinstance(r["value"], str)}
    aborted = "FINISHED_ABORTED" in labels
    other_finished = {value for value in labels if value.startswith("FINISHED_") and value != "FINISHED_ABORTED"}
    observed_cancel = observed.get("client_cancelled") is True
    if planned or observed_cancel or aborted:
        local.expect(planned, f"{client}: unplanned native/client cancellation is not normal completion")
        local.require(captured, f"{client}: no captured client cancellation")
        flag = observed.get("client_cancelled")
        local.require(type(flag) is bool, f"{client}: missing/invalid client_cancelled flag")
        if type(flag) is bool:
            local.expect(flag, f"{client}: planned cancellation did not occur at client")
        local.expect(observed.get("stream_completed") is not True,
                     f"{client}: cancelled client record also claims normal stream completion")
        local.require(aborted or other_finished, f"{client}: missing explicit native FINISHED_ABORTED status")
        local.expect(not other_finished, f"{client}: explicit native non-abort terminal status {sorted(other_finished)}")
    outcome = local.result()
    check.failures.extend(outcome["failures"])
    check.missing.extend(outcome["missing"])
    return {**outcome, "expected_outcome": "cancelled" if planned else "normal",
            "observed_outcome": ("cancelled" if planned and outcome["status"] == "PASS" else "UNVERIFIED")
                                if planned or observed_cancel or aborted else
                                "normal_compute" if any(r["event"] == "target_compute" for r in target) else "UNVERIFIED",
            "client_cancelled": observed.get("client_cancelled"), "native_terminal_statuses": terminal}


def frozen_masks(observation, expected, check):
    explicit = any(field in observation for field in ("content_client_request_ids", "readiness_only_client_request_ids"))
    if not explicit:
        return set(expected), set(), False
    masks = {}
    for field in ("content_client_request_ids", "readiness_only_client_request_ids", "selected_client_request_ids"):
        values = observation.get(field)
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            check.missing.append(f"Missing/invalid frozen {field}")
            values = []
        check.expect(len(values) == len(set(values)), f"Duplicate IDs in frozen {field}")
        masks[field] = set(values)
    content, readiness, selected = (masks[k] for k in ("content_client_request_ids", "readiness_only_client_request_ids", "selected_client_request_ids"))
    check.expect(not content & readiness, "Frozen content and readiness-only selections overlap")
    check.require(content | readiness == selected == set(expected),
                  "Frozen content/readiness selections do not partition all selected trace requests")
    return content, readiness, True


def readiness_coverage(expected, readiness, qualified, wave_placements, check):
    waves = defaultdict(set)
    for client, trace in expected.items():
        waves[trace["wave"]].add(client)
    sequential = {next(iter(ids)) for ids in waves.values() if len(ids) == 1 and ids <= readiness}
    check.require(len(sequential) >= 2, "Missing both frozen sequential no-fence readiness probes")
    check.require(sequential <= qualified, "Incomplete native no-fence evidence for a sequential readiness probe")
    check.require(readiness <= qualified, "Incomplete eight-rank native no-fence evidence for readiness-only requests")
    concurrent = []
    for wave, ids in waves.items():
        if len(ids) < 2 or not ids <= readiness or not ids <= qualified:
            continue
        placement = wave_placements.get(wave, {})
        endpoints = placement.get("decode_endpoints", [])
        if ({0, 1} <= set(placement.get("prefill", [])) and {0, 1} <= set(placement.get("decode", []))
                and len(endpoints) >= 2 and len({endpoint[0] for endpoint in endpoints}) >= 2):
            concurrent.append(wave)
    check.require(concurrent, "No native no-fence multi-request wave covers both P/D lanes and D endpoints")
    check.details.update(sequential_client_request_ids=sorted(sequential), native_concurrent_waves=sorted(concurrent),
                          qualified_readiness_only_client_request_ids=sorted(qualified))


def release_timing(source, target):
    """Report local-clock intervals without inventing a cross-host time axis."""
    def event(rows, name, latest=False):
        matches = [row for row in rows if row['event'] == name]
        if not matches or len({row['host'] for row in matches}) != 1:
            return None
        return (max if latest else min)(matches, key=lambda row: row['monotonic_ns'])

    def interval(before, after):
        if before is None or after is None or before['host'] != after['host']:
            return None
        return dict(host=before['host'], clock='CLOCK_MONOTONIC',
                    start_event=before['event'], end_event=after['event'],
                    start_ns=before['monotonic_ns'], end_ns=after['monotonic_ns'],
                    elapsed_ns=after['monotonic_ns']-before['monotonic_ns'])

    p_free, d_free = event(source, 'source_free'), event(target, 'target_free')
    return {
        'source_hold_to_free': interval(event(source, 'source_hold'), p_free),
        'last_source_notification_to_free': interval(event(source, 'source_notified', True), p_free),
        'target_reservation_to_free': interval(event(target, 'target_reserve'), d_free),
        'last_native_receive_completion_to_target_free': interval(event(target, 'native_read_ready', True), d_free),
        'last_compute_entry_to_target_free': interval(event(target, 'target_compute', True), d_free),
        'target_native_release_guard': {key: d_free[key] for key in
            ('last_sched_seq', 'processed_step_seq', 'defer_block_free', 'deferred_fence_seq', 'boundary')
            if d_free is not None and key in d_free},
    }


def placement(records, transfers, reserves, expected, check):
    selections = [row for row in records if row["event"] == "frontend_select"]
    waves = defaultdict(lambda: {"prefill": set(), "decode": set(), "decode_endpoints": set()})
    clients = defaultdict(list)
    for key, reserve in reserves.items():
        if reserve is None:
            continue
        client = reserve.get("client_request_id")
        if not isinstance(client, str):
            check.missing.append(f"{label(reserve)}: missing client/native request join")
            continue
        clients[client].append(key)
        trace = expected.get(client)
        check.require(trace is not None, f"{client}: no independent trace request")
        if trace:
            check.expect(reserve.get("wave") == trace["wave"], f"{client}: observed/trace wave mismatch")
            if integer(reserve.get("num_prompt_tokens")):
                check.expect(reserve["num_prompt_tokens"] == trace.get("num_prefill_tokens"),
                             f"{client}: observed/trace prompt count mismatch")
        for role, native in (("prefill", transfers[key]), ("decode", key)):
            selected = [r for r in selections if r["role"] == role and r.get("client_request_id") == client]
            check.require(selected, f"{client}: missing {role} frontend selection/load snapshot")
            for row in selected:
                check.expect(row["request_id"] == native[1] and (not row.get("engine_id") or row["engine_id"] == native[0]),
                             f"{client}: frontend/native {role} request mismatch")
                check.expect(row.get("wave") == reserve.get("wave"), f"{client}: frontend wave mismatch")
                check.require(isinstance(row.get("load_snapshot"), (list, dict)) and row["load_snapshot"],
                              f"{client}: missing {role} native load snapshot")
                owners = [r for r in records if identity(r) == native and r["event"] == ("source_hold" if role == "prefill" else "target_reserve")]
                if owners:
                    check.expect(all(r["dp_rank"] == row["dp_rank"] for r in owners), f"{client}: frontend/owner {role} DP mismatch")
                waves[reserve.get("wave")][role].add(row["dp_rank"])
        waves[reserve.get("wave")]["decode_endpoints"].add((reserve["host"], reserve["engine_id"]))
    for client in expected:
        check.require(client in clients, f"{client}: no observed native transfer")
    for client, keys in clients.items():
        check.expect(len(keys) == 1, f"{client}: multiple native transfer owners")
    check.require(any({0, 1} <= w["prefill"] and {0, 1} <= w["decode"] and len(w["decode_endpoints"]) >= 2
                      and len({endpoint[0] for endpoint in w["decode_endpoints"]}) >= 2
                      and sum(r["wave"] == wave for r in expected.values()) >= 2 for wave, w in waves.items()),
                  "No common multi-request wave covers both P/D lanes 0,1 and both D endpoints")
    check.details["waves"] = {wave: {name: sorted(values) for name, values in row.items()} for wave, row in waves.items()}


def judge(stage: Path, platform_status: str) -> dict:
    """Read local evidence only. Incomplete/corrupt instrumentation cannot pass."""
    stage = Path(stage)
    names = ("instrumentation", "content", "mapping", "native_readiness", "lifetime", "tail_and_decode", "cancellation", "boundaries", "placement_and_requests", "runtime")
    checks = {name: Check() for name in names}
    try:
        config = json.loads((stage / "config.json").read_text())
        if not isinstance(config, dict):
            raise ValueError("config.json is not an object")
    except (OSError, ValueError) as error:
        checks["instrumentation"].missing.append(str(error))
        config = {}
    for field in ("pd_state_observation", "engine"):
        if field in config and not isinstance(config[field], dict):
            checks["instrumentation"].missing.append(f"config.{field} is not an object")
            config[field] = {}
    observation = config.get("pd_state_observation", {})
    expected = trace_requests(stage, checks["placement_and_requests"])
    content_ids, readiness_ids, explicit_mask = frozen_masks(observation, expected, checks["instrumentation"])
    clients = client_records(stage, checks["instrumentation"])
    records, native_errors = load(stage, checks["instrumentation"], config, expected, checks["content"])
    registrations = [r for r in records if r["event"] == "state_registration"]
    reference_census = None
    for row in registrations:
        if row.get("complete") is not True:
            continue
        try:
            census = layer_keys(row.get("expected_layers"))
            if reference_census is None:
                reference_census = census
            checks["content"].expect(census == reference_census,
                                       f"{row['engine_id']}/TP{row['tp_rank']}: complete registration differs between workers")
        except (ValueError, TypeError) as error:
            checks["content"].missing.append(f"Invalid complete registration: {error}")
    native = defaultdict(list)
    for row in records:
        if row["event"] not in ("frontend_select", "state_registration"):
            native[identity(row)].append(row)
    transfers, summaries, reserves = {}, [], {}
    content_seen, native_qualified = set(), set()
    for key, target in native.items():
        if not any(r["role"] == "decode" for r in target):
            continue
        sources = {(r.get("remote_engine_id"), r.get("remote_request_id")) for r in target
                   if r.get("remote_engine_id") or r.get("remote_request_id")}
        complete_sources = {v for v in sources if all(isinstance(s, str) and s for s in v)}
        checks["instrumentation"].require(len(complete_sources) >= 1, f"{key}: missing native remote source identity")
        checks["instrumentation"].require(sources == complete_sources, f"{key}: partial remote source identity")
        checks["content"].expect(len(sources) <= 1, f"{key}: conflicting native remote source identities")
        if len(complete_sources) != 1:
            continue
        source_key = next(iter(complete_sources))
        transfers[key] = source_key
        source = native.get(source_key, [])
        checks["instrumentation"].require(source, f"{key}: no source request records for {source_key}")
        for rows in (source, target):
            for rank in TP_RANKS:
                worker = [r for r in rows if r.get("tp_rank") == rank]
                checks["instrumentation"].require(worker, f"{key}: incomplete TP{rank} records")
                checks["instrumentation"].expect(len({(r["host"], r["pid"], r["dp_rank"]) for r in worker}) <= 1,
                                                  f"{key}/TP{rank}: contradictory worker/DP identity")
            checks["instrumentation"].expect(len({r["dp_rank"] for r in rows}) <= 1, f"{key}: multiple DP owners")
        summary = {"decode_engine_id": key[0], "request_id": key[1],
                   "remote_engine_id": source_key[0], "remote_request_id": source_key[1]}
        reserve = next((r for r in target if r["event"] == "target_reserve"), None)
        client = reserve.get("client_request_id") if reserve else None
        checks["instrumentation"].require(isinstance(client, str) and client, f"{key}: missing/invalid frozen client join")
        if not isinstance(client, str):
            client = None
        content_selected = client not in readiness_ids
        summary.update(client_request_id=client, content_selected=content_selected)
        if content_selected:
            content_seen.add(client)
            summary.update(check_content(source, target, config, registrations, checks["content"]))
            summary["content_status"] = "CHECKED"
        else:
            summary.update(compared_rows=0, different_local_page_ids=False, content_status="NOT_SELECTED_BY_FROZEN_MASK")
        reserves[key] = check_lifetime(source, target, checks["lifetime"],
                                      prefix_caching_disabled=config.get("engine", {}).get("enable_prefix_caching") is False,
                                      content_selected=content_selected, explicit_mask=explicit_mask)
        tokens = token_attribution(source, target, reserves[key], config, checks["tail_and_decode"],
                                   frozen_prompt=expected.get(client, {}).get("num_prefill_tokens"))
        summary["tokens"] = {name: value for name, value in tokens.items() if name != "source_transfer_metadata"}
        check_mapping(source, target, reserves[key], config, registrations, tokens, checks["mapping"])
        ready = check_native_readiness(source, target, checks["native_readiness"], no_fence=client in readiness_ids)
        summary.update(native_completion_witness=ready, independent_no_fence_witness=ready and client in readiness_ids)
        if ready and client in readiness_ids:
            native_qualified.add(client)
        summary["disposition"] = request_disposition(target, reserves[key], expected, clients, checks["cancellation"])
        summary["release_timing"] = release_timing(source, target)
        check_tail(source, target, reserves[key], observation.get("region_size"), checks["tail_and_decode"],
                   cancellation_expected=summary["disposition"]["expected_outcome"] == "cancelled")
        summaries.append(summary)
    checks["instrumentation"].require(transfers, "No native transfers identified")
    for name in ("content", "mapping", "native_readiness", "lifetime", "tail_and_decode", "cancellation"):
        checks[name].require(transfers, "No native transfers identified for this check")
    checks["content"].require(content_ids and content_ids <= content_seen, "Missing frozen content-selected transfer evidence")
    for key, rows in native.items():
        if any(r["event"] == "source_hold" for r in rows):
            uses = list(transfers.values()).count(key)
            checks["lifetime"].require(uses > 0, f"{key}: held source has no observed target transfer")
            checks["lifetime"].expect(uses <= 1, f"{key}: more than one target claims the source pending owner")
    placement(records, transfers, reserves, expected, checks["placement_and_requests"])
    if explicit_mask:
        readiness_coverage(expected, readiness_ids, native_qualified,
                           checks["placement_and_requests"].details.get("waves", {}), checks["native_readiness"])
    checks["native_readiness"].details["explicit_selection_masks"] = explicit_mask
    lengths = {r["num_prompt_tokens"] for r in reserves.values() if r and integer(r.get("num_prompt_tokens"))}
    for name, size in (("region", observation.get("region_size")),
                       ("page", observation.get("page_size", config.get("engine", {}).get("block_size")))):
        checks["boundaries"].require(integer(size) and size > 1, f"Missing explicit {name} size")
        if integer(size) and size > 1:
            triplet = any(n % size == 0 and {n - 1, n, n + 1} <= lengths for n in lengths)
            checks["boundaries"].require(triplet, f"Missing before/on/after the same {name} boundary")
    checks["tail_and_decode"].require(integer(observation.get("region_size")) and observation["region_size"] > 0,
                                    "No independently specified region size for decode crossing")
    checks["boundaries"].require(any(r["different_local_page_ids"] for r in summaries), "No unequal source/target local page IDs exercised")
    checks["runtime"].require(platform_status == "Succeeded", f"Platform status is {platform_status}, not Succeeded")
    checks["runtime"].details.update(platform_status=platform_status, native_errors=native_errors)
    results = {name: check.result() for name, check in checks.items()}
    if native_errors:
        results["runtime"]["status"] = "BLOCKED_ON_RUNTIME"
    statuses = {r["status"] for r in results.values()}
    status = next((v for v in ("BLOCKED_ON_RUNTIME", "FAIL", "UNVERIFIED") if v in statuses), "PASS")
    return {"stage": str(stage), "scope": "R2 normal/cancelled content, logical mapping, native readiness and lifetime diagnostics",
            "diagnostic_only": True, "clean_timing_eligible": False, "status": status,
            "scope_pass": status == "PASS", "all_pass": status == "PASS",
            "ticket08_status": "UNVERIFIED", "ticket08_accepted": False,
            "checks": results, "transfers": summaries,
            "unverified_criteria": UNVERIFIED_CRITERIA + ([] if explicit_mask else ["Legacy mask-free evidence does not qualify independent no-fence probe/wave coverage."]),
            "clock_rule": "Order only same-host CLOCK_MONOTONIC; cross-host correlation uses native P request identity, never clock values.",
            "runtime_limits": {k: config.get("engine", {}).get(k) for k in ("max_num_seqs", "max_model_len", "max_num_batched_tokens")},
            "r1_note": "R1 output/metrics/startup evidence is separate and cannot establish R2 content or lifetime."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--platform-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = judge(args.stage_dir, args.platform_status)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"status": result["status"], "diagnostic_only": True, "clean_timing_eligible": False}))
    return 0  # A diagnostic verdict is evidence, never workload/job control.


if __name__ == "__main__":
    raise SystemExit(main())
