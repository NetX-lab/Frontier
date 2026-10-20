"""Pure offline fault matrix for R2: no vLLM, GPU, platform or cloud access."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tests.integration.step5_pd import r2_verdict


LAYERS = [
    {"group_id": 0, "layer": "sparse", "components": ["raw_kv", "summary_mean"]},
    {"group_id": 1, "layer": "swa", "components": ["raw_kv"]},
]
GROUPS = [
    {"group_id": 0, "native_group_id": 0, "block_size": 16, "sliding_window": None},
    {"group_id": 1, "native_group_id": 1, "block_size": 16, "sliding_window": 32},
]


def digest(rank, group, component, page):
    return hashlib.sha256(f"{rank}:{group}:{component}:{page}".encode()).hexdigest()


def configuration():
    return {"engine": {"block_size": 16, "max_num_seqs": 1, "enable_prefix_caching": False},
            "pd_state_observation": {"region_size": 8, "expected_layers": deepcopy(LAYERS), "group_specs": deepcopy(GROUPS)}}


def observations(lengths=(63, 64, 65)):
    """Synthetic native rows: distinct page IDs, a P superset and an SWA suffix."""
    rows, trace = [], []
    for index, prompt in enumerate(lengths):
        lane = index % 2
        p_engine, d_engine = f"P{lane}", f"D{lane}"
        p_id, d_id, client = f"p-native-{index}", f"d-native-{index}", f"client-{index}"
        external = (prompt - 16) // 16 * 16
        trace.append({"request_id": client, "num_prefill_tokens": prompt,
                      "num_decode_tokens": 10, "arrived_at": index * 0.01})
        pages = (prompt + 15) // 16
        start = max(0, (external - 32 + 1) // 16)
        source_blocks = [list(range(10, 10 + pages)), [0] * start + list(range(10000 + start, 10000 + pages))]
        target_blocks = [list(range(1000, 1000 + pages)), [0] * start + list(range(20000 + start, 20000 + pages))]
        wire = [group[:external // 16] for group in source_blocks]
        local = [group[:external // 16] for group in target_blocks]
        wire = [[v for v in group if v] for group in wire]
        local = [[v for v in group if v] for group in local]
        advertised = [[v for v in group[:prompt // 16] if v] for group in source_blocks]

        def event(name, role, time, rank=None, **extra):
            # P's arbitrary clock origin is much greater than D's. Comparing
            # those clocks would incorrectly reject every otherwise valid case.
            base = (10**15 if role == "prefill" else 1000) + index * 1000
            row = {"event": name, "role": role, "host": f"{role}-{lane}",
                   "pid": 100 + rank if rank is not None else 1, "dp_rank": lane,
                   "monotonic_ns": base + time, "request_id": p_id if role == "prefill" else d_id,
                   "engine_id": p_engine if role == "prefill" else d_engine}
            if rank is not None:
                row["tp_rank"] = rank
            if role == "decode":
                row.update(remote_engine_id=p_engine, remote_request_id=p_id)
            row.update(deepcopy(extra))
            rows.append(row)

        event("source_hold", "prefill", 10, block_ids=source_blocks, num_computed_tokens=prompt,
              num_matched_tokens=external, kv_xfer_params={"remote_num_tokens": prompt,
                  "remote_block_ids": advertised, "remote_engine_id": p_engine, "remote_request_id": p_id})
        event("target_reserve", "decode", 10, block_ids=target_blocks,
              num_prompt_tokens=prompt, num_external_tokens=external, client_request_id=client, wave="wave")
        for role in ("prefill", "decode"):
            event("frontend_select", role, 1, client_request_id=client, wave="wave",
                  load_snapshot=[{"dp_rank": 0, "waiting": 1, "running": 1},
                                 {"dp_rank": 1, "waiting": 0, "running": 1}])
        for rank in range(8):
            for spec in LAYERS:
                group = spec["group_id"]
                for component in spec["components"]:
                    common = dict(group_id=group, layer=spec["layer"], component=component, block_bytes=256)
                    event("source_content", "prefill", 11, rank, **common,
                          local_block_ids=[page for page in source_blocks[group] if page],
                          block_hashes=[digest(rank, group, component, page) for page in source_blocks[group] if page])
                    event("target_content", "decode", 45, rank, **common,
                          local_block_ids=local[group], remote_block_ids=wire[group],
                          block_hashes=[digest(rank, group, component, page) for page in wire[group]])
            event("read_metadata", "decode", 15, rank, remote_num_tokens=prompt,
                  advertised_remote_block_ids=advertised, reserved_local_block_ids=target_blocks,
                  local_physical_block_ids=target_blocks)
            event("read_submit", "decode", 20, rank, local_block_ids=local, remote_block_ids=wire)
            event("read_network_done", "decode", 30, rank)
            event("native_read_ready", "decode", 35, rank, observer_fence=False,
                  boundary="native_get_transfer_results_finished_recving")
            event("read_gpu_visible", "decode", 40, rank, observer_fence=True,
                  boundary="native_postprocessing_then_diagnostic_cuda_synchronize")
            event("target_compute", "decode", 50, rank, positions=list(range(external, prompt)),
                  num_computed_tokens=external, num_scheduled_tokens=prompt - external)
            event("target_compute", "decode", 60, rank, positions=list(range(prompt, prompt + 10)))
            event("source_notified", "prefill", 60 + rank, rank)
        source_count = len({v for group in source_blocks for v in group} - {0})
        target_count = len({v for group in target_blocks for v in group} - {0})
        event("source_free", "prefill", 80, block_ids=source_blocks,
              pool_free_before=100, pool_free_after=100 + source_count, released_block_count=source_count, counts_scope="request_release")
        event("target_free", "decode", 80, block_ids=target_blocks,
              pool_free_before=100, pool_free_after=100 + target_count, released_block_count=target_count,
              counts_scope="request_release", last_sched_seq=7, processed_step_seq=7, defer_block_free=True)
    return rows, trace


def write_stage(tmp_path, rows, trace, *, config=None, client_records=()):
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    (stage / "config.json").write_text(json.dumps(configuration() if config is None else config))
    trace_waves = {}
    for row in trace:
        trace_waves.setdefault(row.get("wave", "wave"), []).append(row)
    for wave, wave_rows in trace_waves.items():
        (stage / "trace" / wave).mkdir(parents=True, exist_ok=True)
        (stage / "trace" / wave / "request_ids.json").write_text(json.dumps({"rows": wave_rows}))
    if client_records:
        captured_waves = {}
        for row in client_records:
            captured_waves.setdefault(row.get("wave", "wave"), []).append(row)
        for wave, wave_rows in captured_waves.items():
            (stage / "evidence/waves" / wave).mkdir(parents=True, exist_ok=True)
            (stage / "evidence/waves" / wave / "client_requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in wave_rows))
    by_host = {}
    for row in rows:
        by_host.setdefault(row.get("host", "decode-0"), []).append(row)
    for host, records in by_host.items():
        directory = stage / "evidence/nodes" / host / "pd_state"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return stage


def result(tmp_path, rows=None, trace=None, **kwargs):
    if rows is None:
        rows, trace = observations()
    return r2_verdict.judge(write_stage(tmp_path, rows, trace, **kwargs), "Succeeded")


def first(rows, event, rank=None, index=0, component=None):
    return next(row for row in rows if row["event"] == event
                and row.get("tp_rank") == rank and row.get("request_id", "").endswith(f"-{index}")
                and (component is None or row.get("component") == component))


def assert_check(verdict, check, status):
    assert verdict["checks"][check]["status"] == status, verdict["checks"][check]


def test_complete_scope_is_not_full_ticket_or_clean_timing_acceptance(tmp_path):
    verdict = result(tmp_path)
    assert verdict["status"] == "PASS", verdict["checks"]
    assert verdict["scope_pass"] and verdict["all_pass"]
    assert verdict["diagnostic_only"] is True and verdict["clean_timing_eligible"] is False
    assert verdict["ticket08_status"] == "UNVERIFIED" and verdict["ticket08_accepted"] is False
    assert len(verdict["transfers"]) == 3
    assert [row["compared_rows"] for row in verdict["transfers"]] == [48, 64, 64]
    assert all(row["different_local_page_ids"] for row in verdict["transfers"])
    assert verdict["runtime_limits"]["max_num_seqs"] == 1
    assert any("Cancellation" in entry for entry in verdict["unverified_criteria"])
    timing = verdict['transfers'][0]['release_timing']
    assert timing['source_hold_to_free']['elapsed_ns'] == 70
    assert timing['last_source_notification_to_free']['elapsed_ns'] == 13
    assert timing['target_reservation_to_free']['elapsed_ns'] == 70
    assert timing['last_compute_entry_to_target_free']['elapsed_ns'] == 20


@pytest.mark.parametrize('event', sorted(r2_verdict.METADATA_EVENTS))
def test_actual_collector_metadata_is_not_missing_request_evidence(tmp_path, event):
    rows, trace = observations()
    rows.append(dict(event=event, verified=True))
    assert result(tmp_path, rows, trace)['scope_pass']


def test_unattributed_native_release_cannot_become_an_aggregate_pool_pass(tmp_path):
    rows, trace = observations()
    rows.append(dict(event='release_unverified', reason='selected_deferred_release_missing_per_call_pool_boundary'))
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, 'instrumentation', 'UNVERIFIED')
    assert any('selected_deferred_release' in value for value in verdict['checks']['instrumentation']['missing'])


def test_matching_bytes_must_cover_first_recomputed_query_swa_window(tmp_path):
    rows, trace = observations()
    for row in rows:
        if row.get('request_id') != 'd-native-0':
            continue
        if row['event'] == 'read_submit':
            row['remote_block_ids'][1] = row['remote_block_ids'][1][1:]
            row['local_block_ids'][1] = row['local_block_ids'][1][1:]
        if row['event'] == 'target_content' and row['group_id'] == 1:
            for field in ('local_block_ids', 'remote_block_ids', 'block_hashes'):
                row[field] = row[field][1:]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, 'content', 'PASS')
    assert_check(verdict, 'mapping', 'FAIL')
    assert not verdict['scope_pass']


@pytest.mark.parametrize('fault,status', [
    ('missing_guard', 'UNVERIFIED'), ('missing_last_seq', 'UNVERIFIED'),
    ('early_last_seq', 'FAIL'), ('pending_deferred_fence', 'FAIL')])
def test_target_release_requires_native_gpu_step_completion_guard(tmp_path, fault, status):
    rows, trace = observations()
    released = first(rows, 'target_free')
    if fault == 'missing_guard':
        released.pop('defer_block_free')
    elif fault == 'missing_last_seq':
        released.pop('last_sched_seq')
    elif fault == 'early_last_seq':
        released['processed_step_seq'] = released['last_sched_seq']-1
    else:
        released['deferred_fence_seq'] = released['processed_step_seq']+1
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, 'lifetime', status)
    assert not verdict['scope_pass']


@pytest.mark.parametrize("component", ["raw_kv", "summary_mean"])
@pytest.mark.parametrize("rank", range(8))
def test_every_tp_rank_and_both_components_have_independently_checked_content(tmp_path, component, rank):
    rows, trace = observations()
    first(rows, "target_content", rank, component=component)["block_hashes"][0] = "f" * 64
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "content", "FAIL")
    assert verdict["status"] == "FAIL"


@pytest.mark.parametrize("event", ["source_content", "target_content", "source_notified", "read_submit",
                                   "read_network_done", "read_gpu_visible", "target_compute"])
def test_missing_one_rank_is_unverified_not_an_exit_or_bytes_pass(tmp_path, event):
    rows, trace = observations()
    rows = [row for row in rows if not (row["event"] == event and row.get("tp_rank") == 7)]
    verdict = result(tmp_path, rows, trace)
    assert verdict["status"] == "UNVERIFIED", verdict["checks"]
    assert not verdict["all_pass"]


@pytest.mark.parametrize("component", ["raw_kv", "summary_mean"])
def test_manifest_catches_completely_absent_component_even_when_all_remaining_rows_match(tmp_path, component):
    rows, trace = observations()
    rows = [r for r in rows if r.get("component") != component]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "content", "UNVERIFIED")


def test_manifest_catches_completely_absent_layer_not_just_missing_rows(tmp_path):
    rows, trace = observations()
    config = {"engine": {"block_size": 16}, "pd_state_observation": {
        "region_size": 8, "expected_layers": LAYERS + [{"group_id": 2, "layer": "absent", "components": ["raw_kv"]}]}}
    verdict = result(tmp_path, rows, trace, config=config)
    assert_check(verdict, "content", "UNVERIFIED")


def test_matching_samples_without_independent_layer_census_are_unverified(tmp_path):
    verdict = result(tmp_path, config={"engine": {"block_size": 16}, "pd_state_observation": {"region_size": 8}})
    assert_check(verdict, "content", "UNVERIFIED")


@pytest.mark.parametrize("complete", [True, False])
def test_complete_native_registration_can_supply_census_but_partial_registration_cannot(tmp_path, complete):
    rows, trace = observations()
    for lane in range(2):
        for role in ("prefill", "decode"):
            for rank in range(8):
                row = deepcopy(next(r for r in rows if r["role"] == role and r["dp_rank"] == lane and r.get("tp_rank") == rank))
                row.update(event="state_registration", complete=complete, expected_layers=LAYERS)
                row.pop("request_id")
                rows.append(row)
    verdict = result(tmp_path, rows, trace, config={"engine": {"block_size": 16}, "pd_state_observation": {"region_size": 8}})
    assert_check(verdict, "content", "PASS" if complete else "UNVERIFIED")


def test_registration_disagreement_is_not_silently_replaced_by_sampled_rows(tmp_path):
    rows, trace = observations()
    row = deepcopy(first(rows, "source_content", 0))
    row.update(event="state_registration", complete=True, expected_layers=LAYERS[:1])
    rows.append(row)
    assert_check(result(tmp_path, rows, trace), "content", "FAIL")


def test_complete_registration_must_agree_across_ranks_even_without_frozen_manifest(tmp_path):
    rows, trace = observations()
    for role in ("prefill", "decode"):
        for rank in range(8):
            row = deepcopy(next(r for r in rows if r["role"] == role and r.get("tp_rank") == rank))
            row.update(event="state_registration", complete=True, expected_layers=LAYERS[:1] if rank == 0 else LAYERS)
            rows.append(row)
    config = {"engine": {"block_size": 16}, "pd_state_observation": {"region_size": 8}}
    assert_check(result(tmp_path, rows, trace, config=config), "content", "FAIL")


@pytest.mark.parametrize("fault", ["wrong_source", "wrong_target", "reordered_wire", "wrong_swa_prefix", "row_bytes", "duplicate_wire"])
def test_actual_read_mapping_not_equal_page_ids_or_a_source_prefix_is_authority(tmp_path, fault):
    rows, trace = observations()
    target = first(rows, "target_content", 0)
    if fault == "wrong_source":
        target["remote_block_ids"] = [999, *target["remote_block_ids"][1:]]
    elif fault == "wrong_target":
        target["local_block_ids"] = [999, *target["local_block_ids"][1:]]
    elif fault == "reordered_wire":
        target["remote_block_ids"] = list(reversed(target["remote_block_ids"]))
        target["block_hashes"] = list(reversed(target["block_hashes"]))
    elif fault == "wrong_swa_prefix":
        target = next(r for r in rows if r["event"] == "target_content" and r["tp_rank"] == 0 and r["group_id"] == 1)
        target["remote_block_ids"] = [10002, *target["remote_block_ids"][1:]]
        target["block_hashes"] = [digest(0, 1, "raw_kv", 10002), *target["block_hashes"][1:]]
    elif fault == "row_bytes":
        target["block_bytes"] = 128
    else:
        target["remote_block_ids"] = [10] * len(target["remote_block_ids"])
    assert_check(result(tmp_path, rows, trace), "content", "FAIL")


@pytest.mark.parametrize("event", ["target_content", "read_submit"])
def test_native_remote_tp_rank_cannot_silently_override_the_tp8_mapping(tmp_path, event):
    rows, trace = observations()
    first(rows, event, 0)["remote_tp_rank"] = 1
    assert_check(result(tmp_path, rows, trace), "content", "FAIL")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_read_pages_must_be_inside_native_source_hold_and_target_reservation(tmp_path, role):
    rows, trace = observations()
    hold = first(rows, "source_hold" if role == "prefill" else "target_reserve")
    hold["block_ids"][0] = [999]
    assert_check(result(tmp_path, rows, trace), "lifetime", "FAIL")


@pytest.mark.parametrize("event,role,time", [
    ("source_notified", "prefill", 1), ("source_free", "prefill", 61),
    ("target_reserve", "decode", 21), ("read_network_done", "decode", 19),
    ("read_gpu_visible", "decode", 29), ("target_compute", "decode", 39),
    ("target_free", "decode", 35), ("target_content", "decode", 39),
    ("target_content", "decode", 51), ("source_content", "prefill", 81),
])
def test_same_host_readiness_and_free_ordering_faults(tmp_path, event, role, time):
    rows, trace = observations()
    rank = None if event in r2_verdict.SCHEDULER_EVENTS else 0
    row = first(rows, event, rank)
    row["monotonic_ns"] = (10**15 if role == "prefill" else 1000) + time
    assert_check(result(tmp_path, rows, trace), "lifetime", "FAIL")


@pytest.mark.parametrize("event", ["source_free", "source_hold", "target_free", "target_reserve"])
def test_double_free_or_ambiguous_scheduler_ownership_is_fail(tmp_path, event):
    rows, trace = observations()
    rows.append(deepcopy(first(rows, event)))
    assert_check(result(tmp_path, rows, trace), "lifetime", "FAIL")


@pytest.mark.parametrize("fault", ["different_blocks", "pool_decreased", "wrong_delta", "missing_counts"])
def test_release_records_are_not_replaced_by_zero_usage_gauges(tmp_path, fault):
    rows, trace = observations()
    row = first(rows, "source_free")
    if fault == "different_blocks":
        row["block_ids"] = [[999], []]
    elif fault == "pool_decreased":
        row["pool_free_after"] = 99
    elif fault == "wrong_delta":
        row["released_block_count"] = 3
    else:
        row.pop("pool_free_before")
    assert_check(result(tmp_path, rows, trace), "lifetime", "UNVERIFIED" if fault == "missing_counts" else "FAIL")


@pytest.mark.parametrize("event,expected_count", [("source_free", 7), ("target_free", 7)])
@pytest.mark.parametrize("delta", [0, 1, 6, 9])
def test_exact_release_delta_rejects_leaks_and_over_release_without_a_reported_count(tmp_path, event, expected_count, delta):
    rows, trace = observations()
    row = first(rows, event)
    row.pop("released_block_count")
    row["pool_free_after"] = row["pool_free_before"] + delta
    assert delta != expected_count
    assert_check(result(tmp_path, rows, trace), "lifetime", "FAIL")


@pytest.mark.parametrize("scope", [None, "entire_native_deferred_drain_call"])
def test_unattributed_pool_delta_is_unverified_even_when_it_matches_by_coincidence(tmp_path, scope):
    rows, trace = observations()
    row = first(rows, "source_free")
    if scope is None:
        row.pop("counts_scope")
    else:
        row["counts_scope"] = scope
    assert_check(result(tmp_path, rows, trace), "lifetime", "UNVERIFIED")


@pytest.mark.parametrize("enabled", [None, True])
def test_exact_pool_accounting_requires_known_disabled_prefix_caching(tmp_path, enabled):
    rows, trace = observations()
    engine = {"block_size": 16}
    if enabled is not None:
        engine["enable_prefix_caching"] = enabled
    config = {"engine": engine, "pd_state_observation": {"region_size": 8, "expected_layers": LAYERS}}
    assert_check(result(tmp_path, rows, trace, config=config), "lifetime", "UNVERIFIED")


def test_release_count_deduplicates_cross_group_aliases_and_excludes_null_padding(tmp_path):
    rows, trace = observations()
    hold, free = first(rows, "source_hold"), first(rows, "source_free")
    hold["block_ids"] = [hold["block_ids"][0] + [44, 44, 0, 0], hold["block_ids"][1] + [44, 0]]
    released = len({block for group in hold["block_ids"] for block in group} - {0})
    free.update(block_ids=deepcopy(hold["block_ids"]),
                pool_free_after=free["pool_free_before"]+released, released_block_count=released)
    assert result(tmp_path, rows, trace)["scope_pass"]


def cancelled_observations():
    rows, trace = observations()
    trace[0]["cancel_after_first_content"] = True
    rows = [r for r in rows if not (r["event"] == "target_compute" and r["request_id"] == "d-native-0"
                                   and "num_computed_tokens" not in r)]
    first(rows, "target_free")["request_status"] = "FINISHED_ABORTED"
    clients = [{"request_id": "client-0", "client_cancelled": True, "stream_completed": False}]
    return rows, trace, clients


@pytest.mark.parametrize("field", ["request_status", "finish_reason", "status"])
@pytest.mark.parametrize("value", ["FINISHED_ABORTED", "RequestStatus.FINISHED_ABORTED"])
def test_verified_cancellation_checks_prefix_and_lifetime_without_requiring_next_decode_crossing(tmp_path, field, value):
    rows, trace, clients = cancelled_observations()
    free = first(rows, "target_free")
    free.pop("request_status")
    free[field] = value
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert verdict["scope_pass"], verdict["checks"]
    for check in ("content", "lifetime", "tail_and_decode", "cancellation"):
        assert_check(verdict, check, "PASS")
    disposition = verdict["transfers"][0]["disposition"]
    assert disposition["expected_outcome"] == "cancelled"
    assert disposition["observed_outcome"] == "cancelled"
    assert "usage" not in disposition and "completion_tokens" not in disposition
    assert "usage" not in clients[0]
    assert not verdict["ticket08_accepted"] and not verdict["clean_timing_eligible"]


@pytest.mark.parametrize("fault,expected", [
    ("missing_native_abort", "UNVERIFIED"), ("unknown_native_status", "UNVERIFIED"),
    ("missing_client", "UNVERIFIED"), ("wrong_client", "UNVERIFIED"),
    ("missing_client_flag", "UNVERIFIED"), ("client_not_cancelled", "FAIL"),
    ("native_completed", "FAIL"), ("conflicting_status", "FAIL"), ("client_completed", "FAIL"),
])
def test_cancellation_requires_matching_client_and_explicit_native_abort_evidence(tmp_path, fault, expected):
    rows, trace, clients = cancelled_observations()
    free = first(rows, "target_free")
    if fault == "missing_native_abort":
        free.pop("request_status")
    elif fault == "unknown_native_status":
        free["request_status"] = "4"
    elif fault == "missing_client":
        clients = []
    elif fault == "wrong_client":
        clients[0]["request_id"] = "other-client"
    elif fault == "missing_client_flag":
        clients[0].pop("client_cancelled")
    elif fault == "client_not_cancelled":
        clients[0]["client_cancelled"] = False
    elif fault == "native_completed":
        free["request_status"] = "FINISHED_LENGTH_CAPPED"
    elif fault == "conflicting_status":
        free["finish_reason"] = "FINISHED_LENGTH_CAPPED"
    else:
        clients[0]["stream_completed"] = True
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, "cancellation", expected)
    assert verdict["status"] == expected
    assert verdict["transfers"][0]["disposition"]["observed_outcome"] != "normal_completed"


@pytest.mark.parametrize("status", [4, None, {"name": "FINISHED_ABORTED"}])
def test_unknown_native_abort_representation_is_not_invented_from_values(tmp_path, status):
    rows, trace, clients = cancelled_observations()
    first(rows, "target_free")["request_status"] = status
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, "cancellation", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_captured_cancellation_requires_a_boolean_not_a_truthy_value(tmp_path, flag):
    rows, trace, clients = cancelled_observations()
    clients[0]["client_cancelled"] = flag
    assert_check(result(tmp_path, rows, trace, client_records=clients), "cancellation", "UNVERIFIED")


def test_cancel_after_first_content_does_not_exempt_missing_prompt_tail_scheduling(tmp_path):
    rows, trace, clients = cancelled_observations()
    row = first(rows, "target_compute", 7)
    row["positions"] = row["positions"][:-1]
    row["num_scheduled_tokens"] -= 1
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, "tail_and_decode", "UNVERIFIED")


@pytest.mark.parametrize("fault,check,expected", [
    ("raw_content", "content", "FAIL"), ("summary_content", "content", "FAIL"),
    ("missing_read_rank", "lifetime", "UNVERIFIED"), ("early_source_free", "lifetime", "FAIL"),
    ("early_target_free", "lifetime", "FAIL"), ("leaked_pages", "lifetime", "FAIL"),
])
def test_cancellation_never_bypasses_content_rank_order_or_release_accounting(tmp_path, fault, check, expected):
    rows, trace, clients = cancelled_observations()
    if fault in ("raw_content", "summary_content"):
        component = "raw_kv" if fault == "raw_content" else "summary_mean"
        first(rows, "target_content", 7, component=component)["block_hashes"][0] = "f" * 64
    elif fault == "missing_read_rank":
        rows = [r for r in rows if not (r["event"] == "read_network_done" and r["request_id"] == "d-native-0" and r["tp_rank"] == 7)]
    elif fault == "early_source_free":
        first(rows, "source_free")["monotonic_ns"] = 10**15 + 61
    elif fault == "early_target_free":
        first(rows, "target_free")["monotonic_ns"] = 1000 + 35
    else:
        first(rows, "target_free").update(pool_free_after=100, released_block_count=0)
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, check, expected)
    assert not verdict["scope_pass"]


def test_cancelled_case_does_not_exempt_normal_cases_from_next_region_crossing(tmp_path):
    rows, trace, clients = cancelled_observations()
    rows = [r for r in rows if not (r["event"] == "target_compute" and r["request_id"] == "d-native-1"
                                   and "num_computed_tokens" not in r)]
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, "cancellation", "PASS")
    assert_check(verdict, "tail_and_decode", "UNVERIFIED")


def test_unplanned_abort_is_not_reported_as_normal_completion(tmp_path):
    rows, trace, clients = cancelled_observations()
    trace[0].pop("cancel_after_first_content")
    verdict = result(tmp_path, rows, trace, client_records=clients)
    assert_check(verdict, "cancellation", "FAIL")
    assert verdict["transfers"][0]["disposition"]["observed_outcome"] != "normal_completed"


@pytest.mark.parametrize("fault", ["wrong_first", "counts_disagree", "noncontiguous", "missing_interval", "no_decode", "no_crossing"])
def test_tail_is_locally_computed_and_subsequent_decode_crosses_a_region(tmp_path, fault):
    rows, trace = observations()
    row = first(rows, "target_compute", 0)
    if fault == "wrong_first":
        row["positions"][0] -= 1
    elif fault == "counts_disagree":
        row["num_scheduled_tokens"] += 1
    elif fault == "noncontiguous":
        row["positions"][1] += 1
    elif fault == "missing_interval":
        row.pop("positions")
        row.pop("num_computed_tokens")
    elif fault == "no_decode":
        rows = [r for r in rows if not (r["event"] == "target_compute" and "num_computed_tokens" not in r)]
    else:
        for r in rows:
            if r["event"] == "target_compute" and "num_computed_tokens" not in r:
                r["positions"] = r["positions"][:1]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "tail_and_decode", "FAIL" if fault in ("wrong_first", "counts_disagree", "noncontiguous") else "UNVERIFIED")


def test_compute_interval_form_needs_no_expanded_positions_array(tmp_path):
    rows, trace = observations()
    for row in rows:
        if row["event"] == "target_compute":
            values = row.pop("positions")
            row.update(num_computed_tokens=values[0], num_scheduled_tokens=len(values))
    assert result(tmp_path, rows, trace)["scope_pass"]


@pytest.mark.parametrize("missing", [False, True])
def test_source_matched_tokens_and_target_loaded_tokens_are_joined_without_clock_comparison(tmp_path, missing):
    rows, trace = observations()
    row = first(rows, "source_hold")
    if missing:
        row.pop("num_computed_tokens")
    else:
        row["num_computed_tokens"] += 16
    assert_check(result(tmp_path, rows, trace), "tail_and_decode", "UNVERIFIED" if missing else "FAIL")


@pytest.mark.parametrize("lengths", [(4095,), (4096,), (4097,), (4095, 4096), (4096, 4097)])
def test_boundary_matrix_requires_before_on_and_after_not_only_a_large_prompt(tmp_path, lengths):
    rows, trace = observations(lengths)
    assert_check(result(tmp_path, rows, trace), "boundaries", "UNVERIFIED")


def test_unrelated_residue_classes_do_not_prove_one_boundary_triplet(tmp_path):
    rows, trace = observations((4095, 8192, 12289))
    assert_check(result(tmp_path, rows, trace), "boundaries", "UNVERIFIED")


@pytest.mark.parametrize("fault", ["one_lane", "missing_frontend", "missing_load", "wrong_dp", "wrong_native", "unaccounted_request"])
def test_concurrent_wave_needs_both_roles_lanes_endpoints_and_request_accounting(tmp_path, fault):
    rows, trace = observations()
    if fault == "one_lane":
        rows = [r for r in rows if r["dp_rank"] == 0]
    elif fault == "missing_frontend":
        rows = [r for r in rows if r["event"] != "frontend_select"]
    elif fault == "missing_load":
        first(rows, "frontend_select").pop("load_snapshot")
    elif fault == "wrong_dp":
        first(rows, "frontend_select")["dp_rank"] = 1
    elif fault == "wrong_native":
        first(rows, "frontend_select")["request_id"] = "not-native-0"
    else:
        trace.append({"request_id": "missing-client", "num_prefill_tokens": 128})
    expected = "FAIL" if fault in ("wrong_dp", "wrong_native") else "UNVERIFIED"
    assert_check(result(tmp_path, rows, trace), "placement_and_requests", expected)


@pytest.mark.parametrize("fault", ["bad_json", "unknown_event", "observer_error", "missing_tp", "missing_host", "bad_native_id", "truncated_hashes"])
def test_incomplete_or_broken_observations_remain_unverified(tmp_path, fault):
    rows, trace = observations()
    if fault == "unknown_event":
        rows.append({"event": "some_future_event", "host": "decode-0"})
    elif fault == "observer_error":
        rows.append({"event": "observation_error", "host": "decode-0", "message": "CUDA sample unavailable"})
    elif fault == "missing_tp":
        first(rows, "target_content", 0).pop("tp_rank")
    elif fault == "missing_host":
        first(rows, "target_content", 0).pop("host")
    elif fault == "bad_native_id":
        first(rows, "target_content", 0)["request_id"] = []
    elif fault == "truncated_hashes":
        first(rows, "target_content", 0)["block_hashes"] = []
    stage = write_stage(tmp_path, rows, trace)
    if fault == "bad_json":
        with next((stage / "evidence/nodes").glob("*/pd_state/*.jsonl")).open("a") as stream:
            stream.write('{"event":\n')
    verdict = r2_verdict.judge(stage, "Succeeded")
    assert verdict["status"] == "UNVERIFIED", verdict["checks"]


@pytest.mark.parametrize("field,value", [
    ("component", []), ("block_hashes", None), ("local_block_ids", {}),
    ("remote_block_ids", None), ("remote_engine_id", []), ("remote_request_id", 9),
    ("engine_id", []), ("wave", []), ("group_id", "0"), ("block_bytes", 0),
])
def test_malformed_content_schema_is_reported_without_crashing(tmp_path, field, value):
    rows, trace = observations()
    first(rows, "target_content", 0)[field] = value
    verdict = result(tmp_path, rows, trace)
    assert verdict["status"] == "UNVERIFIED", verdict["checks"]


@pytest.mark.parametrize("field,value", [("engine", None), ("pd_state_observation", []), ("pd_state_observation", None)])
def test_malformed_nested_config_is_unverified(tmp_path, field, value):
    config = {"engine": {"block_size": 16}, "pd_state_observation": {"region_size": 8, "expected_layers": LAYERS}, field: value}
    assert result(tmp_path, config=config)["status"] == "UNVERIFIED"


def test_node_observation_error_remains_visible_even_when_jsonl_looks_complete(tmp_path):
    rows, trace = observations()
    stage = write_stage(tmp_path, rows, trace)
    (stage / "evidence/nodes/decode-0/result.json").write_text(json.dumps({"observation_errors": ["drain incomplete"]}))
    verdict = r2_verdict.judge(stage, "Succeeded")
    assert_check(verdict, "instrumentation", "UNVERIFIED")


def test_runtime_collector_schema_joins_frozen_requests_and_independent_native_registration(tmp_path):
    rows, trace = observations()
    for row in rows:
        for canonical, native in (("block_ids", "blocks"), ("pool_free_before", "pool_free_blocks_before"),
                                  ("pool_free_after", "pool_free_blocks_after"),
                                  ("client_request_id", "selected_client_request_id"),
                                  ("num_external_tokens", "num_external_computed_tokens"),
                                  ("load_snapshot", "lb_engines")):
            if canonical in row:
                row[native] = row.pop(canonical)
        row.pop("wave", None)
        row.pop("num_prompt_tokens", None)
    for lane in range(2):
        for role in ("prefill", "decode"):
            for rank in range(8):
                row = deepcopy(next(r for r in rows if r["role"] == role and r["dp_rank"] == lane and r.get("tp_rank") == rank))
                row.update(event="registration", expected_raw_layers={"sparse": 0, "swa": 1},
                           group_specs=GROUPS,
                           expected_summary_layers={"sparse_mean": "sparse"},
                           registrations=[{"group_id": spec["group_id"], "layer": spec["layer"], "component": component}
                                          for spec in LAYERS for component in spec["components"]])
                row.pop("request_id")
                rows.append(row)
    config = {"engine": {"block_size": 16, "enable_prefix_caching": False}, "pd_state_observation": {
        "region_size": 8, "expected_layer_count": 2, "expected_summary_layer_count": 1}}
    verdict = result(tmp_path, rows, trace, config=config)
    assert verdict["scope_pass"], verdict["checks"]


def test_target_final_allocation_accounts_for_locally_allocated_tail_pages(tmp_path):
    rows, trace = observations()
    reserve = first(rows, "target_reserve")
    allocate = deepcopy(reserve)
    allocate.update(event="block_allocate", monotonic_ns=reserve["monotonic_ns"] + 50)
    allocate["block_ids"] = [reserve["block_ids"][0] + [104], reserve["block_ids"][1]]
    first(rows, "target_free")["block_ids"] = allocate["block_ids"]
    released = len({block for group in allocate["block_ids"] for block in group} - {0})
    first(rows, "target_free").update(pool_free_after=100+released, released_block_count=released)
    rows.append(allocate)
    assert result(tmp_path, rows, trace)["scope_pass"]


def test_deferred_free_queue_entry_is_not_mistaken_for_physical_release(tmp_path):
    rows, trace = observations()
    row = deepcopy(first(rows, "source_free"))
    row.update(event="free_deferred", monotonic_ns=10**15 + 15)
    rows.append(row)
    assert result(tmp_path, rows, trace)["scope_pass"]


def test_configured_engine_id_is_not_a_substitute_for_missing_native_engine_id(tmp_path):
    rows, trace = observations()
    row = first(rows, "target_reserve")
    row["configured_engine_id"] = row.pop("engine_id")
    verdict = result(tmp_path, rows, trace)
    assert verdict["status"] == "UNVERIFIED"


def test_conflicting_native_remote_identity_is_fail(tmp_path):
    rows, trace = observations()
    first(rows, "target_content", 0)["remote_request_id"] = "another-native-request"
    assert_check(result(tmp_path, rows, trace), "content", "FAIL")


def test_out_of_range_tp_rank_is_not_used_to_fill_a_missing_rank(tmp_path):
    rows, trace = observations()
    first(rows, "target_content", 7)["tp_rank"] = 8
    assert_check(result(tmp_path, rows, trace), "instrumentation", "FAIL")


def test_native_error_overrides_partial_instrumentation_without_hiding_failed_checks(tmp_path):
    rows, trace = observations()
    first(rows, "target_content", 0)["block_hashes"][0] = "f" * 64
    rows.append({"event": "native_error", "host": "decode-0", "message": "NIXL READ failed", "request_id": "d-native-0"})
    verdict = result(tmp_path, rows, trace)
    assert verdict["status"] == "BLOCKED_ON_RUNTIME"
    assert_check(verdict, "content", "FAIL")
    assert "NIXL READ failed" in verdict["checks"]["runtime"]["native_errors"][0]["message"]


def test_platform_success_and_byte_counters_cannot_pass_empty_state_evidence(tmp_path):
    verdict = result(tmp_path, [], [])
    assert verdict["status"] == "UNVERIFIED" and not verdict["scope_pass"]
    for check in ("content", "lifetime", "tail_and_decode"):
        assert_check(verdict, check, "UNVERIFIED")


def test_missing_stage_is_a_diagnostic_not_an_exception(tmp_path):
    verdict = r2_verdict.judge(tmp_path / "absent", "Unknown")
    assert verdict["status"] == "UNVERIFIED" and not verdict["scope_pass"]


def test_cli_writes_diagnostic_and_never_returns_a_workload_control_failure(tmp_path, capsys):
    rows, trace = observations()
    stage = write_stage(tmp_path, rows, trace)
    output = tmp_path / "verdict.json"
    assert r2_verdict.main(["--stage-dir", str(stage), "--platform-status", "Failed", "--output", str(output)]) == 0
    verdict = json.loads(output.read_text())
    assert verdict["status"] == "UNVERIFIED"
    assert json.loads(capsys.readouterr().out)["diagnostic_only"] is True


@pytest.mark.parametrize("prompt", [63, 64, 65, 255, 256, 257, 1537])
def test_actual_p_computation_advertised_prompt_and_d_prefix_are_distinct_native_counts(tmp_path, prompt):
    rows, trace = observations((prompt,))
    hold, reserve = first(rows, "source_hold"), first(rows, "target_reserve")
    assert hold["num_computed_tokens"] == hold["kv_xfer_params"]["remote_num_tokens"] == prompt
    assert reserve["num_external_tokens"] == 16 * ((prompt - 16) // 16)
    assert 16 <= prompt - reserve["num_external_tokens"] <= 31
    hold.pop("num_matched_tokens")
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "tail_and_decode", "PASS")
    assert_check(verdict, "mapping", "PASS")


@pytest.mark.parametrize("fault,status", [
    ("advertised_tokens", "FAIL"), ("received_tokens", "FAIL"), ("matched_tokens", "FAIL"),
    ("loaded_prefix", "FAIL"), ("source_below_loaded", "FAIL"), ("wrong_prompt", "FAIL"),
    ("missing_params", "UNVERIFIED"), ("missing_advertised_tokens", "UNVERIFIED"),
    ("missing_received_tokens", "UNVERIFIED"), ("missing_source_count", "UNVERIFIED"),
])
def test_native_prompt_and_loaded_token_attribution_faults(tmp_path, fault, status):
    rows, trace = observations()
    hold, reserve = first(rows, "source_hold"), first(rows, "target_reserve")
    if fault == "advertised_tokens":
        hold["kv_xfer_params"]["remote_num_tokens"] = reserve["num_external_tokens"]
    elif fault == "received_tokens":
        first(rows, "read_metadata", 7)["remote_num_tokens"] = reserve["num_external_tokens"]
    elif fault == "matched_tokens":
        hold["num_matched_tokens"] += 16
    elif fault == "loaded_prefix":
        reserve["num_external_tokens"] += 16
    elif fault == "source_below_loaded":
        hold["num_computed_tokens"] = reserve["num_external_tokens"] - 1
    elif fault == "wrong_prompt":
        reserve["num_prompt_tokens"] += 1
    elif fault == "missing_params":
        hold.pop("kv_xfer_params")
    elif fault == "missing_advertised_tokens":
        hold["kv_xfer_params"].pop("remote_num_tokens")
    elif fault == "missing_received_tokens":
        first(rows, "read_metadata", 7).pop("remote_num_tokens")
    else:
        hold.pop("num_computed_tokens")
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "tail_and_decode", status)
    assert not verdict["scope_pass"]


def change_wire(rows, *, group=0, remote=None, local=None, index=0, rank=0):
    """Make descriptor arrays and target digests agree, independently of scheduler tables."""
    read = first(rows, "read_submit", rank, index)
    if remote is not None:
        read["remote_block_ids"][group] = remote
    if local is not None:
        read["local_block_ids"][group] = local
    for row in rows:
        if (row["event"] == "target_content" and row["request_id"] == f"d-native-{index}"
                and row["tp_rank"] == rank and row["group_id"] == group):
            row["remote_block_ids"] = list(read["remote_block_ids"][group])
            row["local_block_ids"] = list(read["local_block_ids"][group])
            row["block_hashes"] = [digest(rank, group, row["component"], page) for page in read["remote_block_ids"][group]]


@pytest.mark.parametrize("fault", ["different_logical_page", "reversed_pairs", "beyond_external", "missing_full_prefix", "missing_swa_window"])
def test_self_consistent_digests_and_descriptors_do_not_prove_logical_page_mapping(tmp_path, fault):
    rows, trace = observations()
    if fault == "different_logical_page":
        change_wire(rows, local=[1001, 1000])
    elif fault == "reversed_pairs":
        change_wire(rows, remote=[11, 10], local=[1001, 1000])
    elif fault == "beyond_external":
        change_wire(rows, remote=[10, 12], local=[1000, 1002])
    elif fault == "missing_full_prefix":
        change_wire(rows, remote=[10], local=[1000])
    else:
        change_wire(rows, group=1, remote=[], local=[])
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "content", "PASS")
    assert_check(verdict, "mapping", "FAIL")
    assert not verdict["scope_pass"]


def test_hma_null_slots_are_logical_positions_not_compacted_page_lists(tmp_path):
    rows, trace = observations()
    reserve, free = first(rows, "target_reserve", index=1), first(rows, "target_free", index=1)
    assert reserve["block_ids"][1][0] == 0
    compacted = [v for v in reserve["block_ids"][1] if v]
    reserve["block_ids"] = [reserve["block_ids"][0], compacted]
    free["block_ids"] = deepcopy(reserve["block_ids"])
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "content", "PASS")
    assert_check(verdict, "mapping", "FAIL")


@pytest.mark.parametrize("provided", [True, False])
def test_nonidentity_native_group_order_needs_an_explicit_group_map(tmp_path, provided):
    rows, trace = observations()
    for row in rows:
        if row["event"] in ("source_hold", "source_free", "target_reserve", "target_free"):
            row["block_ids"] = list(reversed(row["block_ids"]))
    config = configuration()
    if provided:
        for spec in config["pd_state_observation"]["group_specs"]:
            spec["native_group_id"] = 1 - spec["group_id"]
    else:
        config["pd_state_observation"].pop("group_specs")
    verdict = result(tmp_path, rows, trace, config=config)
    assert_check(verdict, "mapping", "PASS" if provided else "UNVERIFIED")
    assert verdict["status"] == ("PASS" if provided else "UNVERIFIED"), verdict["checks"]


@pytest.mark.parametrize("field", ["native_group_id", "block_size", "sliding_window"])
def test_missing_exact_group_semantics_cannot_be_filled_from_agreeing_rows(tmp_path, field):
    config = configuration()
    config["pd_state_observation"]["group_specs"][1].pop(field)
    verdict = result(tmp_path, config=config)
    assert_check(verdict, "mapping", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("field", ["advertised_remote_block_ids", "reserved_local_block_ids", "local_physical_block_ids"])
@pytest.mark.parametrize("missing", [False, True])
def test_received_mapping_metadata_is_independent_not_rebuilt_from_read_descriptors(tmp_path, field, missing):
    rows, trace = observations()
    row = first(rows, "read_metadata", 7)
    if missing:
        row.pop(field)
    else:
        row[field] = [[999999], []]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "mapping", "UNVERIFIED" if missing else "FAIL")


def test_hma_padding_in_advertised_metadata_is_allowed_but_null_wire_rows_are_not(tmp_path):
    rows, trace = observations()
    for row in rows:
        if row["event"] == "source_hold":
            row["kv_xfer_params"]["remote_block_ids"] = [g[:row["num_computed_tokens"] // 16] for g in row["block_ids"]]
        elif row["event"] == "read_metadata":
            index = int(row["request_id"].rsplit("-", 1)[1])
            row["advertised_remote_block_ids"] = deepcopy(first(rows, "source_hold", index=index)["kv_xfer_params"]["remote_block_ids"])
    assert result(tmp_path, rows, trace)["scope_pass"]


def masked_observations(*, cancelled=False, actual_run=False):
    if actual_run:
        run = json.loads((Path(__file__).parents[1] / "integration/step5_pd/r2_run.json").read_text())
        cases = [(wave["name"], length, wave.get("capture_content") is not False, wave.get("cancel_after_first_content") is True)
                 for wave in run["waves"] for length in wave["prompt_lengths"]]
    else:
        cases = [("content", 63, True, False), ("content", 64, True, False), ("content", 65, True, False),
                 ("native_ready64", 64, False, False), ("native_ready1537", 1537, False, False),
                 ("native_concurrent", 65, False, False), ("native_concurrent", 65, False, False)]
        if cancelled:
            cases[0] = ("cancel1", 63, True, True)
    rows, trace = observations(tuple(case[1] for case in cases))
    content, ready, clients = [], [], []
    for index, (wave, prompt, capture, cancel) in enumerate(cases):
        client = f"client-{index}"
        trace[index].update(wave=wave, cancel_after_first_content=cancel)
        (content if capture else ready).append(client)
        for row in rows:
            if row["request_id"] in (f"p-native-{index}", f"d-native-{index}") and "wave" in row:
                row["wave"] = wave
        if cancel:
            first(rows, "target_free", index=index)["status"] = "RequestStatus.FINISHED_ABORTED"
            clients.append({"request_id": client, "client_cancelled": True, "wave": wave})
    readiness_indexes = {index for index, case in enumerate(cases) if not case[2]}
    cancelled_indexes = {index for index, case in enumerate(cases) if case[3]}
    rows = [row for row in rows if not (
        (int(row["request_id"].rsplit("-", 1)[1]) in readiness_indexes
         and row["event"] in ("source_content", "target_content", "read_gpu_visible"))
        or (int(row["request_id"].rsplit("-", 1)[1]) in cancelled_indexes
            and row["event"] == "target_compute" and "num_computed_tokens" not in row))]
    config = configuration()
    config["pd_state_observation"].update(content_client_request_ids=content, readiness_only_client_request_ids=ready,
                                         selected_client_request_ids=content + ready)
    return rows, trace, config, clients


@pytest.mark.parametrize("cancelled", [False, True])
def test_explicit_masks_separate_content_diagnostics_from_native_no_fence_readiness(tmp_path, cancelled):
    rows, trace, config, clients = masked_observations(cancelled=cancelled)
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert verdict["scope_pass"], verdict["checks"]
    assert len([r for r in verdict["transfers"] if r["content_selected"]]) == 3
    noncontent = [r for r in verdict["transfers"] if not r["content_selected"]]
    assert len(noncontent) == 4 and all(r["compared_rows"] == 0 and r["independent_no_fence_witness"] for r in noncontent)
    assert all(not r["independent_no_fence_witness"] for r in verdict["transfers"] if r["content_selected"])
    assert verdict["checks"]["native_readiness"]["native_concurrent_waves"] == ["native_concurrent"]
    assert len(verdict["checks"]["native_readiness"]["sequential_client_request_ids"]) == 2


def test_actual_twenty_request_fifteen_wave_shape_with_native_cancel_status_and_client_path(tmp_path):
    rows, trace, config, clients = masked_observations(actual_run=True)
    assert len(trace) == 20 and len({row["wave"] for row in trace}) == 15
    assert len(config["pd_state_observation"]["content_client_request_ids"]) == 12
    assert len(config["pd_state_observation"]["readiness_only_client_request_ids"]) == 8
    stage = write_stage(tmp_path, rows, trace, config=config, client_records=clients)
    assert (stage / "evidence/waves/cancel1/client_requests.jsonl").is_file()
    verdict = r2_verdict.judge(stage, "Succeeded")
    assert verdict["scope_pass"], verdict["checks"]
    assert len(verdict["transfers"]) == 20
    assert verdict["checks"]["native_readiness"]["native_concurrent_waves"] == ["concurrent6"]
    assert not verdict["ticket08_accepted"] and not verdict["clean_timing_eligible"]


@pytest.mark.parametrize("fault,status", [
    ("missing_rank", "UNVERIFIED"), ("fence_as_native", "FAIL"), ("missing_fence_flag", "UNVERIFIED"),
    ("missing_boundary", "UNVERIFIED"), ("diagnostic_boundary", "UNVERIFIED"),
    ("early_native", "FAIL"), ("late_native", "FAIL"), ("diagnostic_fence", "FAIL"),
    ("target_readback", "FAIL"), ("source_readback", "FAIL"), ("tail_readback", "FAIL"),
])
def test_no_fence_readiness_requires_native_completion_and_no_pre_forward_diagnostics(tmp_path, fault, status):
    rows, trace, config, clients = masked_observations()
    ready = first(rows, "native_read_ready", 7, index=3)
    if fault == "missing_rank":
        rows.remove(ready)
    elif fault == "fence_as_native":
        ready["observer_fence"] = True
    elif fault == "missing_fence_flag":
        ready.pop("observer_fence")
    elif fault == "missing_boundary":
        ready.pop("boundary")
    elif fault == "diagnostic_boundary":
        ready["boundary"] = "diagnostic_cuda_synchronize"
    elif fault in ("early_native", "late_native"):
        ready["monotonic_ns"] = 1000 + 3 * 1000 + (29 if fault == "early_native" else 51)
    else:
        event = deepcopy(ready)
        event["monotonic_ns"] += 5
        if fault == "diagnostic_fence":
            event.update(event="read_gpu_visible", observer_fence=True)
        elif fault == "target_readback":
            event.update(event="target_content")
        elif fault == "source_readback":
            event = deepcopy(first(rows, "source_notified", 7, index=3))
            event["event"] = "source_content"
        else:
            event.update(event="tail_state")
        rows.append(event)
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert_check(verdict, "native_readiness", status)
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("event", ["read_submit", "read_network_done", "native_read_ready", "read_metadata", "source_notified", "target_compute"])
def test_readiness_only_mask_never_exempts_eight_rank_read_mapping_or_lifetime(tmp_path, event):
    rows, trace, config, clients = masked_observations()
    rows = [r for r in rows if not (r["event"] == event and r["request_id"] in ("d-native-5", "p-native-5") and r.get("tp_rank") == 7)]
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert verdict["status"] == "UNVERIFIED", verdict["checks"]
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("fault", ["wrong_logical_page", "wrong_tokens", "early_free", "leaked_pages", "wrong_dp"])
def test_readiness_only_skips_hashes_not_native_mapping_tokens_free_or_placement(tmp_path, fault):
    rows, trace, config, clients = masked_observations()
    if fault == "wrong_logical_page":
        read = first(rows, "read_submit", 7, index=5)
        read["local_block_ids"][0] = list(reversed(read["local_block_ids"][0]))
    elif fault == "wrong_tokens":
        first(rows, "read_metadata", 7, index=5)["remote_num_tokens"] -= 16
    elif fault == "early_free":
        first(rows, "target_free", index=5)["monotonic_ns"] = 1000 + 5 * 1000 + 29
    elif fault == "leaked_pages":
        first(rows, "target_free", index=5)["pool_free_after"] = 100
    else:
        first(rows, "frontend_select", index=5)["dp_rank"] = 0
    assert result(tmp_path, rows, trace, config=config, client_records=clients)["status"] == "FAIL"


def test_diagnostic_fence_cannot_replace_native_readiness_in_legacy_artifacts(tmp_path):
    rows, trace = observations()
    rows = [r for r in rows if r["event"] != "native_read_ready"]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "native_readiness", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("fault", ["missing_sequential", "missing_concurrent", "concurrent_one_lane", "missing_content", "missing_client", "overlap", "invalid_list"])
def test_explicit_masks_require_both_sequential_probes_and_an_unfenced_concurrent_wave(tmp_path, fault):
    rows, trace, config, clients = masked_observations()
    observation = config["pd_state_observation"]
    if fault == "missing_sequential":
        rows = [r for r in rows if not (r["event"] == "native_read_ready" and r["request_id"] == "d-native-4")]
    elif fault == "missing_concurrent":
        for index in (5, 6):
            trace[index]["wave"] = f"single-{index}"
            for row in rows:
                if row["request_id"] in (f"d-native-{index}", f"p-native-{index}") and "wave" in row:
                    row["wave"] = f"single-{index}"
    elif fault == "concurrent_one_lane":
        # Keep per-request identities consistent but place both concurrent
        # requests on lane 0; content-wave placement cannot substitute for it.
        for row in rows:
            if row["request_id"] in ("d-native-5", "p-native-5"):
                row["dp_rank"] = 0
    elif fault == "missing_content":
        rows = [r for r in rows if not (r["event"] == "target_content" and r["request_id"] == "d-native-0")]
    elif fault == "missing_client":
        observation["selected_client_request_ids"].pop()
    elif fault == "overlap":
        observation["readiness_only_client_request_ids"].append("client-0")
    else:
        observation["readiness_only_client_request_ids"] = "client-3"
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert not verdict["scope_pass"]
    if fault in ("missing_sequential", "missing_concurrent", "concurrent_one_lane"):
        assert_check(verdict, "native_readiness", "UNVERIFIED")


def test_observer_lifecycle_metadata_is_not_a_missing_observation(tmp_path):
    rows, trace = observations()
    rows.extend([{"event": event, "host": "decode-0"} for event in ("observer_started", "hook_installed")])
    assert result(tmp_path, rows, trace)["scope_pass"]


def test_source_received_and_target_token_counts_cannot_jointly_replace_the_frozen_prompt(tmp_path):
    rows, trace = observations()
    first(rows, "source_hold")["num_computed_tokens"] = 62
    first(rows, "source_hold")["kv_xfer_params"]["remote_num_tokens"] = 62
    first(rows, "target_reserve")["num_prompt_tokens"] = 62
    for rank in range(8):
        first(rows, "read_metadata", rank)["remote_num_tokens"] = 62
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "tail_and_decode", "FAIL")
    assert any("independent frozen prompt" in message for message in verdict["checks"]["tail_and_decode"]["failures"])


def test_missing_source_advertised_block_table_is_unverified_not_reconstructed_from_d(tmp_path):
    rows, trace = observations()
    first(rows, "source_hold")["kv_xfer_params"].pop("remote_block_ids")
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "mapping", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("where", ["frozen", "registration"])
def test_explicit_transfer_group_ids_can_supply_the_native_scheduler_mapping(tmp_path, where):
    rows, trace = observations()
    specs = [{key: value for key, value in spec.items() if key != "native_group_id"} for spec in GROUPS]
    config = configuration()
    if where == "frozen":
        config["pd_state_observation"].update(group_specs=specs, transfer_group_ids=[0, 1])
    else:
        config["pd_state_observation"].pop("group_specs")
        for lane in range(2):
            for role in ("prefill", "decode"):
                for rank in range(8):
                    row = deepcopy(next(r for r in rows if r["role"] == role and r["dp_rank"] == lane and r.get("tp_rank") == rank))
                    row.update(event="state_registration", complete=True, expected_layers=LAYERS,
                               group_specs=specs, transfer_group_ids=[0, 1])
                    row.pop("request_id")
                    rows.append(row)
    verdict = result(tmp_path, rows, trace, config=config)
    assert verdict["scope_pass"], verdict["checks"]


def test_unsupported_partial_group_page_coverage_is_unverified_not_a_shape_only_pass(tmp_path):
    rows, trace = observations((79,))
    config = configuration()
    config["pd_state_observation"]["group_specs"][0]["block_size"] = 32
    verdict = result(tmp_path, rows, trace, config=config)
    assert_check(verdict, "mapping", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("rank", range(8))
def test_diagnostic_gpu_visibility_never_fills_a_missing_native_ready_rank(tmp_path, rank):
    rows, trace = observations()
    rows = [row for row in rows if not (row["event"] == "native_read_ready" and row["request_id"] == "d-native-0" and row["tp_rank"] == rank)]
    verdict = result(tmp_path, rows, trace)
    assert_check(verdict, "native_readiness", "UNVERIFIED")
    assert not verdict["scope_pass"]


def test_readiness_only_cannot_drop_a_registered_group_from_all_matching_descriptors(tmp_path):
    rows, trace, config, clients = masked_observations()
    first(rows, "source_hold", index=5)["kv_xfer_params"]["remote_block_ids"].pop()
    for rank in range(8):
        read = first(rows, "read_submit", rank, index=5)
        read["remote_block_ids"].pop()
        read["local_block_ids"].pop()
        metadata = first(rows, "read_metadata", rank, index=5)
        for field in ("advertised_remote_block_ids", "reserved_local_block_ids", "local_physical_block_ids"):
            metadata[field].pop()
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert_check(verdict, "mapping", "FAIL")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("index", [3, 5])
def test_normal_readiness_only_cases_still_require_subsequent_decode_and_region_crossing(tmp_path, index):
    rows, trace, config, clients = masked_observations()
    rows = [row for row in rows if not (row["request_id"] == f"d-native-{index}" and row["event"] == "target_compute"
                                       and "num_computed_tokens" not in row)]
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert_check(verdict, "tail_and_decode", "UNVERIFIED")
    assert not verdict["scope_pass"]


def test_cancelled_content_case_still_rejects_a_digest_consistent_wrong_logical_page(tmp_path):
    rows, trace, config, clients = masked_observations(cancelled=True)
    change_wire(rows, local=[1001, 1000])
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert_check(verdict, "content", "PASS")
    assert_check(verdict, "cancellation", "PASS")
    assert_check(verdict, "tail_and_decode", "PASS")
    assert_check(verdict, "mapping", "FAIL")


def test_aggregate_deferred_drain_counts_stay_unverified_for_unfenced_requests(tmp_path):
    rows, trace, config, clients = masked_observations()
    first(rows, "target_free", index=5)["counts_scope"] = "entire_native_deferred_drain_call"
    verdict = result(tmp_path, rows, trace, config=config, client_records=clients)
    assert_check(verdict, "lifetime", "UNVERIFIED")
    assert not verdict["scope_pass"]


@pytest.mark.parametrize("client", [[], {}, 9, None])
def test_invalid_client_join_remains_a_soft_instrumentation_gap(tmp_path, client):
    rows, trace = observations()
    first(rows, "target_reserve")["client_request_id"] = client
    verdict = result(tmp_path, rows, trace)
    assert verdict["status"] == "UNVERIFIED", verdict["checks"]
    assert not verdict["scope_pass"]
