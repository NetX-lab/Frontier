"""CPU contracts for replay row grouping, signed gaps and aggregate output."""

from dataclasses import replace
import csv
import json
import statistics

import pytest

from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase, segment_class
from frontier.profiling.attention import dsv4_attention_main as main
from frontier.profiling.attention.dsa_rows import KERNEL_ONLY
from frontier.profiling.attention.dsv4_attention_plan import metadata_key
from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows
from frontier.profiling.kernel_gap import dsv4_replay_gaps as gaps
from test_dsv4_attention_profiling import PLAN_ARGS, _engine_record, _metadata
from test_dsv4_graph_bridge_check import bridge
from test_dsv4_replay_gaps import OUT, Q, interval

SCORES = "dsv4_index_scores"
SHORT = "dsv4_index_short_context_fill"
MODE = "FULL_AND_PIECEWISE"


@pytest.fixture
def plan():
    return main.build_plan(main.build_parser().parse_args([
        "plan", *PLAN_ARGS, *bridge.graph_mode_flags(MODE, (1, 2, 4, 8)),
    ]))


def saved_records(plan, specs=None, *, engine_index=0, source="synthetic.jsonl"):
    """Serial graph kernels with an explicit signed PDL gap at the output."""
    specs = specs or [{}]
    engine = dict(_engine_record(engine_index), engine_policy="graph_replay",
                  driver_version="driver", target_tensor_parallel_size=8,
                  runtime_versions={"torch": "2.9.0", "torch_cuda": "12.8", "vllm": "0.28.0"},
                  cudagraph_capture={"cudagraph_mode": MODE}, step_hold_us=0.0, hold_cycles_per_us=None,
                  layers=[dict(global_layer_id=l.global_layer_id, attention_variant=l.attention_variant,
                               compress_ratio=max(1, l.compress_ratio)) for l in plan.shape.layers])
    steps, calls, intervals = [], [], []
    for i, spec in enumerate(specs):
        requests = spec.get("requests", [dict(num_scheduled_tokens=1, num_computed_tokens=64, prompt_len=64)])
        dummy = spec.get("phase") == "dummy"
        tokens = spec.get("tokens", sum(r["num_scheduled_tokens"] for r in requests))
        metadata_requests = [dict(num_scheduled_tokens=tokens, num_computed_tokens=0, prompt_len=tokens)] if dummy else requests
        step = dict(kind="step", engine_index=engine_index, measurement=KERNEL_ONLY,
                    step_index=i, case_id=spec.get("case", "case"), phase=spec.get("phase", "decode"),
                    requests=requests, **_metadata(metadata_requests),
                    cudagraph_dispatch=dict(cg_mode=spec.get("mode", "FULL"), num_tokens=tokens,
                                            num_reqs=spec.get("num_reqs", None if dummy else len(requests))))
        step.update(spec.get("metadata", {}))
        steps.append(step)
        unowned = interval(0, 1, step=i, case=step["case_id"])
        unowned["engine_index"] = engine_index
        intervals.append(unowned)
        scale = spec.get("scale", 1)
        for layer in plan.shape.layers:
            scopes = spec.get("scopes", {}).get(layer.global_layer_id, (Q, OUT))
            end = 100 * layer.global_layer_id + 1
            for scope in scopes:
                # Two Q kernels exercise summing the entry and internal gaps.
                durations = (2, 4) if scope == Q else (5,)
                for k, duration in enumerate(durations):
                    gap = -1 if scope == OUT else 3 if scope == scopes[0] and k == 0 else 1
                    start = end + gap * scale
                    mode = step["cudagraph_dispatch"]["cg_mode"]
                    row = interval(start, start + duration * scale, scope=scope, layer=layer.global_layer_id,
                                   step=i, case=step["case_id"], launch="eager" if mode == "NONE" else "graph",
                                   capture_kind=None if mode == "NONE" else "graph" if mode == "FULL" else "breakable_segment",
                                   gap=gap * scale if mode != "NONE" else None)
                    row["engine_index"] = engine_index
                    intervals.append(row)
                    end = row["end_us"]
                calls.append(dict(kind="call", engine_index=engine_index, measurement=KERNEL_ONLY,
                                  case_id=step["case_id"], step_index=i, global_layer_id=layer.global_layer_id,
                                  scope=scope, time_ms=sum(durations) * scale * 1e-3,
                                  kernel_count=len(durations)))
    return gaps.ReplayRecords(source, engine, tuple(steps), tuple(calls), tuple(intervals))


def _uncompressed(rows):
    return next(row for row in rows if row["attention_variant"] == "uncompressed")


def test_same_key_pools_layers_and_steps_and_statistics_keep_signed_gaps(plan):
    rows = build_dsv4_replay_rows(plan, [saved_records(plan, [{"scale": 1}, {"scale": 3}])])
    assert len(rows) == 5
    row = _uncompressed(rows)
    assert row["global_layer_ids"] == "[0, 1]"
    assert row["num_layers_sampled"] == row["num_steps_sampled"] == 2
    for prefix, values in [(f"time_stats.{Q}", [0.006, 0.006, 0.018, 0.018]),
                           (f"gap_stats.{Q}", [0.004, 0.004, 0.012, 0.012]),
                           (f"gap_stats.{OUT}", [-0.001, -0.001, -0.003, -0.003]),
                           ("layer_window_stats", [0.014, 0.014, 0.042, 0.042])]:
        expected = dict(min=min(values), max=max(values), mean=statistics.fmean(values),
                        median=statistics.median(values), std=statistics.pstdev(values), count=len(values))
        for name, value in expected.items():
            assert row[f"{prefix}.{name}"] == pytest.approx(value)
    assert row[f"time_stats.{Q}.kernel_count"] == 2
    assert json.loads(row["operator_scopes"]) == {Q: [Q], OUT: [OUT]}
    assert f"gap_stats.{Q}.kernel_count" not in row


@pytest.mark.parametrize("coordinate", ["engine", "planner", "mode", "tokens", "num_reqs", "requests", "metadata", "scopes"])
def test_key_separates_each_coordinate(plan, coordinate):
    if coordinate == "planner":
        rows = build_dsv4_replay_rows(plan, [saved_records(plan)])
        assert {row["plans_tile_schedule"] for row in rows if row["attention_variant"] == "csa"} == {True, False}
        return
    if coordinate == "engine":
        rows = build_dsv4_replay_rows(plan, [saved_records(plan), saved_records(plan, engine_index=1, source="second.jsonl")])
        assert len(rows) == 10
        return
    changes = {
        "mode": {"mode": "PIECEWISE"}, "tokens": {"tokens": 2}, "num_reqs": {"num_reqs": 2},
        "requests": {"requests": [dict(num_scheduled_tokens=1, num_computed_tokens=65, prompt_len=64)]},
        "metadata": {"metadata": {metadata_key("uncompressed", "swa"): {"block_size": 128, "num_decodes": 1, "num_prefills": 0}}},
        "scopes": {"scopes": {0: (Q,), 1: (Q,)}},
    }
    rows = build_dsv4_replay_rows(plan, [saved_records(plan, [{}, changes[coordinate]])])
    assert len([row for row in rows if row["attention_variant"] == "uncompressed"]) == 2


def test_row_order_uses_case_variant_role_and_first_step(plan):
    rows = build_dsv4_replay_rows(plan, [saved_records(plan, [dict(case="z", tokens=8), dict(case="a", tokens=4), dict(case="a", tokens=2)])])
    assert [r["case_id"] for r in rows] == ["a"] * 10 + ["z"] * 5
    csa = [r for r in rows if r["case_id"] == "a" and r["attention_variant"] == "csa"]
    assert [(r["plans_tile_schedule"], r["padded_rows"]) for r in csa] == [(True, 4), (True, 2), (False, 4), (False, 2)]


def test_single_call_row_composes_own_scope_costs_and_gaps_to_window(plan):
    row = next(row for row in build_dsv4_replay_rows(plan, [saved_records(plan)])
               if row["attention_variant"] == "csa" and row["plans_tile_schedule"])
    assert row["layer_window_stats.count"] == 1
    scopes = json.loads(row["operator_scopes"])
    result = compose_attention_phase(
        {s: row[f"time_stats.{s}.median"] + row[f"gap_stats.{s}.median"] for s in scopes},
        {s: row[f"time_stats.{s}.kernel_count"] for s in scopes},
        {segment_class(row["graph_runtime_mode"], s): 0.0 for s in scopes},
        variant=row["attention_variant"], graph_runtime_mode=row["graph_runtime_mode"], rows=row["padded_rows"],
        multi_stream_gemm_token_threshold=row["multi_stream_gemm_token_threshold"],
        indexer_short_context=row["indexer_short_context"],
    )
    assert result.elapsed_ms == pytest.approx(row["layer_window_stats.median"])


def test_dummy_has_native_layout_without_request_or_key_columns(plan):
    rows = build_dsv4_replay_rows(plan, [saved_records(plan, [dict(phase="dummy", requests=[], tokens=2)])])
    for row in rows:
        assert row["padded_rows"] == 2 and row["dispatch_num_requests"] == ""
        assert row["phase"] == "dummy" and row["swa_metadata.num_prefills"] == 1
        assert not {"batch_size", "total_tokens", "max_key_len", "swa_key_len", "compressed_key_len",
                    "selected_compressed_key_len", "prefill_seq_lens", "decode_kv_cache_sizes"} & row.keys()


def test_mixed_first_tail_keeps_request_load_without_pure_batch_context(plan):
    requests = [dict(num_scheduled_tokens=1, num_computed_tokens=63, prompt_len=64),
                dict(num_scheduled_tokens=1, num_computed_tokens=80, prompt_len=64)]
    row = _uncompressed(build_dsv4_replay_rows(plan, [saved_records(plan, [dict(phase="first_tail", requests=requests)])]))
    assert row["is_mixed_batch"] and not row["is_prefill"]
    assert row["num_prefill_seqs"] == row["num_decode_seqs"] == 1
    assert row["total_prefill_tokens"] == row["total_decode_tokens"] == 1
    assert row["batch_size"] == row["total_tokens"] == 2 and row["max_key_len"] == 81
    assert row["prefill_kv_cache_sizes"] == "[63]" and row["decode_kv_cache_sizes"] == "[80]"
    assert "prefill_chunk_size" not in row and "kv_cache_size" not in row


@pytest.mark.parametrize("scope,scores,short", [(SCORES, True, False), (SHORT, False, True), (None, False, False)])
def test_index_flags_follow_observed_scopes_instead_of_length_rule(plan, scope, scores, short):
    csa_ids = [layer.global_layer_id for layer in plan.shape.layers if layer.attention_variant == "csa"]
    scopes = (Q, OUT) if scope is None else (Q, scope, OUT)
    rows = build_dsv4_replay_rows(plan, [saved_records(plan, [dict(scopes=dict.fromkeys(csa_ids, scopes))])])
    for row in rows:
        assert row["index_scores_executed"] == (scores if row["attention_variant"] == "csa" else False)
        assert row["indexer_short_context"] == (short if row["attention_variant"] == "csa" else False)


@pytest.mark.parametrize("change,error", [
    ("policy", "engine_policy=graph_replay"), ("engine_policy", "engine_policy=graph_replay"),
    ("mode", "cudagraph_mode"), ("layer", "engine layers do not match"), ("tp", "target TP does not match"),
    ("groups", "kv_cache_groups mismatch"), ("step_measurement", "measurement=KERNEL_ONLY"),
    ("call_measurement", "measurement=KERNEL_ONLY"), ("missing", "layer 0.*missing layer call"),
    ("non_family", "layer 0.*non-family scope"), ("warmup", "layer 0.*warmup-only scope"),
    ("host_bound", "case case, step 0, layer 0.*host_bound"),
    ("no_kernel", "case case, step 0, layer 0.*no_owned_kernel"),
    ("no_dependency", "case case, step 0, layer 0.*no_dependency_end"),
    ("non_positive", "case case, step 0, layer 0.*non_positive_target"),
])
def test_invalid_record_names_source_and_measurement_coordinates(plan, change, error):
    record = saved_records(plan)
    if change == "policy":
        plan = replace(plan, limits=replace(plan.limits, engine_policy="eager_chunked", cudagraph_mode=None, cudagraph_capture_sizes=()))
    elif change in ("engine_policy", "mode", "layer", "tp", "groups"):
        engine = dict(record.engine)
        if change == "engine_policy": engine["engine_policy"] = "eager_chunked"
        if change == "mode": engine["cudagraph_capture"] = {"cudagraph_mode": "NONE"}
        if change == "layer": engine["layers"] = [dict(row, compress_ratio=0) if row["global_layer_id"] == 0 else row for row in engine["layers"]]
        if change == "tp": engine["target_tensor_parallel_size"] = 4
        if change == "groups": engine.pop("kv_cache_groups")
        record = replace(record, engine=engine)
    elif change in ("step_measurement", "call_measurement"):
        name = "steps" if change == "step_measurement" else "calls"
        rows = getattr(record, name)
        record = replace(record, **{name: (dict(rows[0], measurement="CUDA_EVENT"), *rows[1:])})
    elif change == "missing":
        record = replace(record, calls=tuple(r for r in record.calls if r["global_layer_id"] != 0),
                         intervals=tuple(r for r in record.intervals if r["global_layer_id"] != 0))
    elif change in ("non_family", "warmup"):
        scope = "not_attention" if change == "non_family" else "dsv4_query_padding"
        record = replace(record, calls=(dict(record.calls[0], scope=scope), *record.calls[1:]))
    elif change == "host_bound":
        record = replace(record, intervals=tuple(dict(r, launch_end_us=r["start_us"]) if r["global_layer_id"] == 0 else r for r in record.intervals))
    elif change == "no_kernel":
        record = replace(record, calls=tuple(dict(r, kernel_count=0) if r["global_layer_id"] == 0 else r for r in record.calls),
                         intervals=tuple(r for r in record.intervals if r["global_layer_id"] != 0))
    elif change == "no_dependency":
        record = replace(record, intervals=tuple(dict(r, graph_gap_us=None) if r["global_layer_id"] == 0 and r["scope"] == Q else r for r in record.intervals if r["global_layer_id"] is not None))
    elif change == "non_positive":
        rows = list(record.intervals)
        rows[1] = dict(rows[1], graph_gap_us=-1000.0)
        record = replace(record, intervals=tuple(rows))
    with pytest.raises(ValueError, match=error) as caught:
        build_dsv4_replay_rows(plan, [record])
    assert record.source in str(caught.value)


def test_unique_engine_indices_and_routing_identity_required(plan):
    record = saved_records(plan)
    with pytest.raises(ValueError, match="second.jsonl.*engine index 0 recorded twice"):
        build_dsv4_replay_rows(plan, [record, replace(record, source="second.jsonl")])
    second = saved_records(plan, engine_index=1, source="second.jsonl")
    second = replace(second, engine=dict(second.engine, moe_routing={"runtime_path": "other"}))
    with pytest.raises(ValueError, match="second.jsonl.*routing identity"):
        build_dsv4_replay_rows(plan, [record, second])


def test_mixed_kernel_counts_in_same_key_are_rejected(plan):
    record = saved_records(plan, [{}, {}])
    extra = dict(next(r for r in record.intervals if r["global_layer_id"] == 0 and r["scope"] == Q and r["step_index"] == 1),
                 start_us=50.0, end_us=51.0, graph_gap_us=0.0)
    record = replace(record, intervals=(*record.intervals, extra),
                     calls=tuple(dict(r, kernel_count=3) if r["global_layer_id"] == 0 and r["scope"] == Q and r["step_index"] == 1 else r for r in record.calls))
    with pytest.raises(ValueError, match="synthetic.jsonl.*different kernel counts"):
        build_dsv4_replay_rows(plan, [record])


def test_aggregate_writes_replay_filename_schema_and_graph_identity(plan, tmp_path):
    record = saved_records(plan)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dsv4_raw_e0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (record.engine, *record.steps, *record.calls, *record.intervals)))
    main.main(["aggregate", *PLAN_ARGS, *bridge.graph_mode_flags(MODE, (1, 2, 4, 8)),
               "--raw_dir", str(raw), "--output_root", str(tmp_path / "output"), "--device", "h200",
               "--source_revision", "source", "--model_config_revision", "config", "--allow_partial_engines"])
    path, = (tmp_path / "output").rglob("*.csv")
    assert path.name == "attention_graph_replay_kernel_only.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {r["schema_version"] for r in rows} == {"dsv4_attention_replay.v1"}
    assert {r["graph_mode"] for r in rows} == {MODE}
    assert {r["measurement_type"] for r in rows} == {KERNEL_ONLY}
    summary = json.loads((raw / "dsv4_aggregate_summary.json").read_text())
    assert summary["outputs"][KERNEL_ONLY] == {"path": str(path), "rows": 5}
    assert summary["missing_engines"] == [1]
    assert summary["identity"]["parallel_groups"] == {"target_tensor_parallel": 8, "engine_tensor_parallel": 1}
    with pytest.raises(ValueError, match="Raw records lack engines \\[1\\]"):
        main.main(["aggregate", *PLAN_ARGS, *bridge.graph_mode_flags(MODE, (1, 2, 4, 8)),
                   "--raw_dir", str(raw), "--output_root", str(tmp_path / "output"), "--device", "h200",
                   "--source_revision", "source", "--model_config_revision", "config"])
