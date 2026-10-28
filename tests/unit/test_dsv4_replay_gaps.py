"""CPU replay-gap rules, identity admission, diagnostics and CSV contracts."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.execution_time_predictor.kernel_gap import read_dsv4_kernel_gap_ms, read_kernel_gap_ms
from frontier.profiling.kernel_gap import dsv4_replay_gaps as replay
from frontier.profiling.kernel_gap.schema import DSV4_EXECUTION_MODES

Q = "dsv4_q_projection"
SPARSE = "dsv4_sparse_attention"
OUT = "dsv4_output_projection"


def interval(start, end, *, scope=None, layer=0, stream=7, launch="eager", activity="kernel",
             gap=None, launch_end=0.0, capture_kind=None, step=0, case="case", name=None):
    return {"kind": "interval", "engine_index": 0, "case_id": case, "step_index": step,
            "global_layer_id": layer if scope is not None else None, "scope": scope,
            "launch_kind": launch, "activity": activity, "name": name or scope or activity,
            "start_us": float(start), "end_us": float(end), "stream": stream,
            "graph_id": 1 if launch == "graph" else None, "graph_node_id": 10 if launch == "graph" else None,
            "original_node_id": 1 if launch == "graph" else None, "capture": 0 if launch == "graph" else None,
            "capture_kind": capture_kind, "correlation": 1, "launch_end_us": launch_end, "graph_gap_us": gap}


def records(mode, intervals, *, variant="uncompressed", call_costs=None, source=None):
    engine = {"kind": "engine", "engine_policy": "graph_replay", "engine_index": 0,
              "device_name": "NVIDIA H200", "driver_version": "driver", "target_tensor_parallel_size": 8,
              "runtime_versions": {"torch": "torch version", "torch_cuda": "CUDA version"},
              "cudagraph_capture": {"cudagraph_mode": "FULL_AND_PIECEWISE" if mode == "FULL" else mode},
              "multi_stream_gemm_token_threshold": 1024, "step_hold_us": 0.0, "hold_cycles_per_us": None,
              "layers": [{"global_layer_id": 0, "attention_variant": variant, "compress_ratio": 1}]}
    step = {"kind": "step", "engine_index": 0, "step_index": 0, "case_id": "case",
            "cudagraph_dispatch": {"cg_mode": mode, "num_tokens": 1}}
    scopes = {row["scope"] for row in intervals if row["scope"] is not None}
    calls = tuple({"kind": "call", "engine_index": 0, "case_id": "case", "step_index": 0,
                   "global_layer_id": 0, "scope": scope,
                   "time_ms": (call_costs or {}).get(scope, sum(row["end_us"] - row["start_us"] for row in intervals if row["scope"] == scope and row["activity"] == "kernel") * 1e-3),
                   "kernel_count": sum(row["scope"] == scope and row["activity"] == "kernel" for row in intervals)}
                  for scope in sorted(scopes))
    return replay.ReplayRecords(source or mode, engine, (step,), calls, tuple(intervals))


def complete_records():
    full = records("FULL", [interval(0, 1), interval(2, 3, activity="memcpy"),
        interval(4, 6, scope=Q, launch="graph", capture_kind="graph"),
        interval(7, 9, scope=SPARSE, launch="graph", capture_kind="graph", gap=1),
        interval(10, 12, scope=OUT, launch="graph", capture_kind="graph", gap=1)])
    piecewise = records("PIECEWISE", [interval(0, 1), interval(2, 3, activity="memset"),
        interval(4, 6, scope=Q, launch="graph", capture_kind="breakable_segment"),
        interval(7, 9, scope=SPARSE),
        interval(10, 12, scope=OUT, launch="graph", capture_kind="breakable_segment")])
    eager = records("NONE", [interval(0, 1), interval(2, 4, scope=Q),
                              interval(5, 7, scope=SPARSE), interval(8, 10, scope=OUT)])
    return (full, piecewise, eager)


def write_records(tmp_path, saved):
    paths = []
    for index, record in enumerate(saved):
        path = tmp_path / f"raw{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in (record.engine, *record.steps, *record.calls, *record.intervals)))
        paths.append(path)
    return paths


def test_four_classes_roots_and_owned_subset_have_host_ahead_means():
    reduction = replay.reduce_dsv4_gaps(complete_records())
    assert reduction.reasons == ()
    assert reduction.values_us == dict.fromkeys(DSV4_EXECUTION_MODES, 1.0)
    diagnostics = reduction.diagnostics
    assert diagnostics["classes"]["cuda_graph"]["roots"]["host_ahead_count"] == 1
    assert diagnostics["classes"]["cuda_graph_piecewise"]["roots"]["host_ahead_count"] == 2
    assert diagnostics["classes"]["eager_preparation"]["sample_count"] == 4
    assert diagnostics["classes"]["eager_preparation"]["exclusion_counts"] == {"no_dependency_end": 2}
    assert diagnostics["classes_by_step_mode"]["eager"]["PIECEWISE"]["host_ahead_count"] == 1
    assert diagnostics["classes"]["cuda_graph"]["attention_owned"]["mean"] == 1.0
    assert diagnostics["none_unowned"]["exclusion_counts"] == {"no_dependency_end": 1}


def test_signed_pdl_graph_gap_host_wait_split_and_r1_root():
    saved = records("FULL", [interval(0, 2),
        interval(4, 10, scope=Q, launch="graph", capture_kind="graph", launch_end=1),
        interval(8, 13, scope=OUT, launch="graph", capture_kind="graph", gap=-2, launch_end=11)])
    samples = replay.activity_gaps([saved])
    root, pdl = samples[1:]
    assert (root.dependency_end_us, root.gap_us, root.root) == (2, 2, True)
    assert (pdl.dependency_end_us, pdl.gap_us, pdl.host_wait_us, pdl.device_gap_us) == (10, -2, 0, -2)
    assert pdl.host_ahead
    summary = replay.reduce_dsv4_gaps([saved]).diagnostics["classes"]["cuda_graph"]
    assert summary["negative_count"] == 1 and summary["mean"] == 0.0
    assert summary["min"] == -2 and summary["p90"] == pytest.approx(1.6)


@pytest.mark.parametrize("launch_end,host_wait,device_gap", [(1, 0, 3), (3, 1, 2), (20, 3, 0)])
def test_host_wait_is_clamped_to_positive_gap(launch_end, host_wait, device_gap):
    saved = records("NONE", [interval(0, 2), interval(5, 8, scope=Q, launch_end=launch_end)])
    sample = replay.activity_gaps([saved])[1]
    assert sample.gap_us == 3
    assert sample.host_wait_us == host_wait and sample.device_gap_us == device_gap
    summary = replay.reduce_dsv4_gaps([saved]).diagnostics["classes"]["eager"]
    assert summary["host_ahead_count"] == int(host_wait == 0)
    assert summary["host_bound_excluded_count"] == int(host_wait > 0)


def test_graph_root_r1_uses_all_streams_copies_and_strict_start_ties():
    saved = records("FULL", [interval(0, 5, activity="memcpy", stream=9),
        interval(3, 9, activity="memset", stream=11),
        interval(10, 12, scope=Q, launch="graph", capture_kind="graph"),
        interval(10, 99, scope=OUT, launch="graph", capture_kind="graph")])
    roots = [gap for gap in replay.activity_gaps([saved]) if gap.root]
    assert [gap.dependency_end_us for gap in roots] == [9, 9]


def test_eager_scope_predecessor_joins_other_stream_and_layer_entry():
    norm = "dsv4_fused_qkv_rmsnorm"
    compressor = "dsv4_compressor_save_partial_states"
    saved = records("NONE", [interval(0, 7, stream=30),
        interval(10, 12, scope=norm), interval(13, 16, scope=Q),
        interval(14, 25, scope=compressor, stream=21), interval(26, 29, scope=SPARSE)], variant="hca")
    gaps = {gap.interval["scope"]: gap for gap in replay.activity_gaps([saved]) if gap.attention_owned}
    assert gaps[norm].dependency_end_us == 7  # Entry comes from another stream.
    assert gaps[compressor].dependency_end_us == 12  # Fork from the main stream norm.
    assert gaps[SPARSE].dependency_end_us == 25  # Join beats the previous main-stream end (16).


def test_same_stream_topk_scores_nesting_follows_physical_order_without_scope_cycle():
    topk, scores = "dsv4_index_topk", "dsv4_index_scores"
    saved = records("NONE", [interval(0, 1), interval(2, 3, scope=Q),
        interval(4, 5, scope=topk), interval(6, 7, scope=scores), interval(8, 9, scope=topk)], variant="csa")
    gaps = replay.activity_gaps([saved])
    assert [gap.dependency_end_us for gap in gaps] == [None, 1, 3, 5, 7]


def test_piecewise_eager_break_follows_whole_preceding_graph_replay():
    topk, scores = "dsv4_index_topk", "dsv4_index_scores"
    store = "dsv4_index_compressor_fused_compress_store"
    saved = records("PIECEWISE", [interval(0, 1),
        interval(2, 4, scope=Q, launch="graph", capture_kind="breakable_segment"),
        interval(3, 9, scope=store, stream=368, launch="graph", capture_kind="breakable_segment", gap=1),
        interval(10, 11, scope=topk), interval(12, 13, scope=scores), interval(14, 15, scope=topk)], variant="csa")
    gaps = replay.activity_gaps([saved])
    # The topk scope encloses scores; its first kernel waits for the replay node on stream 368.
    assert [gap.dependency_end_us for gap in gaps[3:]] == [9, 11, 13]


def test_piecewise_eager_break_before_every_graph_activity_is_rejected():
    saved = records("PIECEWISE", [interval(0, 1, scope=SPARSE),
        interval(2, 4, scope=OUT, launch="graph", capture_kind="breakable_segment")])
    with pytest.raises(ValueError, match="attention-owned eager activity precedes every graph activity"):
        replay.activity_gaps([saved])


def test_none_unowned_r1_is_diagnostic_only_and_owned_copy_cost_reported():
    saved = records("NONE", [interval(0, 2), interval(3, 4, scope=Q, activity="memcpy"),
        interval(5, 6, scope=Q), interval(8, 9)])
    reduction = replay.reduce_dsv4_gaps([saved])
    assert reduction.values_us["eager"] == 1
    assert reduction.diagnostics["none_unowned"]["mean"] == 2
    assert reduction.diagnostics["owned_non_kernel_time_us_by_engine_mode"]["NONE"] == 1
    assert reduction.diagnostics["classes"]["eager"]["sample_count"] == 1
    assert reduction.diagnostics["classes"]["eager"]["exclusion_counts"] == {}


def test_eager_preparation_before_and_after_graph_prices_all_activities():
    saved = records("FULL", [interval(0, 1), interval(2, 3, activity="memcpy"),
        interval(4, 5, scope=Q, launch="graph", capture_kind="graph"),
        interval(6, 7, activity="memset"), interval(8, 9)])
    samples = [gap for gap in replay.activity_gaps([saved]) if gap.segment == "eager_preparation"]
    assert [gap.gap_us for gap in samples] == [None, 1, 1, 1]
    assert replay.reduce_dsv4_gaps([saved]).values_us["eager_preparation"] == 1


@pytest.mark.parametrize("mode,intervals,error", [
    ("NONE", [interval(1, 2, launch="graph", capture_kind="graph")], "NONE step has a graph"),
    ("FULL", [interval(1, 2)], "FULL step has no graph"),
    ("PIECEWISE", [interval(1, 2)], "PIECEWISE step has no graph"),
    ("FULL", [interval(1, 2, launch="graph", capture_kind="breakable_segment")], "capture kind"),
    ("PIECEWISE", [interval(1, 2, launch="graph", capture_kind="graph")], "capture kind"),
    ("FULL", [interval(1, 2, launch="graph", capture_kind="graph"), interval(3, 4),
              interval(5, 6, launch="graph", capture_kind="graph")], "unowned eager activity"),
])
def test_dispatch_capture_and_outside_activity_invariants(mode, intervals, error):
    with pytest.raises(ValueError, match=f"case case, step 0, activity .*{error}"):
        replay.activity_gaps([records(mode, intervals)])


@pytest.mark.parametrize("field,value", [("device_name", "different"), ("driver_version", "different"),
    ("target_tensor_parallel_size", 4), ("torch", "different"), ("torch_cuda", "different")])
def test_identity_difference_is_rejected(field, value):
    full, piecewise, _ = complete_records()
    engine = dict(piecewise.engine)
    if field in ("torch", "torch_cuda"):
        engine["runtime_versions"] = dict(engine["runtime_versions"], **{field: value})
    else:
        engine[field] = value
    with pytest.raises(ValueError, match=f"identity differs.*{field}"):
        replay.reduce_dsv4_gaps([full, replace(piecewise, engine=engine)])


def test_engine_policy_and_missing_identity_rejected():
    saved = complete_records()[0]
    with pytest.raises(ValueError, match="engine_policy=graph_replay"):
        replay.reduce_dsv4_gaps([replace(saved, engine=dict(saved.engine, engine_policy="eager_chunked"))])
    engine = {field: value for field, value in saved.engine.items() if field != "driver_version"}
    with pytest.raises(ValueError, match="no identity field 'driver_version'"):
        replay.reduce_dsv4_gaps([replace(saved, engine=engine)])


def _cli_args(tmp_path, paths):
    return ["--raw_files", *map(str, paths), "--output_dir", str(tmp_path / "output"),
            "--device", "h200", "--model", "DeepSeek-V4-Flash", "--diagnostics_path", str(tmp_path / "diagnostics.json")]


def test_cli_writes_complete_csv_at_canonical_path_and_reader_accepts(tmp_path):
    paths = write_records(tmp_path, complete_records())
    assert replay.main(_cli_args(tmp_path, paths)) == 0
    csv, = (tmp_path / "output").rglob("kernel_gap.csv")
    table = pd.read_csv(csv)
    assert tuple(table.columns) == replay.CSV_COLUMNS
    assert table["execution_mode"].tolist() == list(DSV4_EXECUTION_MODES)
    assert table["num_tensor_parallel_workers"].tolist() == [8] * 4
    assert table["chain"].tolist() == ["dsv4_graph_replay:DeepSeek-V4-Flash:tp8"] * 4
    assert table["gpu_name"].tolist() == ["NVIDIA H200"] * 4
    assert table["driver_version"].tolist() == ["driver"] * 4
    assert read_dsv4_kernel_gap_ms(str(csv), 8) == dict.fromkeys(DSV4_EXECUTION_MODES, 0.001)
    assert read_kernel_gap_ms(str(csv), 8) == {"eager": 0.001, "cuda_graph": 0.001}
    assert json.loads((tmp_path / "diagnostics.json").read_text())["reasons"] == []


@pytest.mark.parametrize("saved", [[complete_records()[2]], [records("FULL", [interval(0, 10),
    interval(1, 2, scope=Q, launch="graph", capture_kind="graph")])]])
def test_cli_without_complete_nonnegative_values_writes_diagnostics_only(tmp_path, saved):
    assert replay.main(_cli_args(tmp_path, write_records(tmp_path, saved))) == 1
    assert not list(tmp_path.rglob("kernel_gap.csv"))
    assert json.loads((tmp_path / "diagnostics.json").read_text())["reasons"]


def test_cli_invariant_failure_always_writes_diagnostics(tmp_path):
    saved = records("FULL", [interval(0, 1)])
    assert replay.main(_cli_args(tmp_path, write_records(tmp_path, [saved]))) == 1
    assert "no graph activity" in json.loads((tmp_path / "diagnostics.json").read_text())["reasons"][0]


@pytest.mark.parametrize("change", ["missing", "duplicated", "negative", "nan", "inf"])
@pytest.mark.parametrize("mode", DSV4_EXECUTION_MODES)
def test_v4_reader_rejects_each_invalid_row(tmp_path, mode, change):
    rows = [{"execution_mode": name, "num_tensor_parallel_workers": 8, "kernel_gap_us": 1} for name in DSV4_EXECUTION_MODES]
    index = list(DSV4_EXECUTION_MODES).index(mode)
    if change == "missing":
        rows.pop(index)
    elif change == "duplicated":
        rows.append(dict(rows[index]))
    else:
        rows[index]["kernel_gap_us"] = {"negative": -1, "nan": float("nan"), "inf": float("inf")}[change]
    path = tmp_path / "kernel_gap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"one finite, non-negative.*execution_mode={mode}.*=8"):
        read_dsv4_kernel_gap_ms(str(path), 8)


def test_readers_retain_tp_column_and_row_error_messages(tmp_path):
    path = tmp_path / "kernel_gap.csv"
    pd.DataFrame([{"execution_mode": "eager", "kernel_gap_us": 1}]).to_csv(path, index=False)
    for reader in (read_dsv4_kernel_gap_ms, read_kernel_gap_ms):
        with pytest.raises(ValueError, match="has no num_tensor_parallel_workers column, the TP size each row was measured at"):
            reader(str(path), 8)
    pd.DataFrame([{"execution_mode": name, "kernel_gap_us": 1, "num_tensor_parallel_workers": 1}
                  for name in DSV4_EXECUTION_MODES]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="execution_mode=eager and num_tensor_parallel_workers=8"):
        read_kernel_gap_ms(str(path), 8)


def test_graph_copies_are_dependency_ends_without_class_samples():
    saved = records("FULL", [interval(0, 1),
        interval(2, 3, scope=Q, launch="graph", capture_kind="graph"),
        interval(4, 8, activity="memcpy", launch="graph", capture_kind="graph"),
        interval(9, 11, scope=OUT, launch="graph", capture_kind="graph", gap=1)])
    samples = replay.activity_gaps([saved])
    assert samples[2].segment is None and samples[2].dependency_end_us is None
    assert samples[3].dependency_end_us == 8
    assert replay.reduce_dsv4_gaps([saved]).diagnostics["classes"]["cuda_graph"]["sample_count"] == 2


def test_none_outside_copies_take_r1_diagnostic_gaps():
    saved = records("NONE", [interval(0, 1), interval(2, 3, activity="memcpy"),
                              interval(4, 5, activity="memset")])
    summary = replay.reduce_dsv4_gaps([saved]).diagnostics["none_unowned"]
    assert summary["sample_count"] == 3 and summary["host_ahead_count"] == 2
    assert summary["mean"] == 1


def test_unowned_eager_activity_during_last_graph_interval_is_rejected():
    saved = records("FULL", [interval(0, 1),
        interval(2, 10, scope=Q, launch="graph", capture_kind="graph"), interval(5, 6)])
    with pytest.raises(ValueError, match="unowned eager activity"):
        replay.activity_gaps([saved])


def test_container_owned_copy_remains_dependency_only_without_a_scope_position():
    saved = records("NONE", [interval(0, 1),
        interval(2, 3, scope="dsv4_attention_layer", activity="memcpy"), interval(4, 5, scope=Q)])
    reduction = replay.reduce_dsv4_gaps([saved])
    assert reduction.diagnostics["classes"]["eager"]["sample_count"] == 1
    assert reduction.diagnostics["owned_non_kernel_time_us_by_engine_mode"]["NONE"] == 1
    assert reduction.values_us["eager"] == 1


def _held(saved):
    return replace(saved,engine=dict(saved.engine,step_hold_us=200000.0,hold_cycles_per_us=1000.0))


@pytest.mark.parametrize('mode,launch,capture_kind',[('NONE','eager',None),('FULL','graph','graph')])
def test_hold_supplies_r1_stream_and_layer_entry_without_gap_sample(mode,launch,capture_kind):
    saved=_held(records(mode,[interval(10,100,activity='hold',stream=99),
        interval(101,105,scope=Q,launch=launch,capture_kind=capture_kind,stream=7,launch_end=50)]))
    hold,kernel=replay.activity_gaps([saved])
    assert hold.segment is None and hold.exclusion_reason=='hold_dependency'
    assert kernel.dependency_end_us==100 and kernel.gap_us==1 and kernel.host_ahead
    diagnostics=replay.reduce_dsv4_gaps([saved]).diagnostics
    assert diagnostics['none_unowned']['sample_count']==0
    assert diagnostics['classes']['eager_preparation']['sample_count']==0


def test_hold_overlap_exclusion_slack_and_post_hold_launch_counts():
    saved=_held(records('NONE',[interval(10,100,activity='hold'),interval(20,25,activity='memcpy'),
        interval(101,105,scope=Q,launch_end=150),interval(110,114,scope=OUT,launch_end=50),
        interval(120,125,activity='memcpy',launch_end=130)]))
    samples=replay.activity_gaps([saved])
    overlap=next(sample for sample in samples if sample.interval['start_us']==20)
    assert overlap.exclusion_reason=='overlaps_hold' and not overlap.host_ahead
    diagnostic=replay.reduce_dsv4_gaps([saved]).diagnostics['hold_by_step_mode_and_case']['NONE']['case']
    assert diagnostic=={'step_count':1,'realized_hold_duration_us':{'min':90.0,'median':90.0,'max':90.0},
        'slack_us':{'min':-50.0,'median':-50.0},'attention_owned_post_hold_launch_count':1,'unowned_post_hold_launch_count':1}


@pytest.mark.parametrize('configured,count',[(0,1),(1,0),(1,2)])
def test_step_hold_count_invariant(configured,count):
    saved=records('NONE',[interval(i*10,i*10+5,activity='hold') for i in range(count)]+[interval(50,60,scope=Q)])
    if configured:saved=_held(saved)
    with pytest.raises(ValueError,match=f'requires {configured} hold intervals'):
        replay.activity_gaps([saved])


@pytest.mark.parametrize('field',['step_hold_us','hold_cycles_per_us'])
def test_engine_requires_hold_identity_fields(field):
    saved=complete_records()[0]
    engine=dict(saved.engine);engine.pop(field)
    with pytest.raises(ValueError,match='no hold field'):
        replay.reduce_dsv4_gaps([replace(saved,engine=engine)])


@pytest.mark.parametrize("owned", [False, True])
def test_hold_overlap_exclusion_precedes_eager_position_checks(owned):
    rows = [interval(0, 20, activity="hold"),
            interval(5, 6, scope=Q, launch="graph", capture_kind="breakable_segment"),
            interval(15, 16, scope=OUT, launch="graph", capture_kind="breakable_segment")]
    overlapping = interval(1, 2, scope=SPARSE) if owned else interval(8, 9)
    rows.append(overlapping)
    saved = records("PIECEWISE", rows)
    saved = replace(saved, engine=dict(saved.engine, step_hold_us=20.0, hold_cycles_per_us=1.0))
    samples = replay.activity_gaps([saved])
    sample = next(sample for sample in samples if sample.interval is overlapping)
    assert sample.exclusion_reason == "overlaps_hold"
    assert not sample.host_ahead
