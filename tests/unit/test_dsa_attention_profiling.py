"""CPU tests for the DSA attention profiling plan, trace aggregation, and rows."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from frontier.profiling.attention import dsa_vllm_runner as runner
from frontier.profiling.attention.dsa_kernel_trace import (
    ScopeKey,
    aggregate_kernel_trace,
    annotation_name,
    parse_annotation_name,
)
from frontier.profiling.attention.dsa_main import main as dsa_main
from frontier.profiling.attention.dsa_plan import (
    INDEXER_SCOPE,
    MLA_SCOPE,
    PHASE_DECODE,
    PHASE_PREFILL,
    DsaMeasurementLimits,
    build_context_grid,
    build_dsa_profile_plan,
    build_reduced_hf_overrides,
    load_hf_config,
)
from frontier.profiling.attention.dsa_rows import (
    CUDA_EVENT,
    KERNEL_ONLY,
    build_dsa_rows,
)
from frontier.profiling.common.runtime_identity import (
    IDENTITY_COLUMNS,
    RuntimeIdentity,
    attach_identity_columns,
)

MODEL_NAME = "zai-org/GLM-5.3"
CONFIG_PATH = "data/config/models/zai-org__GLM-5.3.json"
PLAN_ARGS = [
    "--model_name", MODEL_NAME,
    "--model_config_path", CONFIG_PATH,
    "--target_tensor_parallel_size", "8",
    "--block_size", "64",
    "--max_context_len", "4160",
    "--max_kv_tokens_per_lane", "20000",
    "--max_num_seqs", "8",
    "--chunk_budgets", "1024", "256",
]


def _limits(**overrides):
    values = dict(
        max_context_len=4160,
        max_kv_tokens_per_lane=20000,
        max_num_seqs=8,
        chunk_budgets=(1024, 256),
        block_size=64,
        decode_warmup_steps=2,
        decode_measured_steps=3,
        prefill_warmup_repeats=1,
        prefill_measured_repeats=2,
    )
    values.update(overrides)
    return DsaMeasurementLimits(**values)


@pytest.fixture(scope="module")
def hf_config():
    return load_hf_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def plan(hf_config):
    return build_dsa_profile_plan(
        model_name=MODEL_NAME,
        hf_config=hf_config,
        target_tensor_parallel_size=8,
        num_layers=8,
        limits=_limits(),
    )


def test_reduced_model_keeps_native_index_reuse(plan):
    variants = [(l.global_layer_id, l.attention_variant) for l in plan.shape.layers]
    assert variants == [
        (0, "full_index"), (1, "full_index"), (2, "full_index"), (3, "shared_index"),
        (4, "shared_index"), (5, "shared_index"), (6, "full_index"), (7, "shared_index"),
    ]
    assert [l.index_producer_layer for l in plan.shape.layers] == [0, 1, 2, 2, 2, 2, 6, 6]
    assert plan.shape.heads_per_rank == 8
    assert plan.hf_overrides["num_hidden_layers"] == 8
    assert plan.hf_overrides["num_attention_heads"] == 8
    assert plan.hf_overrides["first_k_dense_replace"] == 8
    assert plan.hf_overrides["mlp_layer_types"] == ["dense"] * 8
    assert plan.hf_overrides["indexer_types"] == [
        "full", "full", "full", "shared", "shared", "shared", "full", "shared",
    ]


def test_reduced_model_needs_both_index_variants(hf_config):
    with pytest.raises(ValueError, match="full-index and one shared-index"):
        build_dsa_profile_plan(
            model_name=MODEL_NAME, hf_config=hf_config,
            target_tensor_parallel_size=8, num_layers=3, limits=_limits(),
        )


def test_overrides_reject_disagreeing_native_indexer_types(hf_config, plan):
    config = dict(hf_config)
    config["indexer_types"] = ["shared"] * hf_config["num_hidden_layers"]
    with pytest.raises(ValueError, match="disagree with the Frontier index topology"):
        build_reduced_hf_overrides(config, plan.shape.layers, 8)
    config = dict(hf_config)
    config["unknown_layer_types"] = ["x"] * hf_config["num_hidden_layers"]
    with pytest.raises(ValueError, match="unknown_layer_types"):
        build_reduced_hf_overrides(config, plan.shape.layers, 8)


def test_context_grid_has_topk_boundary_and_maximum():
    grid = build_context_grid(64, 2048, 4160)
    assert grid == (64, 128, 256, 512, 1024, 2047, 2048, 2049, 4096, 4160)


def test_cases_respect_target_limits(plan):
    limits = plan.limits
    engines = {engine.engine_index: engine for engine in plan.engines}
    assert [engine.chunk_budget for engine in plan.engines] == [256, 1024]
    decode_engine = max(plan.engines, key=lambda engine: engine.chunk_budget)
    decode = [case for case in plan.cases if case.phase == PHASE_DECODE]
    assert decode and {case.engine_index for case in decode} == {decode_engine.engine_index}
    for case in decode:
        assert case.batch_size <= limits.max_num_seqs
        assert case.prompt_len + case.max_tokens <= limits.max_context_len
        assert case.batch_size * (case.prompt_len + case.max_tokens) <= limits.max_kv_tokens_per_lane
    prefill = {case.case_id: case for case in plan.cases if case.phase == PHASE_PREFILL}
    for engine in plan.engines:
        long_case = prefill[f"e{engine.engine_index}-prefill-p4159"]
        assert long_case.max_tokens == 1
    # A prompt shorter than a budget is one chunk and runs once, on the smallest
    # engine whose budget exceeds it.
    assert engines[prefill["e0-prefill-p128"].engine_index].chunk_budget == 256
    assert engines[prefill["e1-prefill-p512"].engine_index].chunk_budget == 1024
    assert not any(case_id.endswith("-p1024") for case_id in prefill)


def test_plan_covers_short_extend_and_multi_sequence_prefill(plan):
    from frontier.attention.dsa_features import DSA_DECODE_QUERY_LIMIT
    from frontier.profiling.attention.dsa_rows import StepRequest, batch_load_columns

    prefills = [case for case in plan.cases if case.phase == PHASE_PREFILL]
    assert {2, 4, 8, 16, 32, 64, 128, 256} <= {case.prompt_len for case in prefills}
    multi = [case for case in prefills if case.batch_size > 1]
    assert multi
    assert all(case.prompt_len > DSA_DECODE_QUERY_LIMIT for case in multi)
    load, context = batch_load_columns([StepRequest(512, 0, 512), StepRequest(512, 0, 512)])
    assert load["num_prefill_seqs"] == 2
    assert json.loads(load["prefill_seq_lens"]) == [512, 512]
    assert context == {"prefill_chunk_size": 1024, "kv_cache_size": 0}


def test_engine_max_num_seqs_fits_chunk_budget_at_target_limits(hf_config):
    # vLLM rejects an engine whose max_num_batched_tokens is below
    # max_num_seqs; the target limits allow 1024 sequences with 256-token chunks.
    plan = build_dsa_profile_plan(
        model_name=MODEL_NAME,
        hf_config=hf_config,
        target_tensor_parallel_size=8,
        num_layers=8,
        limits=_limits(
            max_context_len=33024,
            max_kv_tokens_per_lane=205632,
            max_num_seqs=1024,
            chunk_budgets=(256, 512, 1024, 2048, 4096, 8192),
        ),
    )
    assert len(plan.engines) == 6
    for engine in plan.engines:
        cases = plan.cases_for_engine(engine.engine_index)
        assert engine.max_num_seqs == max(case.batch_size for case in cases)
        assert engine.max_num_seqs <= engine.chunk_budget
    assert plan.engines[-1].max_num_seqs == 1024


def test_limits_reject_decode_engine_smaller_than_batch():
    with pytest.raises(ValueError, match="max_num_seqs"):
        _limits(chunk_budgets=(4,), max_num_seqs=8)


def test_plan_dry_run_cli(tmp_path, capsys):
    output = tmp_path / "plan.json"
    dsa_main(["plan", *PLAN_ARGS, "--output", str(output)])
    summary = json.loads(capsys.readouterr().out.split("[dsa]")[0])
    full = json.loads(output.read_text())
    assert summary["num_cases"] == len(full["cases"])
    assert summary["num_engines"] == 2


def _trace(events):
    return [dict(event, pid=1, tid=7) if event["cat"] != "kernel" else event for event in events]


def test_kernel_trace_sums_correlated_kernels():
    name_a = annotation_name(3, 0, INDEXER_SCOPE)
    name_b = annotation_name(3, 0, MLA_SCOPE)
    assert parse_annotation_name(name_a) == ScopeKey(3, 0, INDEXER_SCOPE)
    events = _trace([
        {"cat": "user_annotation", "name": name_a, "ts": 100, "dur": 50},
        {"cat": "user_annotation", "name": name_b, "ts": 200, "dur": 50},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 110, "dur": 2, "args": {"correlation": 1}},
        {"cat": "cuda_driver", "name": "cuLaunchKernelEx", "ts": 120, "dur": 2, "args": {"correlation": 2}},
        {"cat": "cuda_runtime", "name": "cudaMemcpyAsync", "ts": 130, "dur": 2, "args": {"correlation": 3}},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 210, "dur": 2, "args": {"correlation": 4}},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 300, "dur": 2, "args": {"correlation": 5}},
        {"cat": "kernel", "name": "k1", "ts": 400, "dur": 10, "args": {"correlation": 1}},
        {"cat": "kernel", "name": "k2", "ts": 410, "dur": 5, "args": {"correlation": 2}},
        {"cat": "gpu_memcpy", "name": "m", "ts": 415, "dur": 7, "args": {"correlation": 3}},
        {"cat": "kernel", "name": "k4", "ts": 420, "dur": 20, "args": {"correlation": 4}},
        {"cat": "kernel", "name": "k5", "ts": 440, "dur": 99, "args": {"correlation": 5}},
    ])
    result = aggregate_kernel_trace(events)
    assert result[ScopeKey(3, 0, INDEXER_SCOPE)].kernel_time_ms == pytest.approx(0.015)
    assert result[ScopeKey(3, 0, INDEXER_SCOPE)].kernel_count == 2
    assert result[ScopeKey(3, 0, MLA_SCOPE)].kernel_time_ms == pytest.approx(0.020)


def test_kernel_trace_rejects_empty_and_overlapping_scopes():
    empty = _trace([{"cat": "user_annotation", "name": annotation_name(1, 0, MLA_SCOPE), "ts": 0, "dur": 5}])
    with pytest.raises(ValueError, match="launched no kernel"):
        aggregate_kernel_trace(empty)
    overlap = _trace([
        {"cat": "user_annotation", "name": annotation_name(1, 0, INDEXER_SCOPE), "ts": 0, "dur": 10},
        {"cat": "user_annotation", "name": annotation_name(1, 0, MLA_SCOPE), "ts": 5, "dur": 10},
    ])
    with pytest.raises(ValueError, match="overlap"):
        aggregate_kernel_trace(overlap)


def _engine_record(engine_index=1, chunk_budget=1024):
    return {
        "kind": "engine",
        "engine_index": engine_index,
        "chunk_budget": chunk_budget,
        "attention_backend": "FLASH_ATTN_MLA_SPARSE",
        "attention_impl": "FlashAttnMLASparseImpl",
        "kv_cache_dtype": "auto",
        "runtime_versions": {"vllm": "0.28.0", "torch": "2.9.0"},
        "device_name": "NVIDIA H800",
        "device_capability": "sm90",
        "layers": [],
    }


def _step_records(plan, measurement, step_index, requests, case_id, phase, time_ms, kernels=7):
    metadata = {"num_actual_tokens": sum(r["num_scheduled_tokens"] for r in requests),
                "num_decodes": 0, "num_prefills": 1, "num_decode_tokens": 0}
    records = [{
        "kind": "step", "measurement": measurement, "engine_index": 1,
        "case_id": case_id, "phase": phase, "step_index": step_index,
        "requests": requests, "mla_metadata": metadata,
        "indexer_metadata": {"num_prefill_tokens": metadata["num_actual_tokens"]},
    }]
    for layer in plan.shape.layers:
        scopes = [INDEXER_SCOPE, MLA_SCOPE] if layer.indexer_executed else [MLA_SCOPE]
        for scope in scopes:
            records.append({
                "kind": "call", "measurement": measurement, "engine_index": 1,
                "case_id": case_id, "step_index": step_index,
                "global_layer_id": layer.global_layer_id, "scope": scope,
                "time_ms": time_ms + layer.global_layer_id,
                "kernel_count": kernels if measurement == KERNEL_ONLY else None,
            })
    return records


def _prefill_request(q, ctx, prompt=4159):
    return [{"num_scheduled_tokens": q, "num_computed_tokens": ctx, "prompt_len": prompt}]


def test_rows_group_layers_and_repeats_per_variant(plan):
    records = [_engine_record()]
    for step_index in (0, 1):
        records += _step_records(plan, CUDA_EVENT, step_index, _prefill_request(1024, 2048),
                                 "e1-prefill-p4159", PHASE_PREFILL, 1.0)
    records += _step_records(plan, KERNEL_ONLY, 0, _prefill_request(1024, 2048),
                             "e1-prefill-p4159", PHASE_PREFILL, 0.5)
    rows = build_dsa_rows(plan, records)
    assert len(rows[CUDA_EVENT]) == 2 and len(rows[KERNEL_ONLY]) == 2
    full, shared = rows[CUDA_EVENT]
    assert full["attention_variant"] == "full_index"
    assert full["global_layer_ids"] == "[0, 1, 2, 6]"
    assert full[f"time_stats.{INDEXER_SCOPE}.count"] == 8
    assert full[f"time_stats.{MLA_SCOPE}.count"] == 8
    assert full["prefill_chunk_size"] == 1024 and full["kv_cache_size"] == 2048
    assert full["max_selected_key_len"] == plan.shape.index_topk
    assert shared["attention_variant"] == "shared_index"
    assert shared["index_producer_layers"] == "[2, 6]"
    assert f"time_stats.{INDEXER_SCOPE}.count" not in shared
    assert rows[KERNEL_ONLY][0][f"time_stats.{MLA_SCOPE}.kernel_count"] == 7
    assert rows[KERNEL_ONLY][0][f"time_stats.{INDEXER_SCOPE}.kernel_count"] == 7
    assert not any(name.startswith("kernel_count.") for name in rows[KERNEL_ONLY][0])


def test_rows_reject_missing_scope_and_kernel_count_drift(plan):
    records = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, _prefill_request(1024, 0), "c", PHASE_PREFILL, 1.0)
    dropped = [r for r in records if not (r["kind"] == "call" and r["global_layer_id"] == 6
                                          and r["scope"] == INDEXER_SCOPE)]
    with pytest.raises(ValueError, match="layer 6"):
        build_dsa_rows(plan, dropped)
    drift = [_engine_record()]
    drift += _step_records(plan, KERNEL_ONLY, 0, _prefill_request(1024, 0), "c", PHASE_PREFILL, 1.0, kernels=7)
    drift += _step_records(plan, KERNEL_ONLY, 1, _prefill_request(1024, 0), "c", PHASE_PREFILL, 1.0, kernels=8)
    with pytest.raises(ValueError, match="different kernel counts"):
        build_dsa_rows(plan, drift)


def test_decode_rows_record_context_spread(plan):
    requests = [
        {"num_scheduled_tokens": 1, "num_computed_tokens": ctx, "prompt_len": 512}
        for ctx in (514, 520, 530)
    ]
    records = [_engine_record()] + _step_records(
        plan, CUDA_EVENT, 0, requests, "e1-decode-b3-p512", PHASE_DECODE, 0.2)
    row = build_dsa_rows(plan, records)[CUDA_EVENT][0]
    assert row["batch_size"] == 3 and row["num_decode_seqs"] == 3
    assert row["decode_kv_cache_sizes"] == "[514, 520, 530]"
    assert row["decode_kv_cache_size_min"] == 514 and row["decode_kv_cache_size_max"] == 530
    assert row["prefill_chunk_size"] == 0 and row["is_prefill"] is False


def test_aggregate_cli_writes_canonical_tables(plan, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    records = [_engine_record()]
    records += _step_records(plan, CUDA_EVENT, 0, _prefill_request(1024, 0), "e1-prefill-p4159", PHASE_PREFILL, 1.0)
    records += _step_records(plan, KERNEL_ONLY, 0, _prefill_request(1024, 0), "e1-prefill-p4159", PHASE_PREFILL, 0.5)
    (raw_dir / "dsa_raw_e1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    common = ["--raw_dir", str(raw_dir), "--output_root", str(tmp_path / "out"), "--device", "h800",
              "--source_revision", "abc", "--model_config_revision", "rev"]
    with pytest.raises(ValueError, match="lack engines"):
        dsa_main(["aggregate", *PLAN_ARGS, *common])
    dsa_main(["aggregate", *PLAN_ARGS, *common, "--allow_partial_engines"])
    base = tmp_path / "out" / "compute" / "h800" / "zai-org" / "GLM-5.3"
    for name, measurement in (("attention.csv", CUDA_EVENT), ("attention_kernel_only.csv", KERNEL_ONLY)):
        with open(base / name, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert {row["measurement_type"] for row in rows} == {measurement}
        assert set(IDENTITY_COLUMNS) <= set(rows[0])
        assert rows[0]["weight_source"] == "dummy" and rows[0]["graph_mode"] == "eager"
        assert rows[0]["quant_signature"].startswith("method=fp8|act=dynamic")
        assert json.loads(rows[0]["parallel_groups"]) == {
            "engine_tensor_parallel": 1, "target_tensor_parallel": 8}


def _identity(**overrides):
    values = dict(
        runtime_versions={"vllm": "0.28.0"}, source_revision="abc",
        model_config_revision="rev", model_config_sha256="sha", device_name="H800",
        device_capability="sm90", schema_version="v1", quant_signature="none",
        parallel_groups={"tp": 8}, graph_mode="eager",
    )
    values.update(overrides)
    return RuntimeIdentity(**values)


def test_runtime_identity_columns_and_validation():
    identity = _identity()
    columns = identity.columns()
    assert tuple(columns) == IDENTITY_COLUMNS
    assert identity.fingerprint == _identity(model_config_revision="other").fingerprint
    assert identity.fingerprint != _identity(runtime_versions={"vllm": "0.10.1"}).fingerprint
    with pytest.raises(ValueError, match="graph_mode"):
        _identity(graph_mode=" ")
    with pytest.raises(ValueError, match="positive int"):
        _identity(parallel_groups={"tp": 0})
    assert attach_identity_columns([{"a": 1}], identity)[0]["weight_source"] == "dummy"
    with pytest.raises(ValueError, match="quant_signature"):
        attach_identity_columns([{"quant_signature": "other"}], identity)


class _FakeEvent:
    clock = 0.0

    def __init__(self, enable_timing):
        assert enable_timing
        self.time = None

    def record(self):
        _FakeEvent.clock += 1.5
        self.time = _FakeEvent.clock

    def elapsed_time(self, other):
        return other.time - self.time


def _fake_torch():
    return SimpleNamespace(cuda=SimpleNamespace(Event=_FakeEvent, synchronize=lambda: None))


def _scheduler_output(new=(), cached=(), finished=()):
    return SimpleNamespace(
        finished_req_ids=set(finished),
        num_scheduled_tokens={rid: q for rid, q, _, _ in (*new, *cached)},
        scheduled_new_reqs=[
            SimpleNamespace(req_id=rid, prompt_token_ids=[0] * prompt, num_computed_tokens=ctx)
            for rid, _, ctx, prompt in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[rid for rid, _, _, _ in cached],
            num_computed_tokens=[ctx for _, _, ctx, _ in cached],
        ),
    )


def test_worker_state_records_only_the_case_window():
    state = runner._WorkerState(engine_index=1, torch_module=_fake_torch())
    state.case = runner._CaseRecorder("c", PHASE_DECODE, 2, skip=1, limit=1, measurement=CUDA_EVENT)
    # A mixed build step (one prefill, one decode) is never recorded.
    assert state.begin_step(_scheduler_output(new=[("a", 64, 0, 64)])) is None
    assert state.begin_step(_scheduler_output(new=[("b", 64, 0, 64)], cached=[("a", 1, 64, 64)])) is None
    decode = _scheduler_output(cached=[("a", 1, 65, 64), ("b", 1, 64, 64)])
    assert state.begin_step(decode) is None  # warmup match
    active = state.begin_step(decode)
    assert active is not None and active.record["requests"][0]["prompt_len"] == 64
    state.active = active
    assert state.timed_call(0, MLA_SCOPE, lambda x: x + 1, (1,), {}) == 2
    state.active = None
    active.record["mla_metadata"] = {"num_decodes": 2}
    active.record["indexer_metadata"] = {}
    state.end_step(active)
    assert state.case.calls[0]["time_ms"] == pytest.approx(1.5)
    assert state.begin_step(decode) is None  # window complete
    state.begin_step(_scheduler_output(finished=["a", "b"]))
    assert state.prompt_lens == {}


def test_end_step_requires_captured_metadata():
    state = runner._WorkerState(engine_index=1, torch_module=_fake_torch())
    state.case = runner._CaseRecorder("c", PHASE_PREFILL, 1, skip=0, limit=1, measurement=CUDA_EVENT)
    active = state.begin_step(_scheduler_output(new=[("a", 256, 0, 512)]))
    with pytest.raises(RuntimeError, match="metadata was not captured"):
        state.end_step(active)


def test_worker_records_all_sequences_in_a_multi_prefill_case():
    state = runner._WorkerState(engine_index=1, torch_module=_fake_torch())
    state.case = runner._CaseRecorder("c", PHASE_PREFILL, 2, skip=0, limit=1, measurement=CUDA_EVENT)
    active = state.begin_step(_scheduler_output(new=[("a", 512, 0, 512), ("b", 512, 0, 512)]))
    assert active is not None
    assert len(active.record["requests"]) == 2


def test_prefill_window_counts_chunks_per_prompt(plan):
    case = next(c for c in plan.cases if c.case_id == "e0-prefill-p4159")
    assert runner._case_window(case, plan.engine(case.engine_index)) == (17, 34)
