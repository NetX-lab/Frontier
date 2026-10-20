"""Dense regular-serving context coverage, distinct from native DP dummy work."""

import json
import random
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from frontier.profiling.attention import dsa_vllm_runner as runner
from frontier.profiling.attention.dsa_main import main
from frontier.profiling.attention.dsa_plan import (
    INDEXER_SCOPE, MLA_SCOPE, PHASE_DECODE, PHASE_PREFILL,
    DsaMeasurementLimits, build_dsa_profile_plan, load_hf_config,
)
from frontier.profiling.attention.dsa_rows import CUDA_EVENT, KERNEL_ONLY, build_dsa_rows


def build_plan(max_context=None, *, min_context=0, prefill_prompt_len=None, **overrides):
    limits = dict(
        max_context_len=33024, max_kv_tokens_per_lane=205632,
        max_num_seqs=1024, chunk_budgets=(256, 512, 1024, 2048, 4096, 8192),
        block_size=64, decode_warmup_steps=2, decode_measured_steps=8,
        prefill_warmup_repeats=1, prefill_measured_repeats=3,
    )
    limits.update(overrides)
    return build_dsa_profile_plan(
        model_name="zai-org/GLM-5.3",
        hf_config=load_hf_config("data/config/models/zai-org__GLM-5.3.json"),
        target_tensor_parallel_size=8, num_layers=8,
        limits=DsaMeasurementLimits(**limits), query_one_context_max=max_context,
        query_one_context_min=min_context, prefill_prompt_len=prefill_prompt_len,
    )


def _native_plan(*, maximum=7166, minimum=6143, prefill=6144, **overrides):
    values = dict(
        engine_policy="native_defaults", max_context_len=7168,
        max_kv_tokens_per_lane=7168, max_num_seqs=128, chunk_budgets=(8192,),
    )
    values.update(overrides)
    return build_plan(
        maximum, min_context=minimum, prefill_prompt_len=prefill, **values
    )


def test_query_one_plan_keeps_zero_prefill_and_dense_decode_separate():
    plan = build_plan(65)
    assert len(plan.engines) == 1
    assert plan.engines[0].max_num_seqs == 1
    assert plan.engines[0].chunk_budget == 256
    zero, small = plan.cases
    original_zero = next(case for case in build_plan().cases if case.case_id == zero.case_id)
    assert asdict(zero) == asdict(original_zero)
    assert zero.phase == PHASE_PREFILL and zero.prompt_len == zero.max_tokens == 1
    assert runner._case_window(zero, plan.engines[0]) == (1, 3)
    assert small.phase == PHASE_DECODE and small.prompt_len == 1
    assert small.max_tokens == 66
    assert runner._case_window(small, plan.engines[0]) == (0, 65)
    assert sum(case.estimated_steps for case in plan.cases) == 70
    assert {layer.attention_variant for layer in plan.shape.layers} == {"full_index", "shared_index"}


def test_existing_full_plan_is_unchanged():
    plan = build_plan()
    assert len(plan.engines) == 8 and len(plan.cases) == 280
    assert sum(case.estimated_steps for case in plan.cases) == 5996
    assert not any(case.case_id.startswith("e0-decode-q1-") for case in plan.cases)


@pytest.mark.parametrize("maximum", [-1, True, 1.5, "65"])
def test_query_one_plan_rejects_invalid_context_limit(maximum):
    with pytest.raises(ValueError, match="query_one_context_max"):
        build_plan(maximum)


@pytest.mark.parametrize("limit", ["max_context_len", "max_kv_tokens_per_lane"])
def test_query_one_request_fits_native_capacity(limit):
    with pytest.raises(ValueError, match="capacity for 33025"):
        build_plan(33023, **{limit: 33024})


def test_exact_zero_only_plan_is_regular_prefill():
    plan = build_plan(0)
    assert [case.case_id for case in plan.cases] == ["e0-prefill-p1"]
    assert plan.summary()["estimated_steps"] == 4


@pytest.mark.parametrize("maximum", range(66))
def test_each_small_context_endpoint_is_in_the_recorded_window(maximum):
    plan = build_plan(maximum)
    assert len(plan.cases) == (2 if maximum else 1)
    if maximum:
        case = plan.cases[1]
        skip, count = runner._case_window(case, plan.engines[0])
        assert range(case.prompt_len + skip, case.prompt_len + skip + count) == range(1, maximum + 1)
        assert case.prompt_len + case.max_tokens == maximum + 2


def test_query_one_cli_dry_run(tmp_path, capsys):
    output = tmp_path / "plan.json"
    main([
        "plan", "--model_name", "zai-org/GLM-5.3",
        "--model_config_path", "data/config/models/zai-org__GLM-5.3.json",
        "--target_tensor_parallel_size", "8", "--block_size", "64",
        "--max_context_len", "33024", "--max_kv_tokens_per_lane", "205632",
        "--max_num_seqs", "1024", "--chunk_budgets", "256", "512", "1024", "2048", "4096", "8192",
        "--query_one_context_max", "65", "--output", str(output),
    ])
    capsys.readouterr()
    plan = json.loads(output.read_text())
    assert len(plan["cases"]) == 2 and len(plan["engines"]) == 1
    assert plan["cases"][1]["measured"] == 65


def _check_runner_contexts(plan, case, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    pending, steps, aborted = {}, [], []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.1

    state = runner._WorkerState(
        engine_index=0, torch_module=SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None)),
    )
    monkeypatch.setattr(runner, "_STATE", state)

    def add_request(request_id, prompt, params):
        assert len(prompt["prompt_token_ids"]) == case.prompt_len
        pending[request_id] = params["max_tokens"]

    def step():
        context = case.prompt_len + len(steps) - 1 if steps else 0
        query = 1 if steps else case.prompt_len
        steps.append(context)
        request_id = next(iter(pending))
        active = state.begin_step(SimpleNamespace(
            finished_req_ids=[], num_scheduled_tokens={request_id: query},
            scheduled_new_reqs=([SimpleNamespace(req_id=request_id,
                prompt_token_ids=[0] * case.prompt_len, num_computed_tokens=0)]
                                if context == 0 else []),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request_id] if context else [], num_computed_tokens=[context] if context else [],
            ),
        ))
        if active is not None:
            state.active = active
            state.capture_metadata("mla_metadata", lambda: {
                "num_actual_tokens": 1, "num_decodes": 1, "num_prefills": 0, "num_decode_tokens": 1,
            })
            state.capture_metadata("indexer_metadata", lambda: {"max_seq_len": context + 1})
            for layer in plan.shape.layers:
                for scope in ((INDEXER_SCOPE, MLA_SCOPE) if layer.indexer_executed else (MLA_SCOPE,)):
                    state.timed_call(layer.global_layer_id, scope, lambda: None, (), {})
            state.active = None
            state.end_step(active)
        for request_id in list(pending):
            pending[request_id] -= 1
            if not pending[request_id]:
                del pending[request_id]

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.clear()

    def rpc(function, args=()):
        return [function(None, *args)]

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(add_request=add_request, step=step, abort_request=abort,
                                   has_unfinished_requests=lambda: bool(pending)),
        collective_rpc=rpc,
    )
    records = runner.run_case(llm, plan, case, CUDA_EVENT, random.Random(1234), 100, tmp_path, False)
    contexts = list(range(case.prompt_len, case.prompt_len + case.measured))
    assert steps == [0, *contexts], "Only the setup prefill is unmeasured"
    assert len(aborted) == 1
    recorded_steps = [record for record in records if record["kind"] == "step"]
    assert [record["requests"][0]["num_computed_tokens"] for record in recorded_steps] == contexts
    assert all(record["requests"][0]["num_scheduled_tokens"] == 1 for record in recorded_steps)
    assert all(record["requests"][0]["prompt_len"] == case.prompt_len for record in recorded_steps)
    assert len([record for record in records if record["kind"] == "call"]) == case.measured * 12


def test_existing_runner_schedules_every_small_context_without_injected_cache(monkeypatch, tmp_path):
    plan = build_plan(65)
    _check_runner_contexts(plan, plan.cases[1], monkeypatch, tmp_path)


def test_native_window_fixture_uses_existing_runner_without_tail_phase_relabel(monkeypatch, tmp_path):
    # CPU fake calls exercise scheduling only; native collection remains rejected.
    plan = _native_plan()
    _check_runner_contexts(plan, plan.cases[1], monkeypatch, tmp_path)


def test_native_plan_preserves_limits_and_exact_workload():
    from frontier.profiling.attention.dsa_rows import StepRequest

    plan = _native_plan()
    assert len(plan.engines) == 1 and len(plan.cases) == 2
    engine = plan.engines[0]
    assert (engine.max_model_len, engine.chunk_budget, engine.max_num_seqs) == (7168, 8192, 128)
    prefill, decode = plan.cases
    assert (prefill.phase, prefill.prompt_len, prefill.max_tokens) == (PHASE_PREFILL, 6144, 1)
    assert (decode.phase, decode.prompt_len, decode.max_tokens) == (PHASE_DECODE, 6143, 1025)
    assert decode.warmup == 0 and decode.measured == 1024
    assert runner._case_window(decode, engine) == (0, 1024)
    assert decode.estimated_steps == 1025
    assert decode.prompt_len + decode.max_tokens == engine.max_model_len
    assert not StepRequest(1, decode.prompt_len, decode.prompt_len).is_prefill
    assert StepRequest(1, 6143, 6144).is_prefill, "Actual remote tail is a different phase"
    assert plan.to_json()["limits"]["engine_policy"] == "native_defaults"
    assert plan.to_json()["timing_collection"] == "pending_native_scope_instrumentation"
    assert plan.summary()["timing_collection"] == "pending_native_scope_instrumentation"


@pytest.mark.parametrize("minimum,maximum", [(0, 0), (0, 65), (1, 1), (6143, 6143), (6143, 7166)])
def test_native_window_includes_both_cached_endpoints(minimum, maximum):
    plan = _native_plan(minimum=minimum, maximum=maximum)
    decodes = [case for case in plan.cases if case.phase == PHASE_DECODE]
    if maximum == 0:
        assert not decodes
    else:
        case, = decodes
        assert case.prompt_len == max(1, minimum)
        assert case.prompt_len + case.measured - 1 == maximum
        assert case.max_tokens == case.measured + 1 and case.warmup == 0
    assert any(case.phase == PHASE_PREFILL and case.prompt_len == 1
               for case in plan.cases) == (minimum == 0)


def test_native_exact_prefill_only_and_decode_only_need_no_broad_grid():
    prefill = _native_plan(minimum=0, maximum=None)
    assert [case.case_id for case in prefill.cases] == ["e0-prefill-p6144"]
    decode = _native_plan(prefill=None)
    assert [case.case_id for case in decode.cases] == ["e0-decode-q1-c6143-7166"]
    zero = _native_plan(minimum=0, maximum=0, prefill=1)
    assert [case.case_id for case in zero.cases] == ["e0-prefill-p1"]


@pytest.mark.parametrize("minimum,maximum", [(-1, 7166), (True, 7166), (1.5, 7166),
                                                     (7167, 7166), (6143, None)])
def test_native_plan_rejects_invalid_window(minimum, maximum):
    with pytest.raises(ValueError, match="query_one_context_min"):
        _native_plan(minimum=minimum, maximum=maximum)


@pytest.mark.parametrize("prefill", [0, -1, True, 1.5, 7168])
def test_native_plan_rejects_invalid_exact_prefill(prefill):
    with pytest.raises(ValueError, match="prefill_prompt_len"):
        _native_plan(prefill=prefill)


@pytest.mark.parametrize("overrides,message", [
    ({"chunk_budgets": (4096,)}, "disables chunking"),
    ({"chunk_budgets": (8192, 16384)}, "exactly one"),
    ({"max_num_seqs": 8193}, "max_num_seqs"),
    ({"engine_policy": "native"}, "engine_policy"),
    ({"max_kv_tokens_per_lane": 7167}, "capacity for 7168"),
])
def test_native_plan_rejects_incoherent_limits(overrides, message):
    with pytest.raises(ValueError, match=message):
        _native_plan(**overrides)


def test_native_plan_does_not_increase_maxlen_or_invent_workload():
    with pytest.raises(ValueError, match="capacity for 7169"):
        _native_plan(maximum=7167)
    with pytest.raises(ValueError, match="explicit prefill or q1 window"):
        _native_plan(minimum=0, maximum=None, prefill=None)
    with pytest.raises(ValueError, match="stay below"):
        _native_plan(engine_policy="eager_chunked")
    assert _native_plan(chunk_budgets=(7168,)).engines[0].chunk_budget == 7168


def test_native_constructor_omits_resolution_overrides(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kw: kw))
    options = runner.build_engine(_native_plan(), 0, "CPU_FAKE_MODEL", 1234, 0.85)
    assert (options["max_model_len"], options["max_num_batched_tokens"],
            options["max_num_seqs"]) == (7168, 8192, 128)
    assert options["enable_prefix_caching"] is False
    assert options["enable_chunked_prefill"] is False
    assert options["tensor_parallel_size"] == 1 and options["load_format"] == "dummy"
    assert not ({"enforce_eager", "compilation_config", "async_scheduling", "quantization",
                 "speculative_config", "long_prefill_token_threshold"} & options.keys())
    with pytest.raises(ValueError, match="does not accept engine_options"):
        runner.build_engine(_native_plan(), 0, "CPU_FAKE_MODEL", 1234, 0.85,
                            compilation_config={"mode": 0})


NATIVE_PLAN_ARGS = [
    "--model_name", "zai-org/GLM-5.3",
    "--model_config_path", "data/config/models/zai-org__GLM-5.3.json",
    "--target_tensor_parallel_size", "8", "--block_size", "64",
    "--max_context_len", "7168", "--max_kv_tokens_per_lane", "7168",
    "--max_num_seqs", "128", "--chunk_budgets", "8192",
    "--engine_policy", "native_defaults", "--prefill_prompt_len", "6144",
    "--query_one_context_min", "6143", "--query_one_context_max", "7166",
]


def test_native_cli_uses_same_plan_and_window(tmp_path, capsys):
    output = tmp_path / "plan.json"
    main(["plan", *NATIVE_PLAN_ARGS, "--output", str(output)])
    summary = json.loads(capsys.readouterr().out.split("[dsa]")[0])
    assert summary["timing_collection"] == "pending_native_scope_instrumentation"
    assert json.loads(output.read_text()) == json.loads(json.dumps(_native_plan().to_json()))


def test_native_run_is_rejected_before_model_io_or_construction(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Native timing must be rejected before engine allocation")

    monkeypatch.setattr(runner, "build_engine", forbidden)
    with pytest.raises(ValueError, match="per-scope instrumentation is not qualified"):
        main(["run", *NATIVE_PLAN_ARGS, "--engine_index", "0", "--model_dir", "MISSING",
              "--raw_dir", str(tmp_path / "raw"), "--trace_dir", str(tmp_path / "traces"),
              "--expected_attention_backend", "UNOBSERVED", "--gpu_memory_utilization", "0.85",
              "--seed", "1234"])
    with pytest.raises(ValueError, match="per-scope instrumentation is not qualified"):
        runner.run_engine(plan=_native_plan(), engine_index=0, model_dir="MISSING",
            vocab_size=10, seed=1234, gpu_memory_utilization=0.85,
            expected_attention_backend="UNOBSERVED", raw_path=tmp_path / "raw.jsonl",
            trace_dir=tmp_path / "traces", keep_traces=False)
    with pytest.raises(ValueError, match="per-scope instrumentation is not qualified"):
        runner.measure_engine(llm=None, plan=_native_plan(), engine_index=0, inventory={},
            vocab_size=10, seed=1234, expected_attention_backend="UNOBSERVED",
            raw_path=tmp_path / "raw.jsonl", trace_dir=tmp_path / "traces",
            keep_traces=False, case_ids=None, log_label="CPU_ONLY")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("partial", [[], ["--allow_partial_engines"]])
def test_native_aggregation_cannot_emit_empty_partial_or_eager_csv(tmp_path, partial):
    with pytest.raises(ValueError, match="per-scope instrumentation is not qualified"):
        main(["aggregate", *NATIVE_PLAN_ARGS, "--raw_dir", str(tmp_path / "raw"),
              "--output_root", str(tmp_path / "out"), "--device", "CPU_ONLY",
              "--source_revision", "fixture", "--model_config_revision", "fixture", *partial])
    with pytest.raises(ValueError, match="per-scope instrumentation is not qualified"):
        build_dsa_rows(_native_plan(), [])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("policy", ["native_defaults", "unknown"])
def test_native_engine_record_cannot_be_aggregated_under_legacy_plan(policy):
    with pytest.raises(ValueError, match="cannot collect or aggregate"):
        build_dsa_rows(build_plan(0), [{
            "kind": "engine", "engine_index": 0, "engine_policy": policy,
        }])


@pytest.mark.parametrize("measurement", [CUDA_EVENT, KERNEL_ONLY])
def test_scope_aggregation_keeps_every_serving_context_and_both_variants(measurement):
    plan = build_plan(65)
    records = [{
        "kind": "engine", "engine_index": 0, "chunk_budget": 256,
        "attention_backend": "FLASH_ATTN_MLA_SPARSE", "attention_impl": "FlashAttnMLASparseImpl",
        "kv_cache_dtype": "auto",
    }]
    for context in range(66):
        case = plan.cases[0 if context == 0 else 1]
        records.append({
            "kind": "step", "measurement": measurement, "engine_index": 0,
            "case_id": case.case_id, "phase": case.phase, "step_index": context,
            "requests": [{"num_scheduled_tokens": 1, "num_computed_tokens": context, "prompt_len": 1}],
            "mla_metadata": {"num_actual_tokens": 1, "num_decodes": 1,
                             "num_prefills": 0, "num_decode_tokens": 1},
            "indexer_metadata": {"num_decodes": 1, "num_decode_tokens": 1, "max_seq_len": context + 1},
        })
        for layer in plan.shape.layers:
            for scope in ((INDEXER_SCOPE, MLA_SCOPE) if layer.indexer_executed else (MLA_SCOPE,)):
                records.append({
                    "kind": "call", "measurement": measurement, "engine_index": 0,
                    "case_id": case.case_id, "step_index": context,
                    "global_layer_id": layer.global_layer_id, "scope": scope,
                    "time_ms": 0.1, "kernel_count": 1 if measurement == KERNEL_ONLY else None,
                })
    rows = build_dsa_rows(plan, records)[measurement]
    assert len(rows) == 132
    for variant in ("full_index", "shared_index"):
        variant_rows = [row for row in rows if row["attention_variant"] == variant]
        assert {row["max_key_len"] for row in variant_rows} == set(range(1, 67))
        zero = next(row for row in variant_rows if row["max_key_len"] == 1)
        assert zero["phase"] == PHASE_PREFILL and zero["is_prefill"]
        assert json.loads(zero["prefill_seq_lens"]) == [1]
        assert json.loads(zero["prefill_kv_cache_sizes"]) == [0]
        small = [row for row in variant_rows if row["phase"] == PHASE_DECODE]
        assert {json.loads(row["decode_kv_cache_sizes"])[0] for row in small} == set(range(1, 66))
        assert all((f"time_stats.{INDEXER_SCOPE}.count" in row) == (variant == "full_index")
                   for row in variant_rows)
