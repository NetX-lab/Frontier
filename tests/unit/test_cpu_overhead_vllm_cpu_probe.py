"""CPU-overhead CSVs from vLLM CPU-probe logs."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from frontier.profiling.cpu_overhead.validation import (
    cpu_overhead_feature_columns,
    validate_cpu_overhead_dataframe,
)
from frontier.profiling.cpu_overhead.vllm_cpu_probe import (
    StageStep,
    cpu_overhead_tables,
    dp_allreduce_latency,
    main,
    stage_log_path,
    stage_overhead_terms,
    step_identities,
)
from frontier.types import MeasurementType

IDENTITY = dict(model_name="tiny", tensor_parallel_degree=1, profiling_precision="BF16", scheduling_mode="sync")
PERIOD_TERMS = ("schedule", "prepare_inputs_e2e", "sampler_e2e", "process_model_outputs")

# Each test's requests: request id -> [prompt length, computed tokens]. Every engine record
# built for a request advances it, so the records carry the probe's request-phase fields.
_REQUESTS: dict[str, list[int]] = {}


@pytest.fixture(autouse=True)
def _fresh_requests():
    _REQUESTS.clear()


def _forward(step: int, stage_id: int, execute_start: float, scheduled: dict[str, int], *,
             device_ms: float = 4.0, mode: str = "NONE", allreduce: float | None = None,
             entry: float = 0.00005, kv_load: float | None = None) -> dict:
    # Preprocess ends 0.3 ms after execute_start and the forward starts 0.05 ms later. Entering the
    # forward context takes `entry` seconds after a DP all-reduce of `allreduce` seconds, the first
    # `kv_load` of them in the KV connector's start_load_kv; the host then launches the forward's
    # kernels for 0.4 ms.
    forward_start = execute_start + 0.00035
    allreduce_end = forward_start + (allreduce or 0.0)
    forward_context_entered = allreduce_end + entry
    return {
        "step": step,
        "pipeline_stage_id": stage_id,
        "recv_start": None,
        "recv_end": None,
        "execute_start": execute_start,
        "preprocess_end": execute_start + 0.0003,
        "forward_start": forward_start,
        "forward_context_entered": forward_context_entered,
        "forward_end": forward_context_entered + 0.0004,
        "sample_end": None,
        "bookkeep_end": None,
        "execute_return": forward_context_entered + 0.0005,
        "dp_allreduce_start": forward_start if allreduce is not None else None,
        "dp_allreduce_end": allreduce_end if allreduce is not None else None,
        "kv_load_start": allreduce_end if kv_load is not None else None,
        "kv_load_end": allreduce_end + kv_load if kv_load is not None else None,
        "send_start": None,
        "send_end": None,
        "num_input_tokens": sum(scheduled.values()),
        "cudagraph_runtime_mode": mode,
        "num_scheduled_tokens": scheduled,
        "forward_device_ms": device_ms,
    }


def _finish(forward: dict, device_end: float, sampling: float) -> float:
    """Sample on the last stage `sampling` seconds after the forward's later stream; return sample_end."""
    sample_end = max(forward["forward_end"], device_end) + sampling
    forward.update(sample_end=sample_end, bookkeep_end=sample_end + 0.0001, execute_return=sample_end + 0.0002)
    return sample_end


def _engine_record(step: int, start: float, scheduled: dict[str, int], sample_end: float, *,
                   engine_loop: str = "step", kv_role: str | None = None) -> dict:
    # Schedule takes 0.1 ms; the engine's output update ends 0.3 ms after the last stage's sampling.
    # A request scheduled more than one token runs its prompt; one first scheduled one token decodes
    # after an 8-token prompt.
    computed, prompt = {}, {}
    for request_id, num_tokens in scheduled.items():
        request = _REQUESTS.setdefault(request_id, [0, 0] if num_tokens > 1 else [8, 8])
        if num_tokens > 1:
            request[0] = max(request[0], request[1] + num_tokens)
        prompt[request_id], computed[request_id] = request
        request[1] += num_tokens
    return {
        "step": step,
        "engine_loop": engine_loop,
        "step_start": start,
        "schedule_end": start + 0.0001,
        "execute_end": sample_end + 0.00025,
        "update_end": sample_end + 0.0003,
        "num_scheduled_tokens": scheduled,
        "num_computed_tokens": computed,
        "num_prompt_tokens": prompt,
        "kv_role": kv_role,
    }


def _single_stage_step(step: int, start: float, scheduled: dict[str, int], **forward_kwargs) -> tuple[dict, dict]:
    # A sync-loop step: the runner starts 0.1 ms after schedule_end and samples for 2 ms.
    forward = _forward(step, 0, start + 0.0002, scheduled, **forward_kwargs)
    device_end = forward["forward_context_entered"] + forward["forward_device_ms"] * 1e-3
    return _engine_record(step, start, scheduled, _finish(forward, device_end, 0.002)), forward


def _idle_engine_record(step: int, start: float) -> dict:
    return _engine_record(step, start, {}, start + 0.0001)


def test_single_stage_terms_cover_the_period_outside_the_forward_device_time() -> None:
    steps = [
        _single_stage_step(0, 10.0, {"a": 16}),
        _single_stage_step(1, 10.0070, {"a": 1}),
        _single_stage_step(2, 10.0140, {"b": 16}),
    ]
    engine = [record for record, _ in steps] + [_idle_engine_record(3, 10.5)]

    out = stage_overhead_terms(engine, [[forward for _, forward in steps]], dp_allreduce_ms=None)

    assert [(step.stage_id, step.family, step.identity) for step in out] == [
        (0, MeasurementType.CUDA_EVENT, (1, 16, 0)),
        (0, MeasurementType.CUDA_EVENT, (1, 0, 1)),
        (0, MeasurementType.CUDA_EVENT, (1, 16, 0)),
    ]
    first = out[0][3]
    assert first["schedule"] == pytest.approx(0.1)
    # Dispatch, preprocess and the forward-context entry, up to forward_context_entered.
    assert first["prepare_inputs_e2e"] == pytest.approx(0.5)
    # Device-bound eager step: the 4.0 ms device stream hides the 0.4 ms launch,
    # and the device runs on for 3.6 ms after it.
    assert first["forward_launch"] == pytest.approx(0.4)
    assert first["forward_drain"] == pytest.approx(3.6)
    assert first["sampler_e2e"] == pytest.approx(2.0)
    assert first["process_model_outputs"] == pytest.approx(7.0 - 6.6)
    assert sum(first[term] for term in PERIOD_TERMS) == pytest.approx(7.0 - 4.0)
    # Step 1 is followed by a step of another request, step 2 by an empty step:
    # the engine may have waited, so the term ends at update_end.
    assert out[1][3]["process_model_outputs"] == pytest.approx(0.3)
    assert out[2][3]["process_model_outputs"] == pytest.approx(0.3)


def test_host_bound_eager_terms_leave_out_the_launch_stream() -> None:
    engine, forward = _single_stage_step(0, 10.0, {"a": 16}, device_ms=0.25)

    [step] = stage_overhead_terms([engine], [[forward]], dp_allreduce_ms=None)

    terms = step.terms
    assert terms["forward_launch"] == pytest.approx(0.4)
    assert terms["forward_drain"] == 0.0
    assert terms["sampler_e2e"] == pytest.approx(2.0)
    assert sum(terms[term] for term in PERIOD_TERMS) + terms["forward_launch"] == pytest.approx(
        (engine["update_end"] - engine["step_start"]) * 1e3
    )


def test_full_graph_replays_are_kernel_only_and_publish_no_launch() -> None:
    engine, forward = _single_stage_step(0, 10.0, {"a": 1, "b": 1}, device_ms=0.25, mode="FULL")

    [step] = stage_overhead_terms([engine], [[forward]], dp_allreduce_ms=None)

    assert (step.stage_id, step.family, step.identity) == (0, MeasurementType.KERNEL_ONLY, (2, 0, 2))
    assert "forward_launch" not in step.terms and "forward_drain" not in step.terms
    assert step.terms["sampler_e2e"] == pytest.approx(2.0)
    assert step.engine_idle_ms is None


def test_prepare_leaves_out_the_recorded_kv_load_only() -> None:
    # The second forward's context entry runs a 5 ms start_load_kv; the third enters as slowly
    # without a KV load, which is host work and stays in prepare.
    steps = [
        _single_stage_step(0, 10.0, {"a": 1}),
        _single_stage_step(1, 10.0070, {"a": 1, "b": 1}, entry=0.00505, kv_load=0.005),
        _single_stage_step(2, 10.0190, {"a": 1, "b": 1}, entry=0.00505),
    ]

    out = stage_overhead_terms([r for r, _ in steps], [[f for _, f in steps]], dp_allreduce_ms=None)

    assert [step.terms["prepare_inputs_e2e"] for step in out] == pytest.approx([0.5, 0.5, 5.5])
    assert out[1][3]["sampler_e2e"] == pytest.approx(2.0)


def test_prepare_leaves_out_a_kv_load_common_to_every_step() -> None:
    # A PD decode instance whose every step loads KV for 5 ms: a median over the stage log would
    # keep the wait in prepare.
    steps = [_single_stage_step(i, 10.0 + 0.012 * i, {f"r{i}": 1}, entry=0.00505, kv_load=0.005)
             for i in range(3)]

    out = stage_overhead_terms([r for r, _ in steps], [[f for _, f in steps]], dp_allreduce_ms=None)

    assert [step.terms["prepare_inputs_e2e"] for step in out] == pytest.approx([0.5, 0.5, 0.5])


def _phase_record(step: int, phases: dict[str, tuple[int, int, int]], kv_role: str | None = None) -> dict:
    """An engine record of requests given as (scheduled, computed before the step, prompt length)."""
    return {
        "step": step,
        "num_scheduled_tokens": {r: phase[0] for r, phase in phases.items()},
        "num_computed_tokens": {r: phase[1] for r, phase in phases.items()},
        "num_prompt_tokens": {r: phase[2] for r, phase in phases.items()},
        "kv_role": kv_role,
    }


def test_a_one_token_prompt_tail_and_a_decode_step_publish_distinct_identities() -> None:
    # Chunk budget 64, prompt 65, output 2: the last prompt token runs alone, then one decode.
    records = [
        _phase_record(0, {"r": (64, 0, 65)}),
        _phase_record(1, {"r": (1, 64, 65)}),
        _phase_record(2, {"r": (1, 65, 65)}),
    ]

    assert step_identities(records) == [(1, 64, 0), (1, 1, 0), (1, 0, 1)]


def test_a_pd_decode_instance_counts_a_request_first_step_as_decode() -> None:
    # The connector loads 64 of 65 prompt tokens; the decode instance computes the last one,
    # Frontier's first decode step on the DECODE role.
    records = [
        _phase_record(0, {"r": (1, 64, 65)}, kv_role="kv_consumer"),
        _phase_record(1, {"r": (1, 65, 65)}, kv_role="kv_consumer"),
    ]

    assert step_identities(records) == [(1, 0, 1), (1, 0, 1)]
    # A co-location prefix-cache hit leaves the same last prompt token, which is prefill.
    assert step_identities([dict(records[0], kv_role=None)]) == [(1, 1, 0)]


def test_a_resumed_preemption_victim_recomputes_as_prefill() -> None:
    # Request r decodes once, is preempted and recomputes its 66 computed tokens in chunks of 64, 1
    # and 1, all prefill; the step after computes its next token, a decode.
    records = [
        _phase_record(0, {"r": (65, 0, 65)}),
        _phase_record(1, {"r": (1, 65, 65)}),
        _phase_record(2, {"r": (64, 0, 65)}),
        _phase_record(3, {"r": (1, 64, 65)}),
        _phase_record(4, {"r": (1, 65, 65)}),
        _phase_record(5, {"r": (1, 66, 65)}),
    ]

    assert step_identities(records) == [(1, 65, 0), (1, 0, 1), (1, 64, 0), (1, 1, 0), (1, 1, 0), (1, 0, 1)]


def test_step_identities_reject_multi_token_decode_steps_and_unknown_kv_roles() -> None:
    with pytest.raises(ValueError, match="schedules 3 decode tokens for request r"):
        step_identities([_phase_record(0, {"r": (3, 65, 65)})])
    with pytest.raises(ValueError, match="KV connector role 'kv_both'"):
        step_identities([_phase_record(0, {"r": (1, 65, 65)}, kv_role="kv_both")])


def test_records_of_an_older_probe_are_rejected() -> None:
    engine, forward = _single_stage_step(0, 10.0, {"a": 16})
    legacy_engine = {key: value for key, value in engine.items() if key != "num_prompt_tokens"}
    legacy_forward = {key: value for key, value in forward.items() if not key.startswith("kv_load")}

    with pytest.raises(ValueError, match=r"engine step 0 has no \['num_prompt_tokens'\]"):
        stage_overhead_terms([legacy_engine], [[forward]], dp_allreduce_ms=None)
    with pytest.raises(ValueError, match="pipeline stage 0 forward 0 has no kv_load_start"):
        stage_overhead_terms([engine], [[legacy_forward]], dp_allreduce_ms=None)


def _dp_pair() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Engine 0 reaches the DP all-reduce first and waits 3 ms for engine 1, whose own all-reduce
    # takes 0.36 ms. Engine 0's second forward pairs with a dummy forward of engine 1, which has
    # no record.
    engine0_first = _single_stage_step(0, 10.0, {"a": 16}, allreduce=0.003)
    late_start = engine0_first[1]["dp_allreduce_end"] - 0.00036 - 0.00055
    engine1_first = _single_stage_step(0, late_start, {"b": 16}, allreduce=0.00036)
    engine0_second = _single_stage_step(1, 10.020, {"a": 1}, allreduce=0.0005)
    return ([engine0_first[0], engine0_second[0]], [engine0_first[1], engine0_second[1]],
            [engine1_first[0]], [engine1_first[1]])


def test_dp_allreduce_latency_is_the_latest_starter_median_over_complete_all_reduces() -> None:
    _, forwards0, _, forwards1 = _dp_pair()

    assert dp_allreduce_latency([[forwards0], [forwards1]]) == pytest.approx(0.36)
    with pytest.raises(ValueError, match="list the logs of every DP engine"):
        dp_allreduce_latency([[forwards0]])


def test_dp_allreduce_latency_without_a_complete_all_reduce_is_the_median_of_all_recorded() -> None:
    # Engine 0's second all-reduce (0.5 ms) pairs with a dummy forward of engine 1, and engine 1's
    # (0.36 ms) with a forward outside the logs: no all-reduce has a record from both engines.
    _, forwards0, _, forwards1 = _dp_pair()
    engine1_alone = [dict(forwards1[0], dp_allreduce_start=20.0, dp_allreduce_end=20.00036)]

    assert dp_allreduce_latency([[forwards0[1:]], [engine1_alone]]) == pytest.approx((0.5 + 0.36) / 2)


def test_prepare_charges_the_dp_all_reduce_latency_and_leaves_out_the_dp_wait() -> None:
    engine0, forwards0, engine1, forwards1 = _dp_pair()

    first0, second0 = stage_overhead_terms(engine0, [forwards0], dp_allreduce_ms=0.36)
    [first1] = stage_overhead_terms(engine1, [forwards1], dp_allreduce_ms=0.36)

    assert first0[3]["prepare_inputs_e2e"] == pytest.approx(0.5 + 0.36)
    assert first1[3]["prepare_inputs_e2e"] == pytest.approx(0.5 + 0.36)
    assert second0[3]["prepare_inputs_e2e"] == pytest.approx(0.5 + 0.36)
    assert first0[3]["forward_launch"] == pytest.approx(0.4)


def _two_stage_step(step: int, start: float, scheduled: dict[str, int], *, stage0_execute_start: float,
                    stage1_execute_start: float, stage0_device_ms: float = 4.0) -> tuple[dict, dict, dict]:
    # Stage 0 sends its output 0.1 ms after its execute_return. Stage 1 launches 0.4 ms after the later
    # of its forward-context entry and stage 0's device end, runs 4 ms on the device and samples for
    # 0.6 ms after its device end.
    stage0 = _forward(step, 0, stage0_execute_start, scheduled, device_ms=stage0_device_ms)
    stage0.update(send_start=stage0["execute_return"], send_end=stage0["execute_return"] + 0.0001)
    stage1 = _forward(step, 1, stage1_execute_start, scheduled)
    stage1.update(recv_start=start, recv_end=stage1_execute_start)
    device_end0 = stage0["forward_context_entered"] + stage0_device_ms * 1e-3
    launch_start1 = max(stage1["forward_context_entered"], device_end0)
    stage1["forward_end"] = launch_start1 + 0.0004
    sample_end = _finish(stage1, launch_start1 + 4.0e-3, 0.0006)
    return _engine_record(step, start, scheduled, sample_end, engine_loop="batch_queue"), stage0, stage1


def _pipelined_batches(*, stage0_device_ms: float = 4.0):
    # The second batch is submitted 0.2 ms after the first and waits for stage 0 to send the first.
    engine0, first0, first1 = _two_stage_step(
        0, 10.0, {"a": 16}, stage0_execute_start=10.0002, stage1_execute_start=10.0010,
        stage0_device_ms=stage0_device_ms,
    )
    engine1, second0, second1 = _two_stage_step(
        1, 10.0002, {"b": 16}, stage0_execute_start=first0["send_end"] + 0.0001,
        stage1_execute_start=first1["execute_return"] + 0.0001,
    )
    engine = [engine0, engine1, _idle_engine_record(2, 10.5)]
    return engine, [[first0, second0], [first1, second1]]


def test_stage_rows_charge_each_interval_to_the_stage_that_runs_it() -> None:
    engine, stages = _pipelined_batches()

    stage0, stage1, *_ = stage_overhead_terms(engine, stages, dp_allreduce_ms=None)

    assert (stage0[0], stage1[0]) == (0, 1)
    assert stage0[2] == stage1[2] == (1, 16, 0)
    assert stage0[3]["schedule"] == pytest.approx(0.1)
    assert stage1[3]["schedule"] == 0.0
    assert stage0[3]["prepare_inputs_e2e"] == pytest.approx(0.5)
    # A later stage's dispatch starts at execute_start, after its receive wait.
    assert stage1[3]["prepare_inputs_e2e"] == pytest.approx(0.4)
    assert stage0[3]["sampler_e2e"] == stage0[3]["process_model_outputs"] == 0.0
    assert stage1[3]["sampler_e2e"] == pytest.approx(0.6)
    # The next batch, submitted before this batch's update_end, holds another request.
    assert stage1[3]["process_model_outputs"] == pytest.approx(0.3)
    assert list(stage0[3]) == list(stage1[3]) == [*PERIOD_TERMS, "forward_launch", "forward_drain"]


def test_stage0_dispatch_leaves_out_queueing_behind_the_previous_batch() -> None:
    engine, stages = _pipelined_batches()

    second_stage0 = stage_overhead_terms(engine, stages, dp_allreduce_ms=None)[2]

    assert second_stage0[0] == 0
    assert second_stage0[3]["prepare_inputs_e2e"] == pytest.approx(0.5)


def test_stage_forward_launch_and_sampler_start_after_the_upstream_device_end() -> None:
    # Stage 0's device runs 30 ms; stage 1's host enters its forward context long before that.
    engine, stages = _pipelined_batches(stage0_device_ms=30.0)

    stage0, stage1, *_ = stage_overhead_terms(engine, stages, dp_allreduce_ms=None)

    assert stage0[3]["forward_launch"] == pytest.approx(0.4)
    assert stage1[3]["forward_launch"] == pytest.approx(0.4)
    assert stage1[3]["sampler_e2e"] == pytest.approx(0.6)


def test_stage_join_requires_matching_token_maps() -> None:
    engine, (stage0, stage1) = _pipelined_batches()

    with pytest.raises(ValueError, match="pipeline stage 1 logged 0 forwards"):
        stage_overhead_terms(engine, [stage0, []], dp_allreduce_ms=None)
    other = [stage1[0], dict(stage1[1], num_scheduled_tokens={"c": 16})]
    with pytest.raises(ValueError, match="pipeline stage 1 logged 2 forwards"):
        stage_overhead_terms(engine, [stage0, other], dp_allreduce_ms=None)


TERMS = {"schedule": 0.1, "prepare_inputs_e2e": 0.5, "sampler_e2e": 2.0, "process_model_outputs": 0.4}
EAGER_TERMS = {**TERMS, "forward_launch": 0.4, "forward_drain": 3.6}


def test_tables_split_families_and_key_rows_by_stage() -> None:
    steps = [
        StageStep(0, MeasurementType.KERNEL_ONLY, (3, 0, 3), TERMS),
        StageStep(0, MeasurementType.CUDA_EVENT, (8, 0, 8), EAGER_TERMS),
        StageStep(1, MeasurementType.CUDA_EVENT, (1, 16, 0), EAGER_TERMS),
        StageStep(0, MeasurementType.CUDA_EVENT, (2, 16, 1), EAGER_TERMS),
        StageStep(0, MeasurementType.CUDA_EVENT, (1, 16, 0), EAGER_TERMS),
    ]

    tables = cpu_overhead_tables(steps, **IDENTITY)

    kernel_only = tables[MeasurementType.KERNEL_ONLY]
    eager = tables[MeasurementType.CUDA_EVENT]
    columns = ["pipeline_stage_id", "batch_size", "num_prefill_tokens", "num_decode_tokens"]
    assert kernel_only[columns].values.tolist() == [[0, 3, 0, 3]]
    assert eager[columns].values.tolist() == [[0, 1, 16, 0], [0, 2, 16, 1], [0, 8, 0, 8], [1, 1, 16, 0]]
    assert (eager["ray_comm_time_mean"] == 0.0).all()
    assert set(eager["measurement_type"]) == {MeasurementType.CUDA_EVENT.value}
    assert set(kernel_only["measurement_type"]) == {MeasurementType.KERNEL_ONLY.value}
    assert "forward_launch_median" not in kernel_only.columns
    assert "forward_drain_median" not in kernel_only.columns
    assert "engine_idle_ms" not in eager.columns


def test_tables_take_mean_and_median_per_tuple() -> None:
    steps = [
        StageStep(0, MeasurementType.KERNEL_ONLY, (2, 0, 2), {**TERMS, "schedule": s, "process_model_outputs": p})
        for s, p in ((0.1, 0.4), (0.2, 0.3), (0.6, 0.8))
    ]

    tables = cpu_overhead_tables(steps, **IDENTITY)

    row = tables[MeasurementType.KERNEL_ONLY].set_index("batch_size").loc[2]
    assert row["schedule_median"] == pytest.approx(0.2)
    assert row["schedule_mean"] == pytest.approx(0.3)
    assert row["process_model_outputs_median"] == pytest.approx(0.4)
    assert row["num_steps"] == 3


def _write_log(path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_engine(log, engine_records, stage_records) -> None:
    _write_log(log, engine_records)
    for stage_id, records in enumerate(stage_records):
        _write_log(stage_log_path(log, stage_id), records)


def test_cli_reads_each_serving_instance_and_writes_one_csv_per_family(tmp_path) -> None:
    # A PD prefill instance runs each request for one eager step; the decode instance replays
    # FULL graphs.
    prefill = [_single_stage_step(0, 5.0, {"a": 16}), _single_stage_step(1, 5.007, {"b": 16})]
    decode = [_single_stage_step(i, 10.0 + 0.007 * i, {"a": 1, "b": 1}, mode="FULL") for i in range(3)]
    (tmp_path / "prefill").mkdir()
    (tmp_path / "decode").mkdir()
    _write_engine(tmp_path / "prefill/cpu_probe.jsonl", [r for r, _ in prefill], [[f for _, f in prefill]])
    _write_engine(tmp_path / "decode/cpu_probe.jsonl", [r for r, _ in decode] + [_idle_engine_record(3, 11.0)],
                  [[f for _, f in decode]])
    eager_file, kernel_file = tmp_path / "out/cpu_overheads.csv", tmp_path / "out/cpu_overheads_kernel_only.csv"

    main([
        "--cpu_probe_logs", str(tmp_path / "prefill/cpu_probe.jsonl"),
        "--cpu_probe_logs", str(tmp_path / "decode/cpu_probe.jsonl"),
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(eager_file), "--kernel_only_output_file", str(kernel_file),
    ])

    eager = validate_cpu_overhead_dataframe(pd.read_csv(eager_file))
    kernel_only = validate_cpu_overhead_dataframe(pd.read_csv(kernel_file))
    columns = ["pipeline_stage_id", "batch_size", "num_prefill_tokens", "num_decode_tokens"]
    assert eager[columns].values.tolist() == [[0, 1, 16, 0]]
    assert eager.loc[0, "num_steps"] == 2
    assert eager.loc[0, "process_model_outputs_median"] == pytest.approx(0.3)
    assert eager.loc[0, "forward_launch_median"] == pytest.approx(0.4)
    assert kernel_only[columns].values.tolist() == [[0, 2, 0, 2]]
    assert kernel_only.loc[0, "num_steps"] == 3
    assert kernel_only.loc[0, "process_model_outputs_median"] == pytest.approx(0.4)


def test_cli_reads_dp_engines_with_their_stage_logs(tmp_path) -> None:
    for dp in (0, 1):
        engine, stages = _pipelined_batches()
        for records in stages:
            for record in records:
                record.update(dp_allreduce_start=record["forward_start"], dp_allreduce_end=record["forward_start"])
        _write_engine(tmp_path / f"cpu_probe_dp{dp}.jsonl", engine, stages)
    kernel_file = tmp_path / "cpu_overheads_kernel_only.csv"

    main([
        "--cpu_probe_logs", str(tmp_path / "cpu_probe_dp0.jsonl"), str(tmp_path / "cpu_probe_dp1.jsonl"),
        "--num_pipeline_stages", "2",
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(tmp_path / "cpu_overheads.csv"), "--kernel_only_output_file", str(kernel_file),
    ])

    eager = validate_cpu_overhead_dataframe(pd.read_csv(tmp_path / "cpu_overheads.csv"))
    assert stage_log_path(tmp_path / "cpu_probe_dp1.jsonl", 0).name == "cpu_probe_dp1_pp0.jsonl"
    assert eager["pipeline_stage_id"].tolist() == [0, 1]
    assert eager["num_steps"].tolist() == [4, 4]
    assert not kernel_file.exists()


def test_validation_requires_both_forward_launch_columns() -> None:
    eager = cpu_overhead_tables([StageStep(0, MeasurementType.CUDA_EVENT, (1, 16, 0), EAGER_TERMS)], **IDENTITY)[
        MeasurementType.CUDA_EVENT
    ]

    with pytest.raises(ValueError, match="must appear together"):
        validate_cpu_overhead_dataframe(eager.drop(columns=["forward_launch_mean"]))


def test_validation_keys_duplicate_rows_by_pipeline_stage() -> None:
    staged = cpu_overhead_tables(
        [StageStep(stage, MeasurementType.CUDA_EVENT, (1, 16, 0), EAGER_TERMS) for stage in (0, 1)], **IDENTITY
    )[MeasurementType.CUDA_EVENT]

    assert validate_cpu_overhead_dataframe(staged)["pipeline_stage_id"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Duplicate CPU overhead rows"):
        validate_cpu_overhead_dataframe(pd.concat([staged, staged.iloc[[1]]], ignore_index=True))
    assert cpu_overhead_feature_columns(staged.drop(columns=["pipeline_stage_id"])) == [
        "batch_size", "num_prefill_tokens", "num_decode_tokens",
    ]
    assert cpu_overhead_feature_columns(staged) == [
        "batch_size", "num_prefill_tokens", "num_decode_tokens", "pipeline_stage_id",
    ]


def test_validation_requires_forward_drain_with_forward_launch() -> None:
    eager = cpu_overhead_tables([StageStep(0, MeasurementType.CUDA_EVENT, (1, 16, 0), EAGER_TERMS)], **IDENTITY)[
        MeasurementType.CUDA_EVENT
    ]

    assert validate_cpu_overhead_dataframe(eager.drop(columns=["forward_drain_mean", "forward_drain_median"]))[
        "forward_launch_median"
    ].tolist() == [0.4]
    with pytest.raises(ValueError, match="must appear together and with"):
        validate_cpu_overhead_dataframe(eager.drop(columns=["forward_drain_mean"]))
    with pytest.raises(ValueError, match="must appear together and with"):
        validate_cpu_overhead_dataframe(eager.drop(columns=["forward_launch_mean", "forward_launch_median"]))


IDLE_EDGES_MS = [0.0, 100.0, 500.0]


def _idle_steps() -> list[tuple[dict, dict]]:
    # Each step's update_end is 6.9 ms after its step_start. Step 1 starts 0.1 ms after step 0's
    # update_end and step 2 186.1 ms after step 1's.
    return [
        _single_stage_step(0, 10.0, {"a": 16}),
        _single_stage_step(1, 10.0070, {"a": 1}),
        _single_stage_step(2, 10.2, {"b": 16}),
        _single_stage_step(4, 11.2, {"c": 16}),
    ]


def test_engine_idle_buckets_count_from_the_last_iteration_or_dummy_forward() -> None:
    steps = _idle_steps()
    # An empty step at 10.5 s ends 0.4 ms later, and step 3 starts 699.6 ms after it.
    engine = [record for record, _ in steps[:3]] + [_idle_engine_record(3, 10.5), steps[3][0]]
    stages = [[forward for _, forward in steps]]

    alone = stage_overhead_terms(engine, stages, dp_allreduce_ms=None, engine_idle_edges_ms=IDLE_EDGES_MS)
    # A dummy forward that ends during step 0 leaves step 1's idle time to step 0's update_end; one
    # that ends 50 ms before step 3 ends the engine's idle time there.
    with_dummies = stage_overhead_terms(
        engine, stages, dp_allreduce_ms=None, engine_idle_edges_ms=IDLE_EDGES_MS, dummy_ends=[10.003, 11.15]
    )

    # An engine's first step carries its one-time start costs and is left out; its forward still
    # ends where the next step's dispatch can start.
    assert [step.engine_idle_ms for step in alone] == [0.0, 100.0, 500.0]
    assert [step.engine_idle_ms for step in with_dummies] == [0.0, 100.0, 0.0]
    assert [step.terms for step in with_dummies] == [step.terms for step in alone]
    assert [step.terms for step in alone] == [
        step.terms for step in stage_overhead_terms(engine, stages, dp_allreduce_ms=None)[1:]
    ]


def test_engine_idle_time_counts_from_the_latest_update_of_any_earlier_step() -> None:
    # A batch-queue step updates its batch's output after later steps start: step 0's update ends at
    # 10.5 s, after the empty step 1 ends, so step 2 at 10.3 s follows no idle time.
    first, second = _single_stage_step(0, 10.0, {"a": 16}), _single_stage_step(2, 10.3, {"a": 1})
    first[0]["update_end"] = 10.5
    engine = [first[0], _idle_engine_record(1, 10.001), second[0]]

    out = stage_overhead_terms(engine, [[first[1], second[1]]], dp_allreduce_ms=None,
                               engine_idle_edges_ms=IDLE_EDGES_MS)

    assert [step.engine_idle_ms for step in out] == [0.0]


def test_tables_key_rows_by_engine_idle_bucket() -> None:
    steps = [
        StageStep(0, MeasurementType.CUDA_EVENT, (1, 16, 0), {**EAGER_TERMS, "schedule": s}, idle)
        for s, idle in ((0.1, 0.0), (0.3, 500.0), (0.2, 0.0))
    ]

    eager = cpu_overhead_tables(steps, **IDENTITY)[MeasurementType.CUDA_EVENT]

    assert eager["engine_idle_ms"].tolist() == [0.0, 500.0]
    assert eager["schedule_median"].tolist() == pytest.approx([0.15, 0.3])
    assert eager["num_steps"].tolist() == [2, 1]
    assert cpu_overhead_feature_columns(eager) == [
        "batch_size", "num_prefill_tokens", "num_decode_tokens", "pipeline_stage_id", "engine_idle_ms",
    ]
    with pytest.raises(ValueError, match="lowest engine_idle_ms bucket edge must be 0"):
        validate_cpu_overhead_dataframe(eager.iloc[[1]])


def _write_dp_engines(tmp_path, dummy_pass_ends: list[float]) -> list[str]:
    # Engine 0 runs the idle steps; engine 1 runs one step and a dummy forward for each of the
    # given ends, logged with the other DP placement records.
    steps = _idle_steps()
    _write_engine(tmp_path / "cpu_probe_dp0.jsonl", [r for r, _ in steps], [[f for _, f in steps]])
    step1 = _single_stage_step(0, 10.0, {"d": 16})
    _write_engine(tmp_path / "cpu_probe_dp1.jsonl", [step1[0]], [[step1[1]]])
    (tmp_path / "dp_placement").mkdir()
    _write_log(tmp_path / "dp_placement/placement.jsonl", [
        {"kind": "engine_iteration", "engine": 0, "monotonic": 11.199},
        *({"kind": "dummy_pass", "engine": 0, "start_monotonic": end - 0.005, "monotonic": end}
          for end in dummy_pass_ends),
    ])
    return [str(tmp_path / "cpu_probe_dp0.jsonl"), str(tmp_path / "cpu_probe_dp1.jsonl")]


def _cli_args(tmp_path, logs: list[str], edges: list[str]) -> list[str]:
    return [
        "--cpu_probe_logs", *logs, "--engine_idle_edges_ms", *edges,
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(tmp_path / "cpu_overheads.csv"),
        "--kernel_only_output_file", str(tmp_path / "cpu_overheads_kernel_only.csv"),
    ]


def test_cli_buckets_dp_engines_by_engine_idle_with_their_dummy_forwards(tmp_path) -> None:
    logs = _write_dp_engines(tmp_path, [11.15])

    main(_cli_args(tmp_path, logs, ["0", "100", "500"]))

    eager = validate_cpu_overhead_dataframe(pd.read_csv(tmp_path / "cpu_overheads.csv"))
    columns = ["batch_size", "num_prefill_tokens", "num_decode_tokens", "engine_idle_ms"]
    # Engine 0's steps 2 and 3 fall in the 100 and 0 buckets; both engines' first steps are left out.
    assert eager[columns].values.tolist() == [[1, 0, 1, 0], [1, 16, 0, 0], [1, 16, 0, 100]]
    assert eager["num_steps"].tolist() == [1, 1, 1]


def test_cli_rejects_dp_idle_buckets_without_dummy_forwards_and_edges_not_from_0(tmp_path) -> None:
    logs = _write_dp_engines(tmp_path, [])

    with pytest.raises(ValueError, match="no dummy_pass record"):
        main(_cli_args(tmp_path, logs, ["0", "100", "500"]))
    with pytest.raises(SystemExit):
        main(_cli_args(tmp_path, logs, ["50", "100"]))
    with pytest.raises(SystemExit):
        main(_cli_args(tmp_path, logs, ["0", "500", "100"]))
