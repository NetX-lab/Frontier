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
    cpu_overhead_tables,
    main,
    stage_cpu_overhead_tables,
    stage_log_path,
    stage_overhead_terms,
    step_overhead_terms,
)
from frontier.types import MeasurementType

IDENTITY = dict(model_name="tiny", tensor_parallel_degree=1, profiling_precision="BF16", scheduling_mode="sync")


def _step(step: int, start: float, scheduled: dict[str, int], forward_device_ms: float | None = 4.0) -> dict:
    # Host phases in seconds after step_start: schedule 0.1 ms, dispatch plus preprocess 0.5 ms,
    # forward launch 0.4 ms, forward launch plus sampling 6.0 ms, bookkeeping to update_end 0.3 ms.
    record = {
        "step": step,
        "step_start": start,
        "schedule_end": start + 0.0001,
        "execute_start": start + 0.0002,
        "preprocess_end": start + 0.0006,
        "forward_end": start + 0.0010,
        "sample_end": start + 0.0066,
        "bookkeep_end": start + 0.0067,
        "execute_return": start + 0.0068,
        "execute_end": start + 0.0068,
        "update_end": start + 0.0069,
        "num_scheduled_tokens": scheduled,
    }
    if forward_device_ms is not None:
        record["forward_device_ms"] = forward_device_ms
        record["sample_device_ms"] = 0.2
    return record


PERIOD_TERMS = ("schedule", "prepare_inputs_e2e", "sampler_e2e", "process_model_outputs")


def test_step_terms_cover_the_period_outside_the_forward_device_time() -> None:
    records = [
        _step(0, 10.0, {"a": 16}),
        _step(1, 10.0070, {"a": 1}),
        _step(2, 10.0140, {"b": 16}),
        _step(3, 10.5, {}),
    ]

    steps = step_overhead_terms(records, largest_graph_batch=0)

    assert [identity for identity, _ in steps] == [(1, 16, 0), (1, 0, 1), (1, 16, 0)]
    first = steps[0][1]
    assert first["schedule"] == pytest.approx(0.1)
    assert first["prepare_inputs_e2e"] == pytest.approx(0.5)
    # Device-bound eager step: the 4.0 ms device stream hides the 0.4 ms launch.
    assert first["forward_launch"] == pytest.approx(0.4)
    assert first["sampler_e2e"] == pytest.approx(6.0 - 4.0)
    assert first["process_model_outputs"] == pytest.approx(7.0 - 6.6)
    assert sum(first[term] for term in PERIOD_TERMS) == pytest.approx(7.0 - 4.0)
    # Step 1 is followed by a step of another request, step 2 by an empty step:
    # the engine may have waited, so the term ends at update_end.
    assert steps[1][1]["process_model_outputs"] == pytest.approx(0.3)
    assert steps[2][1]["process_model_outputs"] == pytest.approx(0.3)


def test_host_bound_eager_step_terms_leave_out_the_launch_stream() -> None:
    steps = step_overhead_terms([_step(0, 10.0, {"a": 16}, forward_device_ms=0.25)], largest_graph_batch=0)

    terms = steps[0][1]
    assert terms["forward_launch"] == pytest.approx(0.4)
    assert terms["sampler_e2e"] == pytest.approx(6.0 - 0.4)
    assert sum(terms[term] for term in PERIOD_TERMS) == pytest.approx(6.9 - 0.4)


def test_graph_replay_terms_leave_out_the_device_time_and_publish_no_launch() -> None:
    steps = step_overhead_terms([_step(0, 10.0, {"a": 1, "b": 1}, forward_device_ms=0.25)], largest_graph_batch=4)

    identity, terms = steps[0]
    assert identity == (2, 0, 2)
    assert "forward_launch" not in terms
    assert terms["sampler_e2e"] == pytest.approx(6.0 - 0.25)


def test_step_terms_require_the_probe_device_times() -> None:
    with pytest.raises(ValueError, match="step 0 has no forward_device_ms"):
        step_overhead_terms([_step(0, 10.0, {"a": 16}, forward_device_ms=None)], largest_graph_batch=0)


def test_tables_split_graph_replayed_decode_from_eager_tuples() -> None:
    terms = {"schedule": 0.1, "prepare_inputs_e2e": 0.5, "sampler_e2e": 2.0, "process_model_outputs": 0.4}
    steps = [((3, 0, 3), terms), ((8, 0, 8), terms), ((1, 16, 0), terms), ((2, 16, 1), terms)]

    tables = cpu_overhead_tables(steps, decode_capture_sizes=[1, 2, 4], **IDENTITY)

    kernel_only = tables[MeasurementType.KERNEL_ONLY]
    eager = tables[MeasurementType.CUDA_EVENT]
    assert kernel_only[["batch_size", "num_prefill_tokens", "num_decode_tokens"]].values.tolist() == [[3, 0, 3]]
    assert eager[["batch_size", "num_prefill_tokens", "num_decode_tokens"]].values.tolist() == [
        [1, 16, 0], [2, 16, 1], [8, 0, 8],
    ]
    assert (eager["ray_comm_time_mean"] == 0.0).all()
    assert set(eager["measurement_type"]) == {MeasurementType.CUDA_EVENT.value}
    assert set(kernel_only["measurement_type"]) == {MeasurementType.KERNEL_ONLY.value}


def test_tables_take_mean_and_median_per_tuple() -> None:
    steps = [
        ((2, 0, 2), {"schedule": s, "prepare_inputs_e2e": 0.5, "sampler_e2e": 2.0, "process_model_outputs": p})
        for s, p in ((0.1, 0.4), (0.2, 0.3), (0.6, 0.8))
    ]

    tables = cpu_overhead_tables(steps, decode_capture_sizes=[8], **IDENTITY)

    row = tables[MeasurementType.KERNEL_ONLY].set_index("batch_size").loc[2]
    assert row["schedule_median"] == pytest.approx(0.2)
    assert row["schedule_mean"] == pytest.approx(0.3)
    assert row["process_model_outputs_median"] == pytest.approx(0.4)
    assert row["num_steps"] == 3


def test_cli_writes_one_csv_per_family_readable_by_the_loader_validation(tmp_path) -> None:
    decode_log = tmp_path / "decode.jsonl"
    records = [_step(i, 10.0 + 0.007 * i, {"a": 1, "b": 1}) for i in range(3)] + [_step(3, 11.0, {})]
    decode_log.write_text("".join(json.dumps(record) + "\n" for record in records))
    # A PD prefill instance runs each request for one step.
    prefill_log = tmp_path / "prefill.jsonl"
    prefill_log.write_text("".join(json.dumps(r) + "\n" for r in [_step(0, 5.0, {"a": 16}), _step(1, 5.007, {"b": 16})]))
    eager_file, kernel_file = tmp_path / "out/cpu_overheads.csv", tmp_path / "out/cpu_overheads_kernel_only.csv"

    main([
        "--cpu_probe_logs", str(prefill_log), str(decode_log),
        "--decode_cudagraph_capture_sizes", "1", "2", "4",
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(eager_file), "--kernel_only_output_file", str(kernel_file),
    ])

    eager = validate_cpu_overhead_dataframe(pd.read_csv(eager_file))
    kernel_only = validate_cpu_overhead_dataframe(pd.read_csv(kernel_file))
    assert eager[["batch_size", "num_prefill_tokens", "num_decode_tokens"]].values.tolist() == [[1, 16, 0]]
    assert eager.loc[0, "num_steps"] == 2
    assert eager.loc[0, "process_model_outputs_median"] == pytest.approx(0.3)
    assert eager.loc[0, "forward_launch_median"] == pytest.approx(0.4)
    assert "forward_launch_median" not in kernel_only.columns
    assert kernel_only[["batch_size", "num_prefill_tokens", "num_decode_tokens"]].values.tolist() == [[2, 0, 2]]
    assert kernel_only.loc[0, "num_steps"] == 3


def test_cli_writes_no_kernel_only_file_for_an_eager_engine(tmp_path) -> None:
    log = tmp_path / "server.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in [_step(0, 1.0, {"a": 1}), _step(1, 1.007, {"a": 1})]))
    kernel_file = tmp_path / "cpu_overheads_kernel_only.csv"

    main([
        "--cpu_probe_logs", str(log), "--decode_cudagraph_capture_sizes",
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(tmp_path / "cpu_overheads.csv"), "--kernel_only_output_file", str(kernel_file),
    ])

    assert (tmp_path / "cpu_overheads.csv").exists()
    assert not kernel_file.exists()


def test_validation_requires_both_forward_launch_columns() -> None:
    terms = {"schedule": 0.1, "prepare_inputs_e2e": 0.5, "sampler_e2e": 2.0,
             "process_model_outputs": 0.4, "forward_launch": 0.4}
    eager = cpu_overhead_tables([((1, 16, 0), terms)], decode_capture_sizes=[], **IDENTITY)[MeasurementType.CUDA_EVENT]

    with pytest.raises(ValueError, match="must appear together"):
        validate_cpu_overhead_dataframe(eager.drop(columns=["forward_launch_mean"]))


def _engine_step(step: int, start: float, scheduled: dict[str, int], sample_end: float) -> dict:
    # Batch-queue engine record: schedule 0.1 ms at submission; the output stage's sampling ends
    # at sample_end and the engine's output update 0.3 ms later.
    return {
        "step": step,
        "engine_loop": "batch_queue",
        "step_start": start,
        "schedule_end": start + 0.0001,
        "sample_end": sample_end,
        "update_end": sample_end + 0.0003,
        "num_scheduled_tokens": scheduled,
    }


def _stage_forward(step: int, stage_id: int, execute_start: float, scheduled: dict[str, int], *,
                   dp_wait: float = 0.0, launch: float = 0.0004, device_ms: float = 4.0) -> dict:
    # Prepare 0.5 ms, then forward_start; the DP token-count all-reduce ends dp_wait later and the
    # host launches the forward's kernels for `launch` seconds after it.
    forward_start = execute_start + 0.0005
    return {
        "step": step,
        "pipeline_stage_id": stage_id,
        "execute_start": execute_start,
        "preprocess_end": forward_start,
        "forward_start": forward_start,
        "dp_sync_end": forward_start + dp_wait,
        "forward_end": forward_start + dp_wait + launch,
        "forward_device_ms": device_ms,
        "num_scheduled_tokens": scheduled,
    }


def _two_stage_step(*, stage0_device_ms: float = 4.0, dp_wait: float = 0.0):
    # Stage 0 starts at 10.0002; stage 1 starts at 10.0010 and its own device work takes 4 ms.
    # When stage 0's device work ends after stage 1's dp_sync_end, stage 1's host blocks inside
    # its forward until the upstream output arrives, then launches for 0.4 ms. The last stage's
    # sampling ends 0.6 ms after its device end.
    scheduled = {"a": 16}
    s0 = _stage_forward(0, 0, 10.0002, scheduled, dp_wait=dp_wait, device_ms=stage0_device_ms)
    s1 = _stage_forward(0, 1, 10.0010, scheduled, dp_wait=dp_wait)
    d0 = s0["forward_start"] + stage0_device_ms * 1e-3
    s1["forward_end"] = max(s1["dp_sync_end"], d0) + 0.0004
    d1 = max(s1["forward_start"], d0) + 4.0e-3
    engine = [_engine_step(0, 10.0, scheduled, d1 + 0.0006), _engine_step(1, 10.0001, {}, d1 + 0.0007)]
    return engine, [[s0], [s1]]


def test_stage_rows_charge_each_interval_to_the_stage_that_runs_it() -> None:
    engine, stages = _two_stage_step()

    (stage0,), (stage1,) = stage_overhead_terms(engine, stages)

    assert stage0[0] == stage1[0] == (1, 16, 0)
    assert stage0[1]["schedule"] == pytest.approx(0.1)
    assert stage1[1]["schedule"] == 0.0
    assert stage0[1]["prepare_inputs_e2e"] == stage1[1]["prepare_inputs_e2e"] == pytest.approx(0.5)
    assert stage0[1]["sampler_e2e"] == stage0[1]["process_model_outputs"] == 0.0
    assert stage1[1]["sampler_e2e"] == pytest.approx(0.6)
    assert stage1[1]["process_model_outputs"] == pytest.approx(0.3)
    assert list(stage0[1]) == list(stage1[1]) == [
        "schedule", "prepare_inputs_e2e", "sampler_e2e", "process_model_outputs", "forward_launch",
    ]


def test_stage_forward_launch_leaves_out_the_dp_token_count_wait() -> None:
    engine, stages = _two_stage_step(dp_wait=0.002)

    (stage0,), (stage1,) = stage_overhead_terms(engine, stages)

    assert stage0[1]["forward_launch"] == pytest.approx(0.4)
    assert stage1[1]["sampler_e2e"] == pytest.approx(0.6)


def test_stage_forward_launch_starts_after_the_upstream_device_end() -> None:
    # Stage 0's device runs 30 ms; stage 1's host reaches dp_sync_end long before that and
    # waits inside its forward for the upstream output.
    engine, stages = _two_stage_step(stage0_device_ms=30.0)

    (stage0,), (stage1,) = stage_overhead_terms(engine, stages)

    assert stage0[1]["forward_launch"] == pytest.approx(0.4)
    assert stage1[1]["forward_launch"] == pytest.approx(0.4)


def test_last_stage_sampler_starts_after_the_last_stage_device_end() -> None:
    engine, stages = _two_stage_step(stage0_device_ms=30.0)

    (_,), (stage1,) = stage_overhead_terms(engine, stages)

    assert stage1[1]["sampler_e2e"] == pytest.approx(0.6)


def test_stage_join_requires_matching_token_maps() -> None:
    engine, (stage0, stage1) = _two_stage_step()

    with pytest.raises(ValueError, match="pipeline stage 1 logged 0 forwards"):
        stage_overhead_terms(engine, [stage0, []])
    other = [dict(stage1[0], num_scheduled_tokens={"b": 16})]
    with pytest.raises(ValueError, match="pipeline stage 1 logged 1 forwards"):
        stage_overhead_terms(engine, [stage0, other])


def test_stage_tables_key_rows_by_pipeline_stage() -> None:
    engine, stages = _two_stage_step()

    tables = stage_cpu_overhead_tables(stage_overhead_terms(engine, stages), decode_capture_sizes=[], **IDENTITY)

    eager = tables[MeasurementType.CUDA_EVENT]
    assert list(tables) == [MeasurementType.CUDA_EVENT]
    assert eager[["pipeline_stage_id", "batch_size", "num_prefill_tokens", "num_decode_tokens"]].values.tolist() == [
        [0, 1, 16, 0], [1, 1, 16, 0],
    ]
    assert eager["sampler_e2e_median"].tolist() == pytest.approx([0.0, 0.6])


def test_cli_reads_each_engine_log_with_its_stage_logs(tmp_path) -> None:
    for dp in (0, 1):
        engine, stages = _two_stage_step(dp_wait=0.001 * dp)
        log = tmp_path / f"cpu_probe_dp{dp}.jsonl"
        log.write_text("".join(json.dumps(r) + "\n" for r in engine))
        for stage_id, records in enumerate(stages):
            stage_log_path(log, stage_id).write_text("".join(json.dumps(r) + "\n" for r in records))
    kernel_file = tmp_path / "cpu_overheads_kernel_only.csv"

    main([
        "--cpu_probe_logs", str(tmp_path / "cpu_probe_dp0.jsonl"), str(tmp_path / "cpu_probe_dp1.jsonl"),
        "--num_pipeline_stages", "2", "--decode_cudagraph_capture_sizes",
        "--model_name", "tiny", "--tensor_parallel_degree", "1",
        "--profiling_precision", "BF16", "--scheduling_mode", "sync",
        "--eager_output_file", str(tmp_path / "cpu_overheads.csv"), "--kernel_only_output_file", str(kernel_file),
    ])

    eager = validate_cpu_overhead_dataframe(pd.read_csv(tmp_path / "cpu_overheads.csv"))
    assert stage_log_path(tmp_path / "cpu_probe_dp1.jsonl", 0).name == "cpu_probe_dp1_pp0.jsonl"
    assert eager["pipeline_stage_id"].tolist() == [0, 1]
    assert eager["num_steps"].tolist() == [2, 2]
    assert not kernel_file.exists()


def test_single_stage_terms_reject_batch_queue_records() -> None:
    engine, _ = _two_stage_step()

    with pytest.raises(ValueError, match="batch-queue loop"):
        step_overhead_terms(engine, largest_graph_batch=0)


def test_validation_keys_duplicate_rows_by_pipeline_stage() -> None:
    terms = {"schedule": 0.1, "prepare_inputs_e2e": 0.5, "sampler_e2e": 2.0,
             "process_model_outputs": 0.4, "forward_launch": 0.4}
    single = cpu_overhead_tables([((1, 16, 0), terms)], decode_capture_sizes=[], **IDENTITY)[MeasurementType.CUDA_EVENT]
    staged = pd.concat([single.assign(pipeline_stage_id=0), single.assign(pipeline_stage_id=1)], ignore_index=True)

    assert validate_cpu_overhead_dataframe(staged)["pipeline_stage_id"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Duplicate CPU overhead rows"):
        validate_cpu_overhead_dataframe(pd.concat([staged, staged.iloc[[1]]], ignore_index=True))
    assert cpu_overhead_feature_columns(single) == ["batch_size", "num_prefill_tokens", "num_decode_tokens"]
    assert cpu_overhead_feature_columns(staged) == [
        "batch_size", "num_prefill_tokens", "num_decode_tokens", "pipeline_stage_id",
    ]
