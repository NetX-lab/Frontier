"""CPU-overhead CSVs from vLLM CPU-probe logs."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from frontier.profiling.cpu_overhead.validation import validate_cpu_overhead_dataframe
from frontier.profiling.cpu_overhead.vllm_cpu_probe import (
    cpu_overhead_tables,
    main,
    step_overhead_terms,
)
from frontier.types import MeasurementType

IDENTITY = dict(model_name="tiny", tensor_parallel_degree=1, profiling_precision="BF16", scheduling_mode="sync")


def _step(step: int, start: float, scheduled: dict[str, int], forward_device_ms: float | None = 4.0) -> dict:
    # Host phases in seconds after step_start: schedule 0.1 ms, dispatch plus preprocess 0.5 ms,
    # forward launch plus sampling 6.0 ms, bookkeeping to update_end 0.3 ms.
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


def test_step_terms_cover_the_period_outside_the_forward_device_time() -> None:
    records = [
        _step(0, 10.0, {"a": 16}),
        _step(1, 10.0070, {"a": 1}),
        _step(2, 10.0140, {"b": 16}),
        _step(3, 10.5, {}),
    ]

    steps = step_overhead_terms(records)

    assert [identity for identity, _ in steps] == [(1, 16, 0), (1, 0, 1), (1, 16, 0)]
    first = steps[0][1]
    assert first["schedule"] == pytest.approx(0.1)
    assert first["prepare_inputs_e2e"] == pytest.approx(0.5)
    assert first["sampler_e2e"] == pytest.approx(6.0 - 4.0)
    assert first["process_model_outputs"] == pytest.approx(7.0 - 6.6)
    assert sum(first.values()) == pytest.approx(7.0 - 4.0)
    # Step 1 is followed by a step of another request, step 2 by an empty step:
    # the engine may have waited, so the term ends at update_end.
    assert steps[1][1]["process_model_outputs"] == pytest.approx(0.3)
    assert steps[2][1]["process_model_outputs"] == pytest.approx(0.3)


def test_step_terms_require_the_probe_device_times() -> None:
    with pytest.raises(ValueError, match="step 0 has no forward_device_ms"):
        step_overhead_terms([_step(0, 10.0, {"a": 16}, forward_device_ms=None)])


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
