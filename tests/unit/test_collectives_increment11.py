"""CPU contracts for the standalone NCCL/RCCL collective profiler."""

from __future__ import annotations

import pandas as pd
import pytest


def test_collective_input_preserves_precision_and_rejects_bad_dtype() -> None:
    from frontier.profiling.collectives.collectives_input import CollectivesInput

    item = CollectivesInput(2, 2, 128, "all_reduce", precision="BF16")
    assert item.precision == "BF16"
    with pytest.raises(ValueError, match="precision"):
        CollectivesInput(2, 2, 128, "all_reduce", precision="FP4")


def test_collective_grid_propagates_precision() -> None:
    from frontier.profiling.utils import get_collectives_inputs

    inputs = get_collectives_inputs(
        num_nodes=1,
        num_workers_per_node_combinations=[1, 2, 4],
        max_collective_size=1024,
        collective="all_reduce",
        total_gpus_available=4,
        precision="FP32",
    )
    assert inputs
    assert {item.precision for item in inputs} == {"FP32"}


def test_collective_precision_to_dtype_is_explicit() -> None:
    import torch
    from frontier.profiling.collectives.main import _precision_to_dtype

    assert _precision_to_dtype("FP16") is torch.float16
    assert _precision_to_dtype("BF16") is torch.bfloat16
    assert _precision_to_dtype("FP32") is torch.float32
    with pytest.raises(ValueError, match="FP4"):
        _precision_to_dtype("FP4")


def test_collective_validity_rejects_partial_node_layout() -> None:
    from frontier.profiling.collectives.collectives_input import CollectivesInput

    item = CollectivesInput(2, 4, 128, "all_reduce")
    assert not item.is_valid(total_gpus_available=8, num_nodes=1)


def test_collective_element_size_is_used_for_byte_accounting() -> None:
    import torch
    from frontier.profiling.collectives.collectives_impl import GraphedCollective

    graph = object.__new__(GraphedCollective)
    graph._buffer = torch.empty(8, dtype=torch.bfloat16)
    assert graph.element_size == 2
    graph._buffer = torch.empty(8, dtype=torch.float32)
    assert graph.element_size == 4


def test_collective_output_writer_emits_flat_schema(tmp_path) -> None:
    from frontier.profiling.collectives.main import _write_results

    class Args:
        output_dir = str(tmp_path / "collective" / "2026-09-14_000000")
        collective = "all_reduce"
        precision = "BF16"

    results = [
        {
            "time_stats": {"all_reduce": {"median": 0.25}},
            "rank": 0,
            "num_workers": 2,
            "size": 256,
            "collective": "all_reduce",
        }
    ]
    output = _write_results(results, Args)
    frame = pd.read_csv(output)
    assert frame.loc[0, "profiling_precision"] == "BF16"
    assert frame.loc[0, "time_stats.all_reduce.median"] == pytest.approx(0.25)
    assert frame.loc[0, "size"] == 256
