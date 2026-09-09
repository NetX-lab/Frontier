from frontier.profiling.collectives.collectives_input import CollectivesInput
from frontier.profiling.collectives.main import SUPPORTED_COLLECTIVE_PRECISIONS
from frontier.profiling.utils import get_collectives_inputs


def test_single_node_collective_grid_uses_exact_worker_counts() -> None:
    inputs = get_collectives_inputs(
        num_nodes=1,
        num_workers_per_node_combinations=[1, 2, 4, 8],
        max_collective_size=1024,
        collective="all_reduce",
        total_gpus_available=8,
        precision="BF16",
    )

    assert [item.num_workers for item in inputs] == [2, 4, 8]
    assert [item.num_workers_per_node for item in inputs] == [2, 4, 8]
    assert all(item.precision == "BF16" for item in inputs)


def test_partial_node_worker_layout_is_invalid() -> None:
    collectives_input = CollectivesInput(
        num_workers=2,
        num_workers_per_node=4,
        collective_size=1024,
        collective="all_reduce",
    )

    assert not collectives_input.is_valid(total_gpus_available=8, num_nodes=1)


def test_collective_cli_exposes_supported_runtime_dtypes() -> None:
    assert SUPPORTED_COLLECTIVE_PRECISIONS == ("FP16", "BF16", "FP32")
