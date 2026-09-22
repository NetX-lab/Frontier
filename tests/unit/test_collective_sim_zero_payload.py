"""Hold the zero-payload boundary between Frontier and the collective-sim backend.

`moe_operator_times` computes `data_size_bytes = embedding_dim * 2 * routed_tokens`
and hands the result to `predict_all_to_all`, so an expert-parallel lane that
routes no token in a step asks for an empty transfer. `predict_reduce_scatter`
floor-divides by the device count and reaches zero the same way. An empty
transfer is still a synchronization point, and `_validate_data_size` accepts it,
so the request reaches the collective-sim runner and must come back as a latency
rather than a backend failure.

The runner-side and schema-side repairs live in the collective-sim submodule
(`fwyc0573/frontier-htsim`); this module covers the Frontier call path that
depends on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.types import ClusterType


BACKEND_ROOT = (
    Path(__file__).resolve().parents[2]
    / "frontier/cc_backend/backends/collective-sim"
)
if not (BACKEND_ROOT / "sim/datacenter/htsim_ndp").is_file():
    pytest.skip(
        "The optional collective-sim backend must be initialized and built: "
        "git submodule update --init frontier/cc_backend/backends/collective-sim "
        "&& make -C frontier/cc_backend/backends/collective-sim/sim",
        allow_module_level=True,
    )

# The canonical mapping from AGENTS.md: one complete pod, TP=4 x DP=2 for
# attention and a single EP=8 collective for the routed experts.
GPUS_PER_SERVER = 8
ATTN_TENSOR_PARALLEL_SIZE = 4
ATTN_DATA_PARALLEL_SIZE = 2
EXPERT_PARALLEL_SIZE = 8

# Eight NVLink-connected ranks exchange over seven hops, each costing
# nvlink_latency_us, and that term does not depend on the payload.
NVLINK_LATENCY_US = 0.5
SYNCHRONIZATION_MS = (EXPERT_PARALLEL_SIZE - 1) * NVLINK_LATENCY_US / 1000.0


@pytest.fixture
def backend(tmp_path):
    """A single-server MoE pod priced with the analytic NVLink intra-server model."""
    from frontier.cc_backend.backends.collective_sim_cc_backend import (
        CollectiveSimCCBackend,
    )

    config = CollectiveSimCCBackendConfig(
        cluster_servers=1,
        cluster_gpus_per_server=GPUS_PER_SERVER,
        parallel_tp=ATTN_TENSOR_PARALLEL_SIZE,
        parallel_cp=1,
        parallel_dp=ATTN_DATA_PARALLEL_SIZE,
        parallel_ep=1,
        runtime_num_replicas=1,
        runtime_num_pipeline_stages=1,
        runtime_attn_tensor_parallel_size=ATTN_TENSOR_PARALLEL_SIZE,
        runtime_attn_dp=ATTN_DATA_PARALLEL_SIZE,
        runtime_moe_tensor_parallel_size=1,
        runtime_moe_expert_parallel_size=EXPERT_PARALLEL_SIZE,
        intra_server_model="nvlink_analytic",
        nvlink_latency_us=NVLINK_LATENCY_US,
        runner_out_dir=str(tmp_path / "runner"),
    )
    return CollectiveSimCCBackend(
        config=config,
        cluster_type=ClusterType.MONOLITHIC,
        device_type="h100_dgx",
        network_device="h100_dgx",
        num_devices=GPUS_PER_SERVER,
    )


def test_an_empty_all_to_all_keeps_its_synchronization_latency(backend):
    predicted_ms = backend.predict_all_to_all(
        data_size_bytes=0,
        num_devices=EXPERT_PARALLEL_SIZE,
        comm_domain="EP",
    )

    assert predicted_ms == pytest.approx(SYNCHRONIZATION_MS)


def test_an_empty_reduce_scatter_keeps_its_synchronization_latency(backend):
    # A payload smaller than the device count floor-divides to zero, which is how
    # this collective reaches an empty transfer.
    predicted_ms = backend.predict_reduce_scatter(
        data_size_bytes=EXPERT_PARALLEL_SIZE - 1,
        num_devices=EXPERT_PARALLEL_SIZE,
        comm_domain="EP",
    )

    assert predicted_ms == pytest.approx(SYNCHRONIZATION_MS)


def test_a_populated_all_to_all_costs_more_than_an_empty_one(backend):
    empty_ms = backend.predict_all_to_all(
        data_size_bytes=0,
        num_devices=EXPERT_PARALLEL_SIZE,
        comm_domain="EP",
    )
    populated_ms = backend.predict_all_to_all(
        data_size_bytes=1 << 20,
        num_devices=EXPERT_PARALLEL_SIZE,
        comm_domain="EP",
    )

    assert populated_ms > empty_ms


def test_a_negative_payload_is_still_rejected(backend):
    with pytest.raises(ValueError, match="data_size_bytes must be non-negative"):
        backend.predict_all_to_all(
            data_size_bytes=-1,
            num_devices=EXPERT_PARALLEL_SIZE,
            comm_domain="EP",
        )
