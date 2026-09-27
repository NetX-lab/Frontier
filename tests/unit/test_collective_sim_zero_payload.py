"""Hold the zero-payload boundary between Frontier and the collective-sim backend.

`moe_operator_times` computes `data_size_bytes = embedding_dim * 2 * routed_tokens`
and hands the result to `predict_all_to_all`, so an expert-parallel lane that
routes no token in a step asks for an empty transfer. An empty transfer is still
a synchronization point, and `_validate_data_size` accepts it, so the request
reaches the collective-sim runner and must come back as a latency rather than a
backend failure, whether the group stays inside one server or crosses servers.

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


def _backend(tmp_path, *, servers, gpus_per_server, attn_tp, attn_dp, **config):
    from frontier.cc_backend.backends.collective_sim_cc_backend import (
        CollectiveSimCCBackend,
    )

    config = CollectiveSimCCBackendConfig(
        cluster_servers=servers,
        cluster_gpus_per_server=gpus_per_server,
        parallel_tp=attn_tp,
        parallel_cp=1,
        parallel_dp=attn_dp,
        parallel_ep=1,
        runtime_num_replicas=1,
        runtime_num_pipeline_stages=1,
        runtime_attn_tensor_parallel_size=attn_tp,
        runtime_attn_dp=attn_dp,
        runtime_moe_tensor_parallel_size=1,
        runtime_moe_expert_parallel_size=attn_tp * attn_dp,
        runner_out_dir=str(tmp_path / "runner"),
        **config,
    )
    return CollectiveSimCCBackend(
        config=config,
        cluster_type=ClusterType.MONOLITHIC,
        device_type="h100_dgx",
        network_device="h100_dgx",
        num_devices=attn_tp * attn_dp,
    )


def test_an_empty_all_to_all_keeps_its_synchronization_latency(tmp_path):
    """A single-server pod priced with the analytic NVLink intra-server model."""
    backend = _backend(
        tmp_path,
        servers=1,
        gpus_per_server=GPUS_PER_SERVER,
        attn_tp=ATTN_TENSOR_PARALLEL_SIZE,
        attn_dp=ATTN_DATA_PARALLEL_SIZE,
        intra_server_model="nvlink_analytic",
        nvlink_latency_us=NVLINK_LATENCY_US,
    )

    predicted_ms = backend.predict_all_to_all(
        data_size_bytes=0,
        num_devices=EXPERT_PARALLEL_SIZE,
        comm_domain="EP",
    )

    assert predicted_ms == pytest.approx(SYNCHRONIZATION_MS)


# Pods that span two servers with the default backend models, so the exchange
# goes through the simulated network. EP=16 needs more than one pairwise phase
# and EP=8 on 4-GPU servers needs one.
@pytest.mark.parametrize(
    "gpus_per_server, attn_tp",
    [pytest.param(8, 8, id="ep16_two_phases"), pytest.param(4, 4, id="ep8_one_phase")],
)
def test_an_empty_all_to_all_across_servers_synchronizes_through_the_network(
    tmp_path, gpus_per_server, attn_tp
):
    backend = _backend(
        tmp_path,
        servers=2,
        gpus_per_server=gpus_per_server,
        attn_tp=attn_tp,
        attn_dp=2,
    )
    expert_parallel_size = 2 * gpus_per_server

    empty_ms = backend.predict_all_to_all(
        data_size_bytes=0, num_devices=expert_parallel_size, comm_domain="EP"
    )
    one_byte_ms = backend.predict_all_to_all(
        data_size_bytes=1, num_devices=expert_parallel_size, comm_domain="EP"
    )

    # The runner rounds each peer's share up to whole bytes, so an empty
    # exchange costs what the smallest non-empty one does, and not nothing.
    assert empty_ms > 0
    assert empty_ms == pytest.approx(one_byte_ms)
