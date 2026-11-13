"""Loader and lookup contracts of the measured KV block zeroing host-time profile."""

import copy
import json
from types import SimpleNamespace

import pytest

from frontier.config import ReplicaConfig, global_vars
from frontier.entities import Batch, Request
from frontier.entities.batch import KVCacheBlockZeroing
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.execution_time_predictor.kv_block_zeroing_host_time import KVBlockZeroingHostTime
from frontier.types import ClusterType

PROFILE = {
    "schema": "kv_block_zeroing_host_time.v1",
    "provenance": {"job": "fixture", "commit": "fixture"},
    "model_architecture_profile": "step5",
    "device": "h200",
    "decode_cuda_graph_mode": "full_and_piecewise",
    "topology": {"attention_tp": 8, "attention_dp": 2, "moe_ep": 16, "pipeline_stages": 1},
    "roles": {
        "prefill": {"time_ms": 7.5},
        "decode": {"by_num_requests": [[1, 2.0], [9, 6.0], [17, 8.0]]},
    },
}


def _replica(attn_dp=2):
    return ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device="h200", network_device="h200_dgx",
        attn_tensor_parallel_size=8, attn_dp=attn_dp, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8 * attn_dp, attention_backend="STEP5_DSA", kv_cache_dtype="auto",
    )


@pytest.fixture(autouse=True)
def _graph_mode(monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")


def _load(tmp_path, profile=PROFILE, *, cluster_type=ClusterType.DECODE, replica=None,
          skip_cpu_overhead_modeling=True):
    path = tmp_path / "host_time.json"
    path.write_text(json.dumps(profile))
    return KVBlockZeroingHostTime.from_file(
        path, replica_config=replica or _replica(), cluster_type=cluster_type,
        predictor_config=SimpleNamespace(skip_cpu_overhead_modeling=skip_cpu_overhead_modeling),
    )


def test_scalar_and_table_entries_by_role(tmp_path):
    assert _load(tmp_path, cluster_type=ClusterType.PREFILL).time_ms(64) == 7.5
    decode = _load(tmp_path)
    assert decode.time_ms(1) == 2.0
    assert decode.time_ms(5) == pytest.approx(4.0)
    assert decode.time_ms(9) == 6.0
    assert decode.time_ms(13) == pytest.approx(7.0)
    for count in (0, 18):
        with pytest.raises(ValueError, match="covers 1-17 requests"):
            decode.time_ms(count)


def _changed(path, value):
    profile = copy.deepcopy(PROFILE)
    *parents, field = path
    target = profile
    for parent in parents:
        target = target[parent]
    if value is None:
        del target[field]
    else:
        target[field] = value
    return profile


@pytest.mark.parametrize("path,value,message", [
    (("schema",), "kv_block_zeroing_host_time.v0", "Unsupported"),
    (("provenance",), {}, "provenance"),
    (("model_architecture_profile",), "step3", "model_architecture_profile"),
    (("device",), "h800", "device"),
    (("decode_cuda_graph_mode",), "none", "decode_cuda_graph_mode"),
    (("topology", "attention_tp"), 4, "topology"),
    (("topology", "attention_dp"), 4, "topology"),
    (("topology", "moe_ep"), 32, "topology"),
    (("topology", "pipeline_stages"), 2, "topology"),
    (("roles", "decode"), None, "no decode entry"),
    (("roles", "decode", "time_ms"), 1.0, "needs time_ms or by_num_requests"),
    (("roles", "decode", "by_num_requests"), None, "needs time_ms or by_num_requests"),
    (("roles", "decode", "by_num_requests"), [[1, 2.0]], "at least two"),
    (("roles", "decode", "by_num_requests"), [[9, 2.0], [1, 6.0]], "strictly increasing"),
    (("roles", "decode", "by_num_requests"), [[1, 2.0], [1, 6.0]], "strictly increasing"),
    (("roles", "decode", "by_num_requests"), [[1, 2.0], [9, -1.0]], "non-negative"),
    (("roles", "decode", "by_num_requests"), [[1, 2.0], [9, True]], "non-negative"),
    (("roles", "decode", "by_num_requests"), [[1, 2.0], [9]], "unpack"),
    (("topology",), None, "Incomplete"),
])
def test_rejects_profiles_measured_on_another_scope(tmp_path, path, value, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _changed(path, value))


def test_rejects_a_dp2_profile_on_a_dp4_run(tmp_path):
    with pytest.raises(ValueError, match="topology"):
        _load(tmp_path, replica=_replica(attn_dp=4))


@pytest.mark.parametrize("cluster_type", [ClusterType.MONOLITHIC, ClusterType.DECODE_ATTN, None])
def test_rejects_roles_without_a_measured_entry(tmp_path, cluster_type):
    with pytest.raises(ValueError, match="supports PREFILL and DECODE"):
        _load(tmp_path, cluster_type=cluster_type)


def test_rejects_cpu_overhead_modeling(tmp_path):
    with pytest.raises(ValueError, match="skip_cpu_overhead_modeling"):
        _load(tmp_path, skip_cpu_overhead_modeling=False)


def test_predictor_charges_only_resetting_batches(tmp_path):
    predictor = SimpleNamespace(_kv_block_zeroing_host_time=_load(tmp_path))
    host_time = BaseExecutionTimePredictor.get_kv_block_zeroing_host_time_ms
    batch = Batch(0, [Request(0.0, 8, 4) for _ in range(5)], [8] * 5, is_moe=True)
    assert host_time(predictor, batch) == 0.0
    batch.cache_block_zeroing = KVCacheBlockZeroing((3, 4), 1024)
    assert host_time(predictor, batch) == pytest.approx(4.0)
    assert host_time(SimpleNamespace(_kv_block_zeroing_host_time=None), batch) == 0.0


def test_dummy_mode_predictor_rejects_the_profile():
    from frontier.config import MetricsConfig
    from frontier.config.execution_time_predictor_config import (
        RandomForrestExecutionTimePredictorConfig,
    )
    from frontier.config.replica_scheduler_config import VllmV1SchedulerConfig
    from frontier.execution_time_predictor.execution_time_predictor_registry import (
        ExecutionTimePredictorRegistry,
    )

    config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=True, kv_block_zeroing_host_profile="/nonexistent/host_time.json",
    )
    with pytest.raises(ValueError, match="not dummy mode"):
        ExecutionTimePredictorRegistry.get(
            config.get_type(), predictor_config=config, replica_config=_replica(),
            replica_scheduler_config=VllmV1SchedulerConfig(),
            metrics_config=MetricsConfig(write_metrics=False), cluster_type=ClusterType.DECODE,
        )
