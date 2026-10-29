"""Admission and predictor binding of the measured PREFILL DP host cadence."""

import copy
import json
from types import SimpleNamespace

import pytest

from frontier.config import MetricsConfig, RandomForrestExecutionTimePredictorConfig, VllmV1SchedulerConfig, global_vars
from frontier.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor
from frontier.execution_time_predictor.dp_dummy_step_host_time import DPDummyStepHostTime
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.types import ClusterType, MeasurementType
from tests.unit.test_kv_block_zeroing_host_time import _replica

PROFILE = {
    "schema": "dp_dummy_step_host_time.v1", "admission": "admitted",
    "provenance": {"job_id": "cpu-fixture", "statistic": "mean"},
    "model_architecture_profile": "step5", "device": "h200",
    "decode_cuda_graph_mode": "full_and_piecewise",
    "topology": {"attention_tp": 8, "attention_dp": 4, "moe_ep": 32, "pipeline_stages": 1},
    "roles": {"prefill": {"group_cadence_ms": 6.307, "samples": 4212}},
}


@pytest.fixture(autouse=True)
def graph_mode(monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")


def load(tmp_path, profile=None, *, role=ClusterType.PREFILL, replica=None, skip=True):
    path = tmp_path / "cadence.json"
    path.write_text(json.dumps(PROFILE if profile is None else profile))
    return DPDummyStepHostTime.from_file(
        path, replica_config=replica or _replica(4), cluster_type=role,
        predictor_config=SimpleNamespace(skip_cpu_overhead_modeling=skip),
    )


@pytest.mark.parametrize("admission", ["admitted", "exploratory"])
def test_valid_prefill_profile_and_shared_decode_option(tmp_path, admission):
    profile = dict(PROFILE, admission=admission)
    result = load(tmp_path, profile)
    assert type(result.group_cadence_ms) is float and result.group_cadence_ms == 6.307
    assert load(tmp_path, profile, role=ClusterType.DECODE) is None
    getter = BaseExecutionTimePredictor.get_dp_dummy_step_cadence_ms
    assert getter(SimpleNamespace(_dp_dummy_step_host_time=result)) == 6.307
    assert getter(SimpleNamespace(_dp_dummy_step_host_time=None)) is None
    assert getter(SimpleNamespace()) is None  # Custom predictors may omit Base initialization.
    # DECODE does not attempt to read a PREFILL-only shared file.
    assert DPDummyStepHostTime.from_file(
        tmp_path / "missing", replica_config=_replica(4), cluster_type=ClusterType.DECODE,
        predictor_config=SimpleNamespace(skip_cpu_overhead_modeling=True),
    ) is None


@pytest.mark.parametrize("role", [ClusterType.MONOLITHIC, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, None])
def test_unsupported_role(tmp_path, role):
    with pytest.raises(ValueError, match="supports PREFILL"):
        load(tmp_path, role=role)


@pytest.mark.parametrize("path,value,message", [
    (("schema",), "wrong", "Unsupported"),
    (("schema",), None, "Incomplete"),
    (("provenance",), None, "Incomplete"),
    (("provenance",), {}, "provenance"),
    (("provenance",), [], "provenance"),
    (("admission",), None, "Incomplete"),
    (("admission",), "rejected", "admission"),
    (("admission",), [], "admission"),
    (("model_architecture_profile",), "step3", "model_architecture_profile"),
    (("device",), "h800", "device"),
    (("decode_cuda_graph_mode",), "none", "decode_cuda_graph_mode"),
    (("topology", "attention_tp"), 4, "topology"),
    (("topology", "attention_dp"), 2, "topology"),
    (("topology", "moe_ep"), 16, "topology"),
    (("topology", "pipeline_stages"), 2, "topology"),
    (("topology",), None, "Incomplete"),
    (("roles", "decode"), {}, "exactly prefill"),
    (("roles", "prefill"), None, "exactly prefill"),
    (("roles", "prefill", "group_cadence_ms"), None, "Incomplete"),
    (("roles", "prefill", "group_cadence_ms"), 0, "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), -1, "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), True, "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), "6.307", "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), float("inf"), "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), float("-inf"), "finite positive"),
    (("roles", "prefill", "group_cadence_ms"), float("nan"), "finite positive"),
])
def test_rejects_bad_profile(tmp_path, path, value, message):
    profile = copy.deepcopy(PROFILE)
    target = profile
    for field in path[:-1]:
        target = target[field]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        load(tmp_path, profile)


@pytest.mark.parametrize("dp,pp", [(1, 1), (4, 2)])
def test_rejects_matching_but_unsupported_topology(tmp_path, dp, pp):
    replica = _replica(dp)
    replica.num_pipeline_stages = pp
    profile = copy.deepcopy(PROFILE)
    profile["topology"].update(attention_dp=dp, moe_ep=8 * dp, pipeline_stages=pp)
    with pytest.raises(ValueError, match="pipeline_stages == 1 and attention_dp > 1"):
        load(tmp_path, profile, replica=replica)


def test_cpu_overhead_cannot_be_charged_twice(tmp_path):
    with pytest.raises(ValueError, match="skip_cpu_overhead_modeling"):
        load(tmp_path, skip=False)


def test_predictor_requires_measured_step_forward_agreement(tmp_path):
    from frontier.execution_time_predictor.execution_time_predictor_registry import ExecutionTimePredictorRegistry

    config = RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True, dp_dummy_step_host_profile="missing")
    with pytest.raises(ValueError, match="not dummy mode"):
        ExecutionTimePredictorRegistry.get(
            config.get_type(), predictor_config=config, replica_config=_replica(4),
            replica_scheduler_config=VllmV1SchedulerConfig(),
            metrics_config=MetricsConfig(write_metrics=False, cache_dir=str(tmp_path)),
            cluster_type=ClusterType.PREFILL,
        )


def test_predictor_loads_profile_and_leaves_decode_unpaced(tmp_path, monkeypatch):
    # Exercise Base initialization without training operator models in this loader test.
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor

    path = tmp_path / "cadence.json"
    path.write_text(json.dumps(PROFILE))
    monkeypatch.setattr(SklearnExecutionTimePredictor, "_initialize_normal_mode", lambda self: None)
    monkeypatch.setattr(BaseExecutionTimePredictor, "__abstractmethods__", frozenset())
    for role, expected in [(ClusterType.PREFILL, 6.307), (ClusterType.DECODE, None)]:
        predictor = object.__new__(BaseExecutionTimePredictor)
        predictor._cluster_type = role
        predictor._initialize_normal_mode = lambda: None
        BaseExecutionTimePredictor.__init__(
            predictor, RandomForrestExecutionTimePredictorConfig(dp_dummy_step_host_profile=str(path)),
            _replica(4), VllmV1SchedulerConfig(), MetricsConfig(write_metrics=False, cache_dir=str(tmp_path)),
        )
        assert predictor.get_dp_dummy_step_cadence_ms() == expected


def test_predictor_bundle_identity_binds_cadence_file(tmp_path):
    from frontier.execution_time_predictor.cache_io import dataset_fingerprint

    path = tmp_path / "cadence.json"
    path.write_text(json.dumps(PROFILE))
    identities = []
    model = SimpleNamespace(_frontier_measurement_identity=json.dumps({"evidence_kind": "cpu_fixture"}))
    manager = SimpleNamespace(
        _cluster_configs={ClusterType.PREFILL: SimpleNamespace(
            replica_config=_replica(4), execution_time_predictor_config=SimpleNamespace(dp_dummy_step_host_profile=str(path)),
        )},
        _get_measurement_types_for_cluster=lambda *args: [MeasurementType.CUDA_EVENT],
        _measurement_family_name=lambda _: "eager",
        _models_view_for_family=lambda *args: {"fixture": model},
        publish_registered_model_bundle=lambda identity, *args: identities.append(identity) or {"identity": identity},
        _step5_model_bundles={},
    )
    ExecutionTimePredictionModelManager._publish_step5_model_bundles(manager)
    assert identities[0]["input_bindings"]["dp_dummy_step_host_profile"] == {
        "path": str(path), "sha256": dataset_fingerprint(path),
    }
    first = identities[0]["input_bindings"]["dp_dummy_step_host_profile"]["sha256"]
    path.write_text(json.dumps(dict(PROFILE, admission="exploratory")))
    ExecutionTimePredictionModelManager._publish_step5_model_bundles(manager)
    assert identities[1]["input_bindings"]["dp_dummy_step_host_profile"]["sha256"] != first
