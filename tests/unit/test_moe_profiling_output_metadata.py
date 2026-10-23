"""Unit tests for MoE profiling output metadata contract."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pandas as pd
import pytest

from frontier.profiling.moe import main as moe_main


def test_worker_forwards_the_same_serialized_cell_without_legacy_scalar_unpacking(monkeypatch):
    from frontier.profiling.moe.inputs import MoESourceFrame, MoERoutingProfileInput, MoERoutingProfileCell
    cell = MoERoutingProfileCell(MoERoutingProfileInput("actual_forward", (MoESourceFrame(0, (4,), (6128,), 8, "standalone_legacy"),), 8), "source_local", 0)
    calls = []
    def profile(**kwargs):
        calls.append(kwargs)
        return {"one_cell": True}
    monkeypatch.setattr(moe_main, "_ensure_torch_available", lambda: SimpleNamespace(cuda=SimpleNamespace(set_device=lambda *args: None)))
    monkeypatch.setattr(moe_main, "_get_moe_wrapper_class", lambda: lambda **kwargs: SimpleNamespace(profile=profile))
    monkeypatch.setattr(moe_main, "ModelConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(moe_main, "_CUDA_INITIALIZED", False)
    monkeypatch.setattr(moe_main, "_CUDA_GPU_ID", None)
    wrapper = dict(num_tensor_parallel_workers=1, expert_parallel_size=16, profile_method="cuda_event",
                   output_dir="unused", use_vllm_kernel=True, routed_layer_group=None)
    assert moe_main._worker_profile_task((0, 0, {}, wrapper, {"routing_cell": cell.to_json()})) == {"one_cell": True}
    assert calls[0]["routing_cell"] == cell.to_json()


def test_attach_moe_output_metadata_writes_model_architecture_profile() -> None:
    df = pd.DataFrame(
        [
            {
                "num_tokens": 1,
                "time_stats.moe_grouped_gemm.mean": 2.5,
            }
        ]
    )

    output = moe_main._attach_moe_output_metadata(  # pylint: disable=protected-access
        df,
        precision_str="FP16",
        model_arch="generic",
        model_architecture_profile="step3_text",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
    )

    assert output.loc[0, "profiling_precision"] == "FP16"
    assert output.loc[0, "model_arch"] == "generic"
    assert output.loc[0, "model_architecture_profile"] == "step3_text"
    assert output.loc[0, "quant_signature"] == "none"
    assert output.loc[0, "measurement_type"] == "CUDA_EVENT"
    assert output.loc[0, "time_stats.moe_grouped_gemm.mean"] == 2.5


@pytest.mark.parametrize(
    ("column_name", "existing_value", "expected_value"),
    [
        ("profiling_precision", "FP16", "BF16"),
        ("measurement_type", "KERNEL_ONLY", "CUDA_EVENT"),
        ("model_arch", "generic", "step3_text"),
        ("model_architecture_profile", "generic", "step3_text"),
        ("quant_signature", "none", "fp8_w8a8"),
    ],
)
def test_attach_moe_output_metadata_rejects_conflicting_metadata(
    column_name: str,
    existing_value: str,
    expected_value: str,
) -> None:
    df = pd.DataFrame(
        [
            {
                "num_tokens": 1,
                column_name: existing_value,
                "time_stats.moe_grouped_gemm.mean": 2.5,
            }
        ]
    )

    kwargs = {
        "precision_str": "FP16",
        "model_arch": "generic",
        "model_architecture_profile": "generic",
        "quant_signature": "none",
        "measurement_type": "CUDA_EVENT",
    }
    if column_name == "profiling_precision":
        kwargs["precision_str"] = expected_value
    elif column_name == "measurement_type":
        kwargs["measurement_type"] = expected_value
    elif column_name == "model_arch":
        kwargs["model_arch"] = expected_value
    elif column_name == "model_architecture_profile":
        kwargs["model_architecture_profile"] = expected_value
    elif column_name == "quant_signature":
        kwargs["quant_signature"] = expected_value
    else:
        raise AssertionError(f"Unhandled metadata column: {column_name}")

    with pytest.raises(ValueError, match=column_name):
        moe_main._attach_moe_output_metadata(  # pylint: disable=protected-access
            df,
            **kwargs,
        )


def test_linear_actual_identity_attachment_keeps_per_row_imports_and_rejects_shared_mismatch(monkeypatch):
    from frontier.profiling.common import runtime_identity
    from frontier.profiling.linear_op.main import _attach_runtime_identity
    monkeypatch.setattr(runtime_identity, "collect_device_identity", lambda: ("CPU_FIXTURE_ONLY", "sm00"))
    monkeypatch.setattr(runtime_identity, "collect_runtime_versions", lambda: {"vllm": "CPU_FIXTURE_ONLY"})
    args = SimpleNamespace(source_revision="source", model_config_revision="local-config")
    config = SimpleNamespace(get_quant_signature=lambda: "CPU_FIXTURE_ONLY")
    facts = {"shared": {"image": "CPU_IDENTITY_ONLY"}, "imports": {}, "callables": {}}
    second = {**facts, "imports": {"norm": "CPU_IMPORT_ONLY"}}
    frame = pd.DataFrame([{"native_facts": json.dumps(item)} for item in (facts, second)])
    output = _attach_runtime_identity(frame, args, "stepfun-ai/Step-5-Preview", config)
    assert output["runtime_fingerprint"].nunique() == 1
    assert [json.loads(value) for value in output["native_facts"]] == [facts, second]
    legacy = _attach_runtime_identity(pd.DataFrame([{"num_tokens": 1}]), args, "stepfun-ai/Step-5-Preview", config)
    assert "native_facts" not in legacy
    frame.loc[1, "native_facts"] = json.dumps({**second, "shared": {"image": "OTHER_CPU_IDENTITY"}})
    with pytest.raises(ValueError, match="shared native identity"):
        _attach_runtime_identity(frame, args, "stepfun-ai/Step-5-Preview", config)


def test_moe_actual_identity_attachment_requires_per_result_receipts_and_preserves_imports(monkeypatch):
    from frontier.profiling.common import runtime_identity
    monkeypatch.setattr(runtime_identity, "collect_device_identity", lambda: ("CPU_FIXTURE_ONLY", "sm00"))
    monkeypatch.setattr(runtime_identity, "collect_runtime_versions", lambda: {"vllm": "CPU_FIXTURE_ONLY"})
    args = SimpleNamespace(source_revision="source", model_config_revision="local-config")
    facts = {"shared": {"image": "CPU_IDENTITY_ONLY"}, "imports": {}, "callables": {}}
    second = {**facts, "imports": {"gate": "CPU_IMPORT_ONLY"}}
    frame = pd.DataFrame([{"native_facts": json.dumps(item)} for item in (facts, second)])
    monkeypatch.setattr(runtime_identity, "verify_step5_profile_runtime", lambda *args: facts)
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST", "CPU_RECEIPT_ONLY")
    output = moe_main._attach_runtime_identity(frame, args, "stepfun-ai/Step-5-Preview", "CPU_FIXTURE_ONLY")
    assert [json.loads(value) for value in output["native_facts"]] == [facts, second]
    assert output["runtime_fingerprint"].nunique() == 1
    with pytest.raises(ValueError, match="per-result"):
        moe_main._attach_runtime_identity(pd.DataFrame([{"num_tokens": 1}]), args, "stepfun-ai/Step-5-Preview", "CPU_FIXTURE_ONLY")


def test_explicit_source_input_parser_preserves_one_cell_protocol_and_rejects_missing_geometry(tmp_path):
    from frontier.profiling.moe.inputs import (
        SCHEMA_VERSION, MoERoutingProfileCell, read_routing_profile_inputs, routing_profile_cell,
    )
    path = tmp_path / "moe_inputs.json"
    record = {"moe_input_id": "D26-tail", "gathered_physical_rows": 17, "source_frames": [
        {"source_lane_id": 0, "query_lens": [16], "context_lens": [6128],
         "physical_rows": 16, "gating_runtime_context": "standalone_legacy"},
        {"source_lane_id": 1, "query_lens": [], "context_lens": [],
         "physical_rows": 1, "gating_runtime_context": "standalone_legacy"}]}
    payload = {"schema_version": SCHEMA_VERSION, "routing_runtime_path": "position_cyclic", "inputs": [record]}
    path.write_text(json.dumps(payload))
    source, = read_routing_profile_inputs(path)
    assert tuple(frame.live_rows for frame in source.source_frames) == (16, 0)
    assert source.source_frames[0].context_lens == (6128,) and source.gathered_physical_rows == 17
    for cell in (MoERoutingProfileCell(source, "source_local", 0), MoERoutingProfileCell(source, "dp_gathered", 1, 15)):
        assert routing_profile_cell(json.loads(json.dumps(cell.to_json()))) == cell
    record["gathered_physical_rows"] = 16
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source widths"):
        read_routing_profile_inputs(path)
    record["gathered_physical_rows"] = 17
    payload["inputs"].append(record)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="distinct"):
        read_routing_profile_inputs(path)


@pytest.mark.parametrize("trainer", ["shared", "independent"])
@pytest.mark.parametrize("damage", ["absent", "invalid"])
def test_automatic_position_training_requires_supplied_source_before_estimators(tmp_path, monkeypatch, trainer, damage):
    from frontier.config import global_vars
    from frontier.config.model_config import BaseModelConfig
    from frontier.types import ClusterType, MeasurementType
    from frontier.execution_time_predictor.moe_dataset_training import position_moe_operator_facts
    from tests.integration import step5_measured_compute_fixture as fixture
    from tests.unit.test_step5_measured_ffn import _config, _replica
    from tests.unit.test_step5_measured_training import _manager
    from tests.unit.test_moe_grouped_gemm_padded_features import _ConcreteMoEPredictor
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", list(fixture.CAPTURE_SIZES))
    monkeypatch.setattr(fixture, "range", lambda *a: (6144,6145) if a == (6143,7167) else range(*a), raising=False)
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    paths, _ = fixture.position_compute_tables(tmp_path, model, device="h800", measurement=MeasurementType.CUDA_EVENT)
    replica = _replica(model)
    replica.moe_expert_parallel_size, replica.moe_routing_distribution_type = 16, "position_cyclic"
    manager = _manager(tmp_path / "cache", MeasurementType.CUDA_EVENT)
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica)}
    rows = pd.read_csv(paths[1])
    for facts in position_moe_operator_facts(rows, replica):
        manager._validate_moe_dataset_contract(str(paths[1]), replica, list(facts), ClusterType.MONOLITHIC, operator_facts=facts)
    if damage == "absent": rows = rows.drop(columns="routing_input")
    else: rows["routing_input"] = "{}"
    rows.to_csv(paths[1], index=False)
    def forbidden(*args, **kwargs): pytest.fail("Unvalidated source reached estimator or legacy helper")
    if trainer == "shared":
        monkeypatch.setattr(manager, "_train_single_model", forbidden)
        call = lambda: manager._train_ffn_models_for_cluster(ClusterType.MONOLITHIC, replica, _config(), str(paths[0]), str(paths[1]), True, set())
    else:
        predictor = object.__new__(_ConcreteMoEPredictor)
        predictor._model_config, predictor._replica_config, predictor._moe_input_file = model, replica, str(paths[1])
        predictor._moe_routing_distribution_type, predictor._active_measurement_type = "position_cyclic", MeasurementType.CUDA_EVENT
        monkeypatch.setattr(predictor, "_get_profiling_metadata", forbidden)
        call = predictor._train_moe_models
    with pytest.raises((ValueError, KeyError), match="routing_input|source_frames"):
        call()
