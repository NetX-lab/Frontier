"""Normal shared-manager attention fitting with explicit modality ownership."""

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config import global_vars
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.execution_time_predictor.step5_cost import step5_measurements_for_role
from frontier.types import ClusterType, MeasurementType
from test_step5_measured_training import _replica, _row, _scheduler, _write
from test_step5_mrv2_inputs import producer_row
from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.profiling.step5.inputs import Step5ProfileInput


@pytest.mark.parametrize("mode", ["none", "full_decode_only", "piecewise", "full_and_piecewise"])
@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE,
                                  ClusterType.MONOLITHIC, ClusterType.DECODE_FFN])
def test_attention_policy_is_independent_from_generic_operator_family(mode, role):
    decode = MeasurementType.KERNEL_ONLY if mode == "none" else MeasurementType.CUDA_EVENT
    expected = {
        ClusterType.PREFILL: tuple(dict.fromkeys((MeasurementType.KERNEL_ONLY, decode))),
        ClusterType.DECODE: tuple(dict.fromkeys((MeasurementType.KERNEL_ONLY, decode))),
        ClusterType.MONOLITHIC: tuple(dict.fromkeys((MeasurementType.KERNEL_ONLY, decode))),
        ClusterType.DECODE_FFN: (),
    }
    assert step5_measurements_for_role(role, mode) == expected[role]


def test_normal_manager_fits_both_required_p_attention_families(tmp_path, monkeypatch):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    path = _write(tmp_path, [_row(model, layer, context=8176) for layer in (0, 3)])
    event_dir = tmp_path / "event_attention"
    event_dir.mkdir()
    event_path = _write(event_dir, [producer_row(model, Step5ProfileInput.from_mrv2_dummy(
        layer, MRv2DummyDescriptor(1, 1, "full", None, "CPU_P_FULL_DUMMY_FIXTURE", 1),
        block_table_columns=512), duration=.321, count=17) for layer in (0, 3)],
        MeasurementType.CUDA_EVENT)
    gap = tmp_path / "kernel_gap.csv"
    pd.DataFrame({"execution_mode": ["eager", "cuda_graph"],
                  "num_tensor_parallel_workers": [8, 8],
                  "kernel_gap_us": [1., 2.]}).to_csv(gap, index=False)
    predictor_config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        atten_input_file=str(event_path),
        atten_kernel_only_input_file=str(path), kernel_gap_input_file=str(gap),
        num_estimators=[2], max_depth=[3], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )
    replica = _replica(model)
    replica.moe_tensor_parallel_size = 1
    replica.moe_expert_parallel_size = 1
    replica.attn_dp = 4
    replica.num_pipeline_stages = 1
    replica.network_device = "cpu_fixture_network"
    replica.speculative_decoding_config = SimpleNamespace(enabled=False)
    scheduler = _scheduler()
    scheduler.get_type = lambda: "cpu_fixture_scheduler"
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=scheduler,
                              execution_time_predictor_config=predictor_config,
                              cc_backend_config=SimpleNamespace(get_name=lambda: "cpu_fixture_comm"))
    metrics = SimpleNamespace(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(global_vars, "get_sys_arch", lambda: "colocated")
    monkeypatch.setattr(global_vars, "get_decode_cuda_graph_mode", lambda: "full_decode_only")
    calls = []

    def generic(self, *args, **kwargs):
        calls.append(self._active_measurement_type)
        return {}

    # This test isolates the attention component. Full FFN/CPU fitting is separate.
    for name in ("_train_ffn_models_for_cluster", "_train_residual_models_for_cluster",
                 "_train_cpu_overhead_models_for_cluster", "_train_step5_input_layernorm_for_cluster"):
        monkeypatch.setattr(ExecutionTimePredictionModelManager, name, generic)
    monkeypatch.setattr(ExecutionTimePredictionModelManager, "_should_train_communication_models",
                        lambda *_: False)
    # Empty generic category spies above are not a complete role bundle.
    monkeypatch.setattr(ExecutionTimePredictionModelManager, "_publish_step5_model_bundles", lambda *_: None)
    manager = ExecutionTimePredictionModelManager({ClusterType.PREFILL: cluster}, metrics)
    assert calls == [MeasurementType.CUDA_EVENT] * 4
    models = manager.get_models_for_cluster(ClusterType.PREFILL)
    assert set(models["kernel_only"]) == {"step5_layer_e2e", "step5_layer_e2e_kernel_count"}
    assert set(models["eager"]) == {"step5_layer_e2e", "step5_layer_e2e_kernel_count"}
    assert set(manager._trained_models) == {f"{family}:{name}"
        for family in ("eager", "kernel_only")
        for name in ("step5_layer_e2e", "step5_layer_e2e_kernel_count")}


def test_normal_manager_fits_and_binds_complete_cpu_fixture_bundle(tmp_path, monkeypatch):
    """No category fitting/publication spies: native qualification is not asserted."""
    import test_step5_measured_ffn as ffn_fixture
    from frontier.types import CCBackendType

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    attention_dir, compute_dir = tmp_path / "attention", tmp_path / "compute"
    attention_dir.mkdir()
    compute_dir.mkdir()
    path = _write(attention_dir, [_row(model, layer, context=8176) for layer in range(92)])
    event_dir = tmp_path / "event_attention"
    event_dir.mkdir()
    event_path = _write(event_dir, [producer_row(model, Step5ProfileInput.from_mrv2_dummy(
        layer, MRv2DummyDescriptor(1, 1, "full", None, "CPU_P_FULL_DUMMY_FIXTURE", 1),
        block_table_columns=512), duration=.321, count=17) for layer in range(92)],
        MeasurementType.CUDA_EVENT)
    metadata = ffn_fixture._metadata
    with monkeypatch.context() as context:
        # Construct distinct, labelled event fixtures; no measured rows are relabelled.
        context.setattr(ffn_fixture, "_metadata", lambda config: {
            **metadata(config), "measurement_type": MeasurementType.CUDA_EVENT.value,
        })
        linear, moe = ffn_fixture._tables(compute_dir, model, token_counts=(1, 8, 16, 32))
    gap = tmp_path / "kernel_gap.csv"
    pd.DataFrame({"execution_mode": ["eager", "cuda_graph"],
                  "num_tensor_parallel_workers": [8, 8],
                  "kernel_gap_us": [1., 2.]}).to_csv(gap, index=False)
    config = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        linear_op_input_file=str(linear), moe_input_file=str(moe),
        atten_input_file=str(event_path),
        atten_kernel_only_input_file=str(path), kernel_gap_input_file=str(gap),
        num_estimators=[2], max_depth=[3], min_samples_split=[2],
        k_fold_cv_splits=2, num_training_job_threads=1,
    )
    transfer_input = tmp_path / "nixl_input_binding_fixture.json"
    transfer_input.write_text('{"evidence_kind":"cpu_fixture","purpose":"binding_only"}')
    config.nixl_read_profile = str(transfer_input)
    replica = ffn_fixture._replica(model)
    replica.num_pipeline_stages = 1
    replica.attn_dp = 4
    replica.network_device = "cpu_fixture_network"
    replica.speculative_decoding_config = SimpleNamespace(enabled=False)
    scheduler = _scheduler()
    scheduler.get_type = lambda: "cpu_fixture_scheduler"
    backend = SimpleNamespace(get_type=lambda: CCBackendType.COLLECTIVE_SIM,
                              get_name=lambda: "collective_sim")
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=scheduler,
                              execution_time_predictor_config=config, cc_backend_config=backend)
    monkeypatch.setattr(global_vars, "get_sys_arch", lambda: "colocated")
    monkeypatch.setattr(global_vars, "get_decode_cuda_graph_mode", lambda: "full_decode_only")
    manager = ExecutionTimePredictionModelManager(
        {ClusterType.PREFILL: cluster}, SimpleNamespace(cache_dir=str(tmp_path / "cache")),
    )
    views = manager.get_models_for_cluster(ClusterType.PREFILL)
    assert {"step5_layer_e2e", "step5_layer_e2e_kernel_count"} <= set(views["kernel_only"])
    assert {"input_layernorm", "mlp_up_proj", "share_expert_up_proj",
            "moe_grouped_gemm", "step5_ffn_merge"} <= set(views["eager"])
    receipts = manager._step5_model_bundles[ClusterType.PREFILL]
    assert set(receipts) == {"eager", "kernel_only"}
    for receipt in receipts.values():
        assert receipt["identity"]["input_bindings"]["nixl_read_profile"]["path"] == str(transfer_input)
        loaded = manager.load_model_bundle(receipt["path"], receipt["identity"])
        assert loaded
    event_precisions = {
        registration.get("precision")
        for registration in receipts["eager"]["identity"]["registrations"].values()
    }
    assert event_precisions >= {"BF16", "FP8", "FP32"}
    assert {
        registration.get("precision")
        for registration in receipts["kernel_only"]["identity"]["registrations"].values()
    } == {"BF16"}
    transfer_input.write_text('{"evidence_kind":"cpu_fixture","purpose":"changed_input"}')
    with pytest.raises(ValueError, match="input binding checksum mismatch"):
        manager.load_model_bundle(receipt["path"], receipt["identity"])
    with pytest.raises(ValueError, match="input binding checksum mismatch"):
        manager.get_models_for_cluster(ClusterType.PREFILL)
