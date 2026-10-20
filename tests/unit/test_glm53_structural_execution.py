"""Exercise GLM source contracts with synthetic costs through the real factory.

Predictor tests use TP8/MoE-TP8/EP1 to establish ordered cost accounting.
The metrics writer test uses TP8/DP8/MoE-TP1/EP64 and checks source-defined
parameter storage estimates. Neither proves native allocation, 64-GPU execution
or measured timing fidelity.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from predictor_cache_fixtures import predictor_fixture_config
from frontier.attention.families import DSA_ATTENTION_FAMILY
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.config import (
    BaseModelConfig,
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    VllmV1SchedulerConfig,
    global_vars,
)
from frontier.config.quantization_manager import QuantizationManager
from frontier.entities import Batch, Request
from frontier.entities.time_components import MLPTime, MoETime
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY_MS = 0.25
CPU_FIELDS = (
    "schedule_time", "sampler_e2e_time", "prepare_inputs_e2e_time",
    "process_model_outputs_time", "ray_comm_time",
)


@pytest.fixture
def predictor_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Keep factory selection and precision state local to each CPU test."""
    monkeypatch.chdir(REPO_ROOT)
    for name, value in (
        ("SYS_ARCH", "co-location"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("KV_CONNECTOR", "none"), ("QUANTIZATION_MANAGER", None),
        ("CUDAGRAPH_CAPTURE_SIZES", None),
        ("ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC", False),
        ("KV_SEND_SLOWDOWN_PER_LAYER_MS", 0.0),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
    ):
        monkeypatch.setattr(global_vars, name, value)
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)

    model = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    inputs = predictor_fixture_config(
        model_config=model, total_experts=256, router_topk=8,
    )
    replica = ReplicaConfig(
        model_name="zai-org/GLM-5.3", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, moe_tensor_parallel_size=8,
        moe_expert_parallel_size=1, total_expert_num=256, router_topk=8,
    )
    replica.model_config = model
    replica.cluster_num_replicas = 1
    inputs.update(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(block_size=64),
        predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True, dummy_execution_time_ms=DUMMY_MS,
        ),
        metrics_config=MetricsConfig(
            cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "outputs"),
        ),
    )
    return inputs


def _batch(phase: str = "prefill") -> Batch:
    prefill = Request(arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4)
    decode = Request(
        arrived_at=0.0, num_prefill_tokens=16, num_decode_tokens=4,
        num_processed_tokens=16,
    )
    if phase == "prefill":
        requests, tokens = [prefill], [16]
    elif phase == "decode":
        requests, tokens = [decode], [1]
    else:
        assert phase == "mixed"
        requests, tokens = [prefill, decode], [8, 1]
    return Batch(replica_id=0, requests=requests, num_tokens=tokens, is_moe=True)


def _predict(predictor, *, layer_id: int = 0, num_layers: int = 78,
             include_attention: bool = True, phase: str = "prefill"):
    return predictor.predict_stage_execution_time(
        _batch(phase), stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        layer_id=layer_id, num_layers=num_layers,
        include_attention=include_attention,
    )


def _assert_cpu_once(stage) -> None:
    for field in CPU_FIELDS:
        assert getattr(stage, field) == DUMMY_MS
        assert getattr(stage.stage_execution_time, field) == DUMMY_MS
        assert [getattr(layer, field) for layer in stage.layer_execution_times] == [
            DUMMY_MS, *([0.0] * (stage.num_layers - 1)),
        ]


@pytest.mark.parametrize("phase", ["prefill", "decode", "mixed"])
def test_real_factory_preserves_glm_backbone_and_scope_accounting(
    predictor_inputs, phase,
):
    predictor = RandomForrestExecutionTimePredictor(**predictor_inputs)
    assert isinstance(predictor, SklearnMoEExecutionTimePredictor)
    stage = _predict(predictor, phase=phase)
    layers = stage.layer_execution_times
    full_ids = (0, 1, 2, *range(6, 78, 4))

    assert stage.num_layers == 78
    assert stage.global_layer_ids == tuple(range(78))
    assert stage.attention_family_ids == ("dsa_attention",) * 78
    assert stage.attention_variant_ids == tuple(
        "full_index" if layer_id in full_ids else "shared_index"
        for layer_id in range(78)
    )
    assert sum(layer.attn_dsa_indexer_time > 0 for layer in layers) == 21
    assert sum(layer.attn_dsa_indexer_time == 0 for layer in layers) == 57
    assert all(layer.attn_dsa_mla_time == DUMMY_MS for layer in layers)
    assert all(isinstance(layer.moe_or_mlp_time_component, MLPTime) for layer in layers[:3])
    assert all(isinstance(layer.moe_or_mlp_time_component, MoETime) for layer in layers[3:])
    assert dict(stage.attention_operator_times.op_times) == {
        "attn_dsa_indexer": 21 * DUMMY_MS, "attn_dsa_mla": 78 * DUMMY_MS,
    }
    assert stage.attention_operator_times.total_time() == 99 * DUMMY_MS
    assert stage.attention_time_component.total_time() == (78 * 4 + 99) * DUMMY_MS
    assert stage.attention_all_reduce_time == 78 * DUMMY_MS
    assert stage.attention_time == (78 * 4 + 99) * DUMMY_MS
    for layer in layers:
        component = layer.attention_time_component
        assert component.attention_prefill_execution_time == 0.0
        assert component.attention_decode_execution_time == 0.0
        assert component.attention_kv_cache_save_execution_time == 0.0
    _assert_cpu_once(stage)


def test_real_factory_slice_crosses_dense_moe_and_full_shared_boundaries(predictor_inputs):
    predictor = RandomForrestExecutionTimePredictor(**predictor_inputs)
    stage = _predict(predictor, layer_id=2, num_layers=6)

    assert stage.global_layer_ids == (2, 3, 4, 5, 6, 7)
    assert stage.attention_variant_ids == (
        "full_index", "shared_index", "shared_index", "shared_index",
        "full_index", "shared_index",
    )
    assert isinstance(stage.layer_execution_times[0].moe_or_mlp_time_component, MLPTime)
    assert all(isinstance(layer.moe_or_mlp_time_component, MoETime)
               for layer in stage.layer_execution_times[1:])
    assert stage.attn_dsa_indexer_time == 2 * DUMMY_MS
    assert stage.attn_dsa_mla_time == 6 * DUMMY_MS
    assert stage.attention_operator_times.total_time() == 8 * DUMMY_MS
    _assert_cpu_once(stage)


def test_post_attention_moe_stage_has_no_attention_scopes(predictor_inputs):
    predictor = RandomForrestExecutionTimePredictor(**predictor_inputs)
    stage = _predict(predictor, layer_id=3, num_layers=4, include_attention=False)

    assert stage.attention_operator_times is None
    assert stage.attention_time == 0.0
    assert stage.attn_dsa_indexer_time == stage.attn_dsa_mla_time == 0.0
    for layer in stage.layer_execution_times:
        assert layer.attention_operator_times is None
        assert "attn_dsa_indexer" not in layer.op_times
        assert "attn_dsa_mla" not in layer.op_times
    _assert_cpu_once(stage)


def test_structural_trace_requires_explicit_admission(predictor_inputs):
    predictor = RandomForrestExecutionTimePredictor(**predictor_inputs)
    stage = _predict(predictor, layer_id=2, num_layers=2)
    for layer in stage.layer_execution_times:
        with pytest.raises(NotImplementedError, match="DSA"):
            get_attention_trace_op_times(layer, DSA_ATTENTION_FAMILY)
    full, shared = stage.layer_execution_times
    assert [(op.name, value) for op, value in get_attention_trace_op_times(
        full, DSA_ATTENTION_FAMILY, allow_structural=True,
    )] == [("attn_dsa_indexer", DUMMY_MS), ("attn_dsa_mla", DUMMY_MS)]
    assert [(op.name, value) for op, value in get_attention_trace_op_times(
        shared, DSA_ATTENTION_FAMILY, allow_structural=True, skip_zero=False,
    )] == [("attn_dsa_indexer", 0.0), ("attn_dsa_mla", DUMMY_MS)]


def test_real_factory_rejects_normal_mode_before_loading_profiles(predictor_inputs):
    predictor_inputs["predictor_config"] = replace(
        predictor_inputs["predictor_config"], enable_dummy_mode=False,
    )
    with pytest.raises(NotImplementedError, match="DSA"):
        RandomForrestExecutionTimePredictor(**predictor_inputs)


def test_real_factory_rejects_historical_homogeneous_dsa(predictor_inputs):
    replica = predictor_inputs["replica_config"]
    replica.model_config = replace(replica.model_config, index_topk=None)
    assert {spec.variant_id for spec in replica.model_config.get_layer_attention_specs()} == {
        "dsa",
    }
    with pytest.raises(ValueError, match="requires full/shared index bindings"):
        RandomForrestExecutionTimePredictor(**predictor_inputs)


@pytest.mark.parametrize(("layer_id", "producer"), [(0, None), (1, 0), (3, 1)])
def test_real_factory_rejects_malformed_index_producer(
    predictor_inputs, monkeypatch: pytest.MonkeyPatch, layer_id, producer,
):
    model = predictor_inputs["replica_config"].model_config
    specs = list(model.get_layer_attention_specs())
    specs[layer_id] = replace(specs[layer_id], index_producer_layer_id=producer)
    # Inject a corrupt normalized input without replacing the predictor gate.
    monkeypatch.setattr(model, "_layer_attention_specs_cache", tuple(specs))
    with pytest.raises(ValueError, match="invalid index producer binding"):
        RandomForrestExecutionTimePredictor(**predictor_inputs)


@pytest.mark.parametrize("block_size", [16, 32, 128])
def test_real_factory_rejects_non64_indexer_blocks(predictor_inputs, block_size):
    predictor_inputs["replica_scheduler_config"] = VllmV1SchedulerConfig(
        block_size=block_size,
    )
    with pytest.raises(NotImplementedError, match="block size 64"):
        RandomForrestExecutionTimePredictor(**predictor_inputs)


@pytest.mark.parametrize(
    ("model_name", "family_id", "includes_dsa"),
    [
        pytest.param("zai-org/GLM-5.3", "dsa_attention", True, id="glm53"),
        pytest.param("meta-llama/Llama-2-7b-hf", "dense_attention", False, id="dense"),
        pytest.param("deepseek-v3", "latent_mla_attention", False, id="mla"),
        pytest.param(
            "Qwen3.8-2.4T-A95B-Quark-MXFP4", "gated_delta_net", False, id="gdn",
        ),
    ],
)
def test_real_metrics_initializer_limits_dsa_catalogs_to_dsa_models(
    predictor_inputs, tmp_path, model_name, family_id, includes_dsa,
):
    replica = ReplicaConfig(
        model_name=model_name, device="h800", network_device="h800_dgx",
    )
    assert family_id in {
        spec.family_id for spec in replica.model_config.get_layer_attention_specs()
    }
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=predictor_inputs["replica_scheduler_config"],
        execution_time_predictor_config=predictor_inputs["predictor_config"],
    )
    config = SimulationConfig(
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(tmp_path / "metrics"), write_metrics=False, store_plots=True,
        ),
    )
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster})
    dsa_metrics = {OperationMetrics.ATTN_DSA_INDEXER, OperationMetrics.ATTN_DSA_MLA}
    expected_catalog = set(OperationMetrics) - {OperationMetrics.KV_CACHE_BLOCK_ZERO}
    from frontier.operators.sequence_parallel_moe import (
        EMBEDDING_LOGITS_OPERATORS, SP_TP_OPERATORS,
    )

    # These fixtures use TP1, so no sequence-parallel MoE calls are reachable.
    expected_catalog -= {OperationMetrics(name) for name in SP_TP_OPERATORS}
    if not includes_dsa:
        expected_catalog -= {OperationMetrics(name) for name in EMBEDDING_LOGITS_OPERATORS}
    if not includes_dsa:
        expected_catalog -= dsa_metrics

    for catalogs in (store._operation_metrics, store._operation_metrics_per_batch):
        catalog = catalogs[ClusterType.MONOLITHIC]
        assert set(catalog) == expected_catalog
        assert all(series._save_plots for series in catalog.values())


def test_real_metrics_writer_preserves_glm_parameter_and_storage_ownership(
    predictor_inputs, tmp_path,
):
    replica = ReplicaConfig(
        model_name="zai-org/GLM-5.3", device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=8,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=64,
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=predictor_inputs["replica_scheduler_config"],
        execution_time_predictor_config=predictor_inputs["predictor_config"],
    )
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    config = SimulationConfig(
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(output_dir), write_metrics=True, store_plots=False,
        ),
    )
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster})
    store._write_system_metrics()
    report = json.loads(
        (Path(config.metrics_config.output_dir) / "system_metrics.json").read_text()
    )
    memory = report["model_weight_memory"]["MONOLITHIC"]
    assert "error" not in memory
    # TP8/DP8/EP64 is sequence-parallel MoE: each rank keeps a full shared
    # expert, adding 75 * 37,748,736 * 7 / 8 coefficients and their scales
    # (75 * (37,748,736 + 9,216) * 7 / 8 bytes) to the TP-sharded totals.
    assert memory["total_parameters"] == 17_503_822_848
    assert memory["total_memory_bytes"] == 18_172_004_352
    assert memory["memory_accounting"] == (
        "source-defined weight storage estimate; native allocation unverified"
    )
    breakdown = memory["breakdown"]
    assert breakdown["attention_parameters"] == 2_906_267_904
    assert breakdown["attention_memory_bytes"] == 3_214_089_984
    assert breakdown["ffn_parameters"] == 14_358_694_656
    assert breakdown["ffn_memory_bytes"] == 14_480_193_792
    assert breakdown["other_parameters"] == 238_860_288
    assert breakdown["other_memory_bytes"] == 477_720_576
    assert memory["total_memory_bytes"] == sum(
        breakdown[key] for key in (
            "attention_memory_bytes", "ffn_memory_bytes", "other_memory_bytes",
        )
    )
