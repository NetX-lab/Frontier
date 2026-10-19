"""Original V4 source-declared allocations, without native storage qualification."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.config import ReplicaConfig, SyntheticRequestGeneratorConfig
from frontier.entities.replica import Replica
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


RAW_CONFIG = json.loads(
    (Path(__file__).resolve().parents[2]
     / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json").read_text()
)


def _replica(ep=16):
    return ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        attn_tensor_parallel_size=8, attn_dp=ep // 8,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=ep,
        num_pipeline_stages=1,
    )


def _logical_oracle(raw, ep):
    """Independent unpacked coefficient count from the original JSON shapes."""

    h, d, rq = raw["hidden_size"], raw["head_dim"], raw["q_lora_rank"]
    n, g, ro = raw["num_attention_heads"], raw["o_groups"], raw["o_lora_rank"]
    layers, vocab, mult = raw["num_hidden_layers"], raw["vocab_size"], raw["hc_mult"]
    q = (2 + mult) * mult
    main = (rq + d) * h + n * d // 8 * rq
    main += g * ro // 8 * (n * d // g) + h * (g * ro // 8)
    main += rq + d + n // 8
    attention = layers * main
    for ratio in raw["compress_ratios"][:layers]:
        if not ratio:
            continue
        overlap = 1 + (ratio == 4)
        attention += 2 * overlap * d * h + ratio * overlap * d + d
        if ratio == 4:
            ni, di = raw["index_n_heads"], raw["index_head_dim"]
            attention += ni * di * rq + ni * h + 4 * di * h + 8 * di + di
    expert = 3 * h * raw["moe_intermediate_size"]
    shared = expert * raw["n_shared_experts"]
    mlp = layers * (raw["n_routed_experts"] // ep * expert + shared)
    mlp += layers * h * raw["n_routed_experts"]
    mlp += (layers - raw["num_hash_layers"]) * raw["n_routed_experts"]
    other = 2 * vocab * h // 8 + (2 * layers + 1) * h
    other += layers * 2 * (q * mult * h + q + 3) + mult * mult * h + mult + 1
    return attention, mlp, other


@pytest.mark.parametrize(("ep", "memory"), [
    (16, 12_393_714_140), (32, 7_794_659_804), (64, 5_495_132_636),
])
def test_source_constructor_oracle_and_logical_coefficients_are_separate(ep, memory):
    counter = ParamCounter(_replica(ep))
    attention, mlp, other = _logical_oracle(RAW_CONFIG, ep)
    assert counter.get_num_attention_parameters_per_device() == attention
    assert counter.get_num_mlp_parameters_per_device() == mlp
    assert counter.get_num_parameters_per_device() == attention + mlp + other
    assert counter.get_attention_parameter_memory_per_device_bytes() == 1_612_866_432
    assert counter.get_parameter_memory_per_device_bytes() == memory
    assert counter.get_parameter_memory_per_device_bytes() != 2 * (attention + mlp + other)
    local_experts = RAW_CONFIG["n_routed_experts"] // ep
    assert memory - 43 * local_experts * 13_369_344 == 3_195_605_468
    assert counter.get_conditional_mhc_broadcast_memory_bytes() == 393_216
    assert counter.get_parameter_memory_per_device_bytes() == memory
    assert counter.get_resident_attention_stage_id() == 0
    assert counter.get_num_mtp_parameters_per_device() == 0


def test_all_43_layer_owners_have_the_exact_attention_schedule():
    replica = _replica()
    counter = ParamCounter(replica)
    specs = replica.model_config.get_layer_attention_specs()
    assert Counter(spec.variant_id for spec in specs) == {
        "uncompressed": 2, "csa": 21, "hca": 20,
    }
    assert len(replica.model_config.compress_ratios) == 44
    assert len(specs) == 43
    for spec in specs:
        expected = {"uncompressed": 18_878_848, "csa": 48_785_536, "hca": 27_530_624}
        assert counter._get_v4_attention_layer_totals(spec.global_layer_id)[1] == expected[spec.variant_id]
    assert counter.get_num_attention_params_per_layer(0) == 18_875_912
    with pytest.raises(ValueError, match="out of range"):
        counter.get_num_attention_params_per_layer(43)


def test_mixed_v4_inventory_does_not_call_homogeneous_family_api(monkeypatch):
    replica = _replica()

    def forbidden():
        raise AssertionError("homogeneous attention API must not bind mixed V4")

    monkeypatch.setattr(replica.model_config, "get_attention_family", forbidden)
    counter = ParamCounter(replica)
    assert counter.get_parameter_memory_per_device_bytes() == 12_393_714_140
    assert DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled is False
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        DEEPSEEK_V4_ATTENTION_FAMILY.require_enabled_for_execution()


def test_sp_shared_weights_are_replicated_and_all_resident_experts_are_counted():
    replica = _replica()
    model = replica.model_config
    contract = model.get_model_architecture_profile().resolve_layer_contract(
        model, layer_kind=LayerKind.SHARED, layer_id=0,
        attention_tp_size=8, attention_dp_size=2,
        expert_parallel_size=16, num_pipeline_stages=1,
    )
    assert contract.tensor_parallel_mode is TensorParallelMode.REPLICATED
    assert contract.tensor_parallel_size == 1
    original = ParamCounter(replica)
    replica.model_config = replace(model, num_experts_per_tok=4, index_topk=256)
    changed = ParamCounter(replica)
    assert changed.get_num_parameters_per_device() == original.get_num_parameters_per_device()
    # Top-k changes only three int32 hash-table widths, not resident expert matrices.
    assert original.get_mlp_parameter_memory_per_device_bytes() - changed.get_mlp_parameter_memory_per_device_bytes() == 3 * model.vocab_size * 2 * 4
    assert original.get_num_mlp_params_per_layer(0) == changed.get_num_mlp_params_per_layer(0)


def test_conditional_mhc_storage_uses_config_and_is_not_constructor_storage():
    replica = _replica()
    before = ParamCounter(replica)
    before_bytes = before.get_parameter_memory_per_device_bytes()
    replica.model_config = replace(replica.model_config, hc_mult=8)
    after = ParamCounter(replica)
    assert after.get_conditional_mhc_broadcast_memory_bytes() == (2 + 8) * 8 * 4096 * 4
    old_mhc = 43 * 2 * (24 * 16384 + 24 + 3) + 4 * 16384 + 4 + 1
    new_mhc = 43 * 2 * (80 * 32768 + 80 + 3) + 8 * 32768 + 8 + 1
    assert after.get_parameter_memory_per_device_bytes() - before_bytes == 4 * (new_mhc - old_mhc)
    assert after.get_num_parameters_per_device() - before.get_num_parameters_per_device() == new_mhc - old_mhc


@pytest.mark.parametrize(("attribute", "value"), [
    ("num_pipeline_stages", 2), ("attn_tensor_parallel_size", 4),
    ("attn_dp", 1), ("attn_dp", True), ("moe_tensor_parallel_size", 2),
    ("moe_expert_parallel_size", 32), ("device", "h200"),
    ("speculative_decoding_config", SimpleNamespace(enabled=True)),
])
def test_unrepresented_topologies_are_explicitly_rejected(attribute, value):
    replica = _replica()
    setattr(replica, attribute, value)
    with pytest.raises((ValueError, NotImplementedError), match="V4"):
        ParamCounter(replica)


def test_dp8_real_replica_and_memory_planner_use_ep64_source_storage():
    replica_config = _replica(64)
    replica_config.network_device = "h800_dgx"
    replica_config.attention_backend = "FLASHMLA_SPARSE_DSV4"
    replica_config.kv_cache_dtype = "fp8"
    replica_config.indexer_kv_cache_dtype = "fp8"
    replica = Replica(
        replica_config,
        SyntheticRequestGeneratorConfig(),
        ClusterType.MONOLITHIC,
    )

    assert replica_config.world_size == 64
    assert replica_config.local_expert_num == 4
    assert replica.local_num_experts == 4
    assert replica.num_attn_tensor_parallel_workers == 8
    assert replica.num_moe_expert_parallel_size == 64
    planner = MemoryPlanner(replica_config, replica, ClusterType.MONOLITHIC)
    assert planner.get_num_blocks(block_size=256, gpu_memory_utilization=0.9) == 71_653


def test_dp16_parameter_storage_remains_unrepresented():
    with pytest.raises(NotImplementedError, match="DP2/4/8"):
        ParamCounter(_replica(128))


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_FFN])
def test_v4_source_inventory_does_not_enable_pd_ownership(role):
    with pytest.raises(NotImplementedError, match="MONOLITHIC"):
        ParamCounter(_replica(), role)


@pytest.mark.parametrize(("field", "value"), [
    ("expert_dtype", "fp8"), ("torch_dtype", "float32"),
    ("tie_word_embeddings", True), ("topk_method", "greedy"),
    ("head_dim", True), ("num_hash_layers", -1), ("hc_mult", None),
    ("o_groups", 7), ("routed_mlp_hidden_dim", 2176),
    ("q_lora_rank", 1000), ("head_dim", 500),
    ("o_lora_rank", 1000), ("share_expert_dim", 2000),
])
def test_unsupported_source_metadata_is_rejected(field, value):
    replica = _replica()
    replica.model_config = replace(replica.model_config, **{field: value})
    with pytest.raises((ValueError, NotImplementedError), match="V4"):
        ParamCounter(replica)


def test_explicit_head_dimension_does_not_require_hidden_size_per_head():
    replica = _replica()
    replica.model_config = replace(replica.model_config, num_q_heads=96)

    counter = ParamCounter(replica)

    assert replica.model_config.embedding_dim % replica.model_config.num_q_heads != 0
    assert counter.get_num_attention_parameters_per_device() > 0


@pytest.mark.parametrize(("field", "value"), [
    ("scale_fmt", None), ("weight_block_size", (64, 64)),
    ("ignored_layers", ("model.layers.0.self_attn.wq_b",)),
    ("quantized_operations", ("moe_grouped_gemm",)),
])
def test_unrepresented_quantized_storage_is_rejected(field, value):
    replica = _replica()
    quant = replace(replica.model_config.quantization_config, **{field: value})
    replica.model_config = replace(replica.model_config, quantization_config=quant)
    with pytest.raises(NotImplementedError, match="UE8M0"):
        ParamCounter(replica)


@pytest.mark.parametrize("mutation", ["missing", "extra_mtp", "wrong_family", "wrong_variant"])
def test_incomplete_backbone_or_noncanonical_bindings_are_rejected(monkeypatch, mutation):
    replica = _replica()
    specs = replica.model_config.get_layer_attention_specs()
    if mutation == "missing":
        changed = specs[:-1]
    elif mutation == "extra_mtp":
        changed = (*specs, replace(specs[-1], global_layer_id=43))
    elif mutation == "wrong_family":
        changed = (replace(specs[0], family_id="dense_attention"), *specs[1:])
    else:
        changed = (*specs[:2], replace(specs[2], variant_id="hca"), *specs[3:])
    monkeypatch.setattr(replica.model_config, "get_layer_attention_specs", lambda: changed)
    with pytest.raises(ValueError, match="V4"):
        ParamCounter(replica)


def test_non_v4_conditional_mhc_accessor_preserves_existing_models():
    for name in ("zai-org/GLM-5.3", "llama2_7b_dense_example"):
        replica = _replica()
        from frontier.config.model_config import BaseModelConfig

        replica.model_config = BaseModelConfig.create_from_name(name)
        assert ParamCounter(replica).get_conditional_mhc_broadcast_memory_bytes() == 0


@pytest.mark.parametrize(("ep", "expected_bytes"), [
    (16, 12_393_714_140), (32, 7_794_659_804), (64, 5_495_132_636),
])
def test_real_metrics_export_keeps_packed_bytes_and_conditional_storage_separate(
    ep, expected_bytes, tmp_path,
):
    from frontier.config import ClusterConfig, MetricsConfig, SimulationConfig
    from frontier.metrics.metrics_store import MetricsStore

    cluster = ClusterConfig(replica_config=_replica(ep))
    output = tmp_path / "metrics"
    output.mkdir()
    config = SimulationConfig(
        cluster_config=cluster,
        metrics_config=MetricsConfig(output_dir=str(output), write_metrics=True, store_plots=False),
    )
    store = MetricsStore(config, {ClusterType.MONOLITHIC: cluster})
    store._write_system_metrics()
    report = json.loads((Path(config.metrics_config.output_dir) / "system_metrics.json").read_text())
    memory = report["model_weight_memory"]["MONOLITHIC"]
    assert "error" not in memory
    assert memory["total_memory_bytes"] == expected_bytes
    assert memory["conditional_mhc_broadcast_memory_bytes"] == 393_216
    assert memory["conditional_mhc_broadcast_native_verified"] is False
    assert memory["memory_accounting"] == (
        "source-defined weight storage estimate; native allocation unverified"
    )
    parts = memory["breakdown"]
    assert parts["attention_memory_bytes"] == 1_612_866_432
    if ep == 64:
        assert memory["device_info"]["expert_parallel_size"] == 64
        assert parts["ffn_memory_bytes"] == 3_481_250_304
        assert parts["other_memory_bytes"] == 401_015_900
    assert expected_bytes == sum(parts[k] for k in (
        "attention_memory_bytes", "ffn_memory_bytes", "other_memory_bytes",
    ))
