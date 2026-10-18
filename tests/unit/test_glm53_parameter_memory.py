"""Source-defined GLM parameter estimates; native physical allocation is unqualified."""

from dataclasses import replace

import pytest

from frontier.config import ReplicaConfig
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter


FULL_MODEL_ROLES = (ClusterType.MONOLITHIC, ClusterType.PREFILL, ClusterType.DECODE)


def _replica(
    *, ep: int = 16, tp: int = 8, role: ClusterType = ClusterType.MONOLITHIC,
) -> ReplicaConfig:
    return ReplicaConfig(
        model_name="zai-org/GLM-5.3", device="h800",
        cluster_prefix=None if role is ClusterType.MONOLITHIC else role.name.lower(),
        attn_tensor_parallel_size=tp, attn_dp=ep // tp,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=ep,
        num_pipeline_stages=1,
    )


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
@pytest.mark.parametrize(("ep", "mlp_elements", "mlp_bytes", "elements", "memory"), [
    (16, 45_855_296_256, 45_984_484_992, 49_000_424_448, 49_676_295_552),
    (32, 23_206_054_656, 23_329_713_792, 26_351_182_848, 27_021_524_352),
    (64, 11_881_433_856, 12_002_328_192, 15_026_562_048, 15_694_138_752),
])
def test_original_source_inventory_distinguishes_parameters_from_storage(
    role: ClusterType, ep: int, mlp_elements: int, mlp_bytes: int,
    elements: int, memory: int,
) -> None:
    """Pinned 78-layer counts include all resident experts, scales and one KV-B backing."""

    replica = _replica(ep=ep, role=role)
    assert replica.world_size == ep
    counter = ParamCounter(replica, role)
    assert counter.get_num_attention_parameters_per_device() == 2_906_267_904
    assert counter.get_attention_parameter_memory_per_device_bytes() == 3_214_089_984
    assert counter.get_num_mlp_parameters_per_device() == mlp_elements
    assert counter.get_mlp_parameter_memory_per_device_bytes() == mlp_bytes
    assert counter.get_num_parameters_per_device() == elements
    assert counter.get_parameter_memory_per_device_bytes() == memory
    other_elements = elements - 2_906_267_904 - mlp_elements
    assert other_elements == 238_860_288  # Two vocabulary shards and decoder/final norms.
    assert memory - 3_214_089_984 - mlp_bytes == 2 * other_elements
    assert counter.get_resident_attention_stage_id() == 0
    assert counter.get_num_mtp_parameters_per_device() == 0


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
def test_unified_roles_retain_the_full_backbone_and_both_vocabulary_endpoints(
    role: ClusterType,
) -> None:
    replica = _replica(ep=32, role=role)
    counter = ParamCounter(replica, role)
    monolithic = ParamCounter(_replica(ep=32))
    model = replica.model_config
    assert model.num_layers == 78
    assert [layer for layer in range(78) if not model.is_moe_layer(layer)] == [0, 1, 2]
    assert sum(model.is_moe_layer(layer) for layer in range(78)) == 75
    assert counter.get_attention_stage_layer_counts() == ((0, 78),)
    for layer_id in range(78):
        assert counter.get_num_attention_params_per_layer(layer_id) == (
            monolithic.get_num_attention_params_per_layer(layer_id)
        )
        assert counter.get_num_mlp_params_per_layer(layer_id) == (
            monolithic.get_num_mlp_params_per_layer(layer_id)
        )
    # Three dense layers and75 routed-plus-shared layers stay on each role.
    assert counter.get_num_mlp_params_per_layer(0) == 28_311_552
    assert counter.get_num_mlp_params_per_layer(3) == 308_281_600
    backbone_elements = sum(counter.get_num_parameters_per_layer(layer) for layer in range(78))
    endpoints_and_final_norm = 2 * 154_880 * 6_144 // 8 + 6_144
    assert counter.get_num_parameters_per_device() - backbone_elements == endpoints_and_final_norm
    assert counter.get_parameter_memory_per_device_bytes() == 27_021_524_352


def test_full_and_shared_index_layers_have_distinct_parameter_owners() -> None:
    replica = _replica()
    counter = ParamCounter(replica)
    full = {spec.global_layer_id for spec in replica.model_config.get_layer_attention_specs()
            if spec.variant_id == "full_index"}
    assert len(full) == 21
    for layer_id in range(78):
        expected = 44_108_544 if layer_id in full else 34_736_640
        assert counter.get_num_attention_params_per_layer(layer_id) == expected
    assert counter.get_num_attention_params_per_layer() == 44_108_544
    assert counter.get_num_mlp_params_per_layer(0) == 28_311_552
    assert counter.get_num_mlp_params_per_layer(3) == 610_271_488
    assert counter.get_num_parameters_per_layer(0) == 44_108_544 + 28_311_552 + 12_288
    with pytest.raises(ValueError, match="out of range"):
        counter.get_num_attention_params_per_layer(78)


@pytest.mark.parametrize("extra_layer", [False, True])
def test_attention_bindings_cover_exactly_the_backbone(
    extra_layer: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica = _replica()
    model = replica.model_config
    specs = model.get_layer_attention_specs()
    invalid = (
        (*specs, replace(specs[-1], global_layer_id=model.num_layers))
        if extra_layer else specs[:-1]
    )
    monkeypatch.setattr(model, "get_layer_attention_specs", lambda: invalid)
    with pytest.raises(ValueError, match="every backbone layer in order"):
        ParamCounter(replica)


def test_index_topk_changes_work_but_not_resident_weight_counts() -> None:
    replica = _replica()
    original = ParamCounter(replica).get_parameter_memory_per_device_bytes()
    replica.model_config = replace(replica.model_config, index_topk=1024, num_experts_per_tok=4)
    counter = ParamCounter(replica)
    assert counter.get_parameter_memory_per_device_bytes() == original
    assert counter.get_num_parameters_per_device() == 49_000_424_448


def test_geometry_valid_tp4_uses_larger_shards_without_dividing_replicated_indexers() -> None:
    """Independent TP4 source arithmetic is structural evidence, not a native fit claim."""

    counter = ParamCounter(_replica(tp=4))
    assert counter.get_num_attention_parameters_per_device() == 4_358_021_376
    assert counter.get_attention_parameter_memory_per_device_bytes() == 4_952_459_136
    assert counter.get_num_parameters_per_device() == 51_128_902_656
    assert counter.get_parameter_memory_per_device_bytes() == 52_329_392_256


def test_geometry_is_derived_from_config_and_typed_ffn_widths() -> None:
    replica = _replica()
    replica.model_config = replace(
        replica.model_config, embedding_dim=4096, q_lora_rank=1024,
        kv_lora_rank=256, qk_nope_head_dim=128, qk_rope_head_dim=64,
        qk_head_dim=192, v_head_dim=128, index_n_heads=16,
        dense_mlp_hidden_dim=8192, routed_mlp_hidden_dim=1024,
        share_expert_dim=1024, vocab_size=131072,
    )
    counter = ParamCounter(replica)
    # Main learned11,797,760 / bytes12,850,560; full indexer2,687,232 / bytes3,277,824.
    assert counter.get_num_attention_parameters_per_device() == 976_657_152
    assert counter.get_attention_parameter_memory_per_device_bytes() == 1_071_177_984
    assert counter.get_num_mlp_parameters_per_device() == 15_333_870_336
    assert counter.get_num_parameters_per_device() == 16_445_388_288
    assert counter.get_parameter_memory_per_device_bytes() == 16_757_195_136


def test_mla_hidden_width_does_not_impose_dense_attention_head_divisibility() -> None:
    replica = _replica()
    replica.model_config = replace(
        replica.model_config, embedding_dim=4096, num_q_heads=48, num_kv_heads=48,
    )
    counter = ParamCounter(replica)
    expected_main = (
        4096 * 2624 + 6 * 256 * 2048 + 6 * 448 * 512 + 4096 * 6 * 256 + 2560
    )
    expected_index = 4096 * 2048 + 4096 * 160 + 256
    assert counter.get_num_attention_params_per_layer(0) == expected_main + expected_index
    assert counter.get_num_attention_params_per_layer(3) == expected_main


@pytest.mark.parametrize("suffixes", [
    ("fused_qkv_a_proj",),
    ("q_a_proj", "kv_a_proj_with_mqa"),
    ("fused_qkv_a_proj", "q_a_proj"),
])
def test_exact_uniform_fused_exclusions_select_one_bf16_projection(
    suffixes: tuple[str, ...],
) -> None:
    replica = _replica()
    model = replica.model_config
    quant = replace(model.quantization_config, ignored_layers=[
        *model.quantization_config.ignored_layers,
        *(f"model.layers.0.self_attn.{suffix}" for suffix in suffixes),
    ])
    replica.model_config = replace(model, quantization_config=quant)
    counter = ParamCounter(replica)
    assert counter.get_num_parameters_per_device() == 49_000_424_448
    # Replace one16,121,856-byte FP8 A +4032-byte scales by BF16 A, once.
    assert counter.get_parameter_memory_per_device_bytes() == 49_676_295_552 + 16_117_824


@pytest.mark.parametrize(("exclusions", "extra_bytes"), [
    (("model.layers.3.mlp.shared_experts.gate_up_proj",), 3_144_960),
    (("model.layers.3.mlp.shared_experts.gate_proj",
      "model.layers.3.mlp.shared_experts.up_proj"), 3_144_960),
    (("model.layers.3.mlp.shared_experts.gate_up_proj",
      "model.layers.3.mlp.shared_experts.gate_proj"), 3_144_960),
    (("model.layers.3.mlp.shared_experts.gate_up_proj.weight",), 0),
    (("model.layers.3.mlp.shared_experts.down_proj.weight",), 1_572_480),
    (("checkpoint.model.layers.3.mlp.shared_experts.down_proj.weight",), 1_572_480),
])
def test_shared_expert_exclusions_preserve_packed_precedence_and_weight_matching(
    exclusions: tuple[str, ...], extra_bytes: int,
) -> None:
    replica = _replica()
    model = replica.model_config
    replica.model_config = replace(
        model, quantization_config=replace(
            model.quantization_config,
            ignored_layers=[*model.quantization_config.ignored_layers, *exclusions],
        ),
    )
    counter = ParamCounter(replica)
    assert counter.get_num_parameters_per_device() == 49_000_424_448
    # The same TP8 shared weights replace FP8 coefficients and FP32 scales with BF16.
    assert counter.get_parameter_memory_per_device_bytes() == 49_676_295_552 + extra_bytes


def test_exclusions_use_exact_module_names_instead_of_leaf_or_parent_matching() -> None:
    replica = _replica()
    model = replica.model_config
    quant = replace(model.quantization_config, ignored_layers=[
        *model.quantization_config.ignored_layers,
        "q_b_proj", "model.layers.0.self_attn", "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.88.mlp.experts.0.gate_proj",
    ])
    replica.model_config = replace(model, quantization_config=quant)
    assert ParamCounter(replica).get_parameter_memory_per_device_bytes() == 49_676_295_552


def test_empty_exclusions_keep_native_embedding_head_and_norms_in_bf16() -> None:
    replica = _replica()
    model = replica.model_config
    replica.model_config = replace(
        model, quantization_config=replace(model.quantization_config, ignored_layers=[]),
    )
    # Native FP8 methods do not quantize VocabParallelEmbedding/ParallelLMHead.
    counter = ParamCounter(replica)
    assert counter.get_num_parameters_per_device() == 49_000_424_448
    assert counter.get_parameter_memory_per_device_bytes() == 49_676_295_552


@pytest.mark.parametrize(("excluded", "exception", "message"), [
    ("model.layers.0.self_attn.q_a_proj", ValueError, "mixed shard precision"),
    ("model.layers.3.mlp.shared_experts.gate_proj", ValueError, "mixed shard precision"),
    ("checkpoint.model.layers.3.mlp.experts.0.gate_proj", NotImplementedError, "unquantized routed"),
    ("model.layers.0.self_attn.kv_b_proj", NotImplementedError, "BF16 KV-B alias"),
    ("model.layers.3.mlp.experts.0.gate_proj", NotImplementedError, "unquantized routed"),
])
def test_unrepresented_or_mixed_precision_exclusions_fail_explicitly(
    excluded: str, exception: type[Exception], message: str,
) -> None:
    replica = _replica()
    model = replica.model_config
    quant = replace(model.quantization_config, ignored_layers=[
        *model.quantization_config.ignored_layers, excluded,
    ])
    replica.model_config = replace(model, quantization_config=quant)
    with pytest.raises(exception, match=message):
        ParamCounter(replica).get_parameter_memory_per_device_bytes()


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
@pytest.mark.parametrize(("changes", "message"), [
    ({"torch_dtype": "float16"}, "BF16 model dtype"),
    ({"tie_word_embeddings": True}, "untied embeddings"),
    ({"tie_word_embeddings": None}, "untied embeddings"),
    ({"topk_method": "greedy"}, "noaux_tc"),
    ({"use_qkv_bias": True}, "bias-free"),
    ({"q_lora_rank": None}, "q_lora_rank"),
    ({"vocab_size": 154881}, "unpadded vocabulary"),
])
def test_unrepresented_model_contracts_fail_explicitly(
    role: ClusterType, changes: dict, message: str,
) -> None:
    replica = _replica(role=role)
    replica.model_config = replace(replica.model_config, **changes)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        ParamCounter(replica, role).get_parameter_memory_per_device_bytes()


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
@pytest.mark.parametrize(("changes", "message"), [
    ({"quant_method": "fp4", "weight_block_size": None}, "serialized dynamic block FP8"),
    ({"weight_block_size": None}, "128x128"),
    ({"weight_block_size": (64, 128)}, "128x128"),
    ({"scale_fmt": "ue8m0"}, "initial FP32 scales"),
    ({"quantized_operations": ["mlp_up_proj"]}, "native operation selection"),
])
def test_unrepresented_quantization_contracts_fail_explicitly(
    role: ClusterType, changes: dict, message: str,
) -> None:
    replica = _replica(role=role)
    model = replica.model_config
    replica.model_config = replace(
        model, quantization_config=replace(model.quantization_config, **changes),
    )
    with pytest.raises(NotImplementedError, match=message):
        ParamCounter(replica, role)


@pytest.mark.parametrize(
    "role", [ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, ClusterType.TRANS],
)
def test_unrepresented_role_ownership_is_rejected(role: ClusterType) -> None:
    with pytest.raises(NotImplementedError, match="PP1 full-model ownership"):
        ParamCounter(_replica(), role)


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
def test_unrepresented_pipeline_moe_tp_and_mtp_are_rejected(role: ClusterType) -> None:
    replica = _replica(role=role)
    replica.num_pipeline_stages = 2
    with pytest.raises(NotImplementedError, match="PP1 full-model ownership"):
        ParamCounter(replica, role)
    replica.num_pipeline_stages = 1
    replica.moe_tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match="MoE TP1"):
        ParamCounter(replica, role)
    replica.moe_tensor_parallel_size = 1
    replica.speculative_decoding_config.enabled = True
    with pytest.raises(NotImplementedError, match="exclude speculative"):
        ParamCounter(replica, role)


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
def test_replicated_shared_experts_need_a_separate_sp_contract(
    role: ClusterType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica = _replica(role=role)
    model = replica.model_config
    profile = model.get_model_architecture_profile()
    contracts = tuple(
        replace(contract, tensor_parallel_mode=TensorParallelMode.REPLICATED)
        if contract.layer_kind is LayerKind.SHARED else contract
        for contract in profile.layer_contracts
    )
    monkeypatch.setattr(model, "get_model_architecture_profile", lambda: replace(
        profile, layer_contracts=contracts,
    ))
    with pytest.raises(NotImplementedError, match="non-SP typed FFN ownership"):
        ParamCounter(replica, role).get_parameter_memory_per_device_bytes()


@pytest.mark.parametrize("role", FULL_MODEL_ROLES)
@pytest.mark.parametrize(("changes", "message"), [
    ({"q_lora_rank": 2050}, "output partitions"),
    ({"v_head_dim": 257}, "row-parallel O input"),
    ({"routed_mlp_hidden_dim": 2050}, "local FFN width"),
    ({"num_experts": 255}, "divisible"),
])
def test_incompatible_native_shard_geometry_is_rejected(
    role: ClusterType, changes: dict, message: str,
) -> None:
    replica = _replica(role=role)
    replica.model_config = replace(replica.model_config, **changes)
    with pytest.raises(ValueError, match=message):
        ParamCounter(replica, role).get_parameter_memory_per_device_bytes()


def test_existing_mla_count_and_legacy_ffn_byte_policy_are_preserved() -> None:
    replica = ReplicaConfig(model_name="deepseek-v3", device="h800")
    model = replica.model_config
    assert model.get_attention_family().family_id == "latent_mla_attention"
    counter = ParamCounter(replica)
    expected = 4 * model.embedding_dim * model.get_head_dim() * model.num_q_heads
    assert counter.get_num_attention_params_per_layer() == expected
    assert counter.get_mlp_parameter_memory_per_device_bytes() == 2 * counter.get_num_mlp_parameters_per_device()
    assert counter.get_parameter_memory_per_device_bytes() == 2 * counter.get_num_parameters_per_device()


def test_legacy_optional_tie_metadata_preserves_mtp_default_tied_head() -> None:
    replica = ReplicaConfig(model_name="qwen3-a3b-30b-moe", device="h800")
    model = replica.model_config
    assert model.tie_word_embeddings is None
    spec = replica.speculative_decoding_config
    spec.enabled = True
    spec.method = "qwen3_moe_mtp"
    spec.mtp_n_predict = 1
    spec.mtp_num_layers = 1
    default_count = ParamCounter(replica).get_num_mtp_parameters_per_device()
    assert default_count > 0
    replica.model_config = replace(model, tie_word_embeddings=True)
    assert ParamCounter(replica).get_num_mtp_parameters_per_device() == default_count
    replica.model_config = replace(model, tie_word_embeddings=False)
    expected_head = model.vocab_size * model.embedding_dim // replica.attn_tensor_parallel_size
    assert ParamCounter(replica).get_num_mtp_parameters_per_device() == default_count + expected_head
