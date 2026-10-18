"""Step source-required storage, distinct from native-observed unique allocations.

The independent arithmetic follows the locked Step model constructors, FP8
factories and vocabulary sharding. It is not calibrated or native fit evidence.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config import ReplicaConfig
from frontier.config.precision_type import PrecisionType
from frontier.model_architectures import LayerKind
from frontier.operators.spec import TensorParallelMode
from frontier.types import ActivationType, ClusterType
from frontier.utils.param_counter import ParamCounter


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = json.loads((REPO_ROOT / "data/config/models/stepfun-ai__Step-5-Preview.json").read_text())


def _replica(*, tp=8, dp=None, moe_tp=1, ep=32, device="h800"):
    return ReplicaConfig(
        model_name="stepfun-ai/Step-5-Preview", device=device,
        attn_tensor_parallel_size=tp,
        attn_dp=(ep * moe_tp // tp) if dp is None else dp,
        moe_tensor_parallel_size=moe_tp, moe_expert_parallel_size=ep,
        num_pipeline_stages=1,
    )


def _ceil(value, block):
    return (value + block - 1) // block


def _source_oracle(raw, *, tp, moe_tp, ep, norm_bytes=4, tied=False):
    """Unpacked constructor arithmetic independent of Frontier layer metadata.

    Native model.py:474/704/1136/1226/1711/1963/2075/2371 establish
    projections, replicas and norms. fp8.py:621/628 establishes expert scales.
    ExpertMapManager:60 and embedding:254/301 establish map and vocab storage.
    """
    text = raw["text_config"]
    h, experts = text["hidden_size"], text["moe_num_experts"]
    routed_ids = set(text["moe_layer_list"])
    excluded = raw["quantization_config"]["modules_to_not_convert"]
    bn, bk = raw["quantization_config"]["weight_block_size"]
    attention, ffn = [], []
    for layer_id, attention_type in enumerate(text["layer_types"][:text["num_hidden_layers"]]):
        sparse = attention_type == "full_attention"
        other = text["attention_other_setting"]
        q, kv, d = text["num_attention_heads"], text["num_attention_groups"], text["head_dim"]
        if attention_type == other["attention_type"]:
            q, kv, d = other["num_attention_heads"], other["num_attention_groups"], other["head_dim"]
        if not sparse:
            q = text["num_sliding_attention_heads"]
        local_q, local_kv = q // tp, max(1, kv // tp)
        # All main projections in the locked fixture are BF16. Fusion changes
        # module ownership, not the combined Q/K/V/gate coefficient count.
        projection_coefficients = h * ((local_q + 2 * local_kv) * d + local_q)
        projection_coefficients += h * local_q * d
        norm_coefficients = 2 * d
        attention_elements = projection_coefficients + norm_coefficients
        # Four registered FP32 attention scales remain resident with BF16 KV.
        attention_bytes = 2 * projection_coefficients + norm_bytes * norm_coefficients + 4 * 4
        if sparse:
            index = text["sparse_config"]
            providers = max(1, index["num_provider_groups"] // tp)
            iq = index["sparse_indexer_num_heads"] // index["num_provider_groups"] * providers
            ik, proxy = max(1, index["sparse_indexer_num_k_heads"] // tp), index["proxy_dim"]
            index_projection = h * ((iq + 2 * ik) * proxy + iq)
            # Index Q RMSNorm, K LayerNorm weight/bias inherit the BF16 loader
            # default. ssmax_s is retained FP32 with all logical Q heads.
            attention_elements += index_projection + 3 * proxy + q
            attention_bytes += 2 * (index_projection + 3 * proxy) + 4 * q
        attention.append((attention_elements, attention_bytes))
        if layer_id not in routed_ids:
            width = text["intermediate_size"] // tp
            coefficients = 3 * h * width
            ffn.append((coefficients, 2 * coefficients))
            continue
        local_experts = experts // ep
        width = text["moe_intermediate_size"] // moe_tp
        expert_coefficients = local_experts * 3 * h * width
        expert_prefix = f"model.layers.{layer_id}.moe.experts"
        if any(expert_prefix in name for name in excluded if "experts" in name):
            expert_bytes = 2 * expert_coefficients
        else:
            # The native auto loader refines the square checkpoint blocks
            # to the largest common divisor >=32 that fits a TP shard.
            block = max(size for size in range(32, min(bn, bk) + 1)
                        if bn % size == bk % size == width % size == h % size == 0)
            w13_scales = local_experts * 2 * _ceil(width, block) * _ceil(h, block)
            w2_scales = local_experts * _ceil(h, block) * _ceil(width, block)
            expert_bytes = expert_coefficients + 4 * (w13_scales + w2_scales)
        shared = 3 * h * (text["share_expert_dim"] // tp)
        gate_and_bias = experts * h + experts
        elements = expert_coefficients + shared + gate_and_bias
        storage = expert_bytes + 2 * shared + 4 * gate_and_bias
        if ep > 1:
            storage += 4 * experts  # One int32 map; not learned coefficients.
        ffn.append((elements, storage))
    endpoints = (1 if tied else 2) * text["vocab_size"] * h // tp
    norms = (2 * text["num_hidden_layers"] + 1) * h
    return attention, ffn, (endpoints + norms, 2 * endpoints + norm_bytes * norms)


@pytest.fixture
def replica(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return _replica()


@pytest.mark.parametrize(("tp", "dp", "moe_tp", "ep", "expected_storage"), [
    (8, 4, 1, 32, 23_158_461_632),
    (8, 2, 1, 16, 42_056_012_480),
    (16, 2, 1, 32, 21_573_276_864),
    (4, 2, 1, 8, 83_021_483_712),
    (4, 1, 4, 1, 158_611_563_200),
    (8, 1, 8, 1, 79_902_692_032),
    (16, 1, 16, 1, 40_599_958_208),
    (1, 1, 1, 1, 632_624_296_640),
])
def test_all_92_backbone_layers_match_independent_source_storage(
    tp, dp, moe_tp, ep, expected_storage, monkeypatch,
):
    monkeypatch.chdir(REPO_ROOT)
    counter = ParamCounter(_replica(tp=tp, dp=dp, moe_tp=moe_tp, ep=ep))
    attention, ffn, other = _source_oracle(RAW, tp=tp, moe_tp=moe_tp, ep=ep)
    for layer_id in range(92):
        assert counter.get_num_attention_params_per_layer(layer_id) == attention[layer_id][0]
        assert counter.get_num_mlp_params_per_layer(layer_id) == ffn[layer_id][0]
        assert counter.get_num_parameters_per_layer(layer_id) == (
            attention[layer_id][0] + ffn[layer_id][0] + 2 * 4096
        )
    assert counter.uses_source_parameter_storage is True
    assert counter.get_num_attention_parameters_per_device() == sum(count for count, _ in attention)
    assert counter.get_attention_parameter_memory_per_device_bytes() == sum(memory for _, memory in attention)
    assert counter.get_num_mlp_parameters_per_device() == sum(count for count, _ in ffn)
    assert counter.get_mlp_parameter_memory_per_device_bytes() == sum(memory for _, memory in ffn)
    assert counter.get_num_parameters_per_device() == (
        sum(count for count, _ in attention) + sum(count for count, _ in ffn) + other[0]
    )
    assert counter.get_parameter_memory_per_device_bytes() == (
        sum(memory for _, memory in attention) + sum(memory for _, memory in ffn) + other[1]
    ) == expected_storage
    assert counter.get_attention_stage_layer_counts() == ((0, 92),)
    assert counter.get_resident_attention_stage_id() == 0
    assert counter.get_num_mtp_parameters_per_device() == 0
    assert counter.get_conditional_mhc_broadcast_memory_bytes() == 0
    assert counter.get_parameter_memory_per_device_bytes() != 2 * counter.get_num_parameters_per_device()


def test_native_head_norm_and_provider_replicas_are_not_divided_again(replica):
    counter = ParamCounter(replica)
    assert counter.get_num_attention_params_per_layer(0) == 14_188_928
    assert counter.get_num_attention_params_per_layer(3) == 20_497_600
    assert counter.get_attention_parameter_memory_per_device_bytes() == 2_901_036_736
    # TP8 sparse packed3464x4096, provider weights4x4096, output4096x1536.
    projection_count = 3464 * 4096 + 4 * 4096 + 4096 * 1536
    assert counter.get_num_attention_params_per_layer(3) == projection_count + 2 * 192 + 3 * 256 + 64
    assert counter._get_step5_attention_layer_totals(3)[1] == (
        2 * projection_count + 4 * 2 * 192 + 2 * 3 * 256 + 4 * 64 + 4 * 4
    )
    # Increasing TP beyond provider/KV count retains complete provider heads.
    tp16 = ParamCounter(_replica(tp=16, dp=2, ep=32))
    assert counter.get_num_attention_params_per_layer(3) > tp16.get_num_attention_params_per_layer(3)
    assert tp16.get_num_attention_params_per_layer(3) > counter.get_num_attention_params_per_layer(3) // 2


@pytest.mark.parametrize(("layer_id", "matrix_elements", "index_norms", "ssmax"), [
    (0, 4096 * (8 * 192 + 2 * 192 + 8) + 4096 * 8 * 192, 0, 0),
    (3, 4096 * 3464 + 4096 * 4 + 4096 * 1536, 3 * 256, 64),
])
def test_attention_scalar_buffers_are_storage_without_learned_coefficients(
    replica, layer_id, matrix_elements, index_norms, ssmax,
):
    counter = ParamCounter(replica)
    # attention.py:134-140 retains four FP32 buffers rather than parameters.
    norms = 2 * 192
    elements = matrix_elements + norms + index_norms + ssmax
    storage = 2 * matrix_elements + 4 * norms + 2 * index_norms + 4 * ssmax
    assert counter._get_step5_attention_layer_totals(layer_id) == (elements, storage + 16)


def test_fp8_scales_bf16_exclusions_and_router_alias_have_separate_storage(replica):
    counter = ParamCounter(replica)
    coefficients = 11 * 3 * 4096 * 1536
    scales = 11 * (2 * 12 * 32 + 32 * 12)
    shared = 3 * 4096 * 192
    router = 352 * 4096 + 352
    learned = coefficients + shared + router
    fp8_storage = coefficients + 4 * scales + 2 * shared + 4 * router + 4 * 352
    bf16_storage = 2 * coefficients + 2 * shared + 4 * router + 4 * 352
    assert counter._get_step5_mlp_layer_totals(3) == (learned, fp8_storage)
    for layer_id in (88, 89, 90):
        assert counter._get_step5_mlp_layer_totals(layer_id) == (learned, bf16_storage)
    assert sum(counter._get_step5_mlp_layer_totals(i)[1] == fp8_storage for i in range(92)) == 85
    # Router weight and correction bias remain FP32 even without an exclusion.
    model = replica.model_config
    exclusions = [name for name in model.quantization_config.ignored_layers if name != "model.layers.3.moe.gate"]
    replica.model_config = replace(model, quantization_config=replace(model.quantization_config, ignored_layers=exclusions))
    assert ParamCounter(replica)._get_step5_mlp_layer_totals(3) == (learned, fp8_storage)


def test_embedding_head_tie_and_final_norm_have_native_ownership(replica):
    before = ParamCounter(replica)
    model = replica.model_config
    backbone = sum(before.get_num_parameters_per_layer(i) for i in range(92))
    endpoint = 128896 * 4096 // 8
    assert before.get_num_parameters_per_device() - backbone == 2 * endpoint + 4096
    assert before._get_step5_other_totals() == (132_747_264, 267_010_048)
    replica.model_config = replace(model, tie_word_embeddings=True)
    tied = ParamCounter(replica)
    assert before.get_num_parameters_per_device() - tied.get_num_parameters_per_device() == endpoint
    assert before.get_parameter_memory_per_device_bytes() - tied.get_parameter_memory_per_device_bytes() == 2 * endpoint
    exclusions = [name for name in model.quantization_config.ignored_layers if name != "lm_head"]
    replica.model_config = replace(model, quantization_config=replace(model.quantization_config, ignored_layers=exclusions))
    assert ParamCounter(replica).get_parameter_memory_per_device_bytes() == before.get_parameter_memory_per_device_bytes()


def test_main_norm_precision_does_not_change_indexer_router_precision(replica):
    before = ParamCounter(replica)
    model = replica.model_config
    replica.model_config = replace(model, norm_dtype="bfloat16")
    after = ParamCounter(replica)
    main_norms = (2 * 92 + 1) * 4096 + 92 * 2 * 192
    assert before.get_num_parameters_per_device() == after.get_num_parameters_per_device()
    assert before.get_parameter_memory_per_device_bytes() - after.get_parameter_memory_per_device_bytes() == 2 * main_norms
    assert before.get_mlp_parameter_memory_per_device_bytes() == after.get_mlp_parameter_memory_per_device_bytes()
    assert before._get_step5_attention_layer_totals(3)[1] - after._get_step5_attention_layer_totals(3)[1] == 2 * 2 * 192


def test_shape_changes_are_config_driven_without_hidden_per_head_inference(replica):
    raw = deepcopy(RAW)
    raw["text_config"]["hidden_size"] = 3840
    replica.model_config = replace(replica.model_config, embedding_dim=3840)
    counter = ParamCounter(replica)
    attention, ffn, other = _source_oracle(raw, tp=8, moe_tp=1, ep=32)
    assert counter.get_parameter_memory_per_device_bytes() == (
        sum(memory for _, memory in attention) + sum(memory for _, memory in ffn) + other[1]
    )
    assert counter._get_step5_layer(3).geometry.head_dim == 192
    assert replica.model_config.embedding_dim // replica.model_config.num_q_heads == 60


def test_topk_mtp_inventory_and_rotary_length_do_not_duplicate_storage(replica):
    before = ParamCounter(replica)
    model = replica.model_config
    replica.model_config = replace(
        model, num_experts_per_tok=4, max_position_embeddings=8192,
        sparse_config={**model.sparse_config, "topk": 256},
    )
    after = ParamCounter(replica)
    assert model.num_nextn_predict_layers == 3
    assert before.get_parameter_memory_per_device_bytes() == after.get_parameter_memory_per_device_bytes()
    assert before.get_num_parameters_per_device() == after.get_num_parameters_per_device()
    for layer_id in (92, 93, 94, -1, True):
        for getter in (after.get_num_attention_params_per_layer, after.get_num_mlp_params_per_layer, after.get_num_parameters_per_layer):
            with pytest.raises(ValueError, match="out of range"):
                getter(layer_id)


def _remove_exclusions(model, suffixes, layer_id):
    prefixes = {f"model.layers.{layer_id}.{suffix}" for suffix in suffixes}
    return replace(model, quantization_config=replace(
        model.quantization_config,
        ignored_layers=[name for name in model.quantization_config.ignored_layers if name not in prefixes],
    ))


@pytest.mark.parametrize(("suffixes", "layer_id", "coefficients", "scale_elements"), [
    (("self_attn.o_proj",), 3, 4096 * 1536, 32 * 12),
    (("mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj"), 0, 4096 * 4096, 32 * 32),
    (("mlp.down_proj",), 0, 4096 * 2048, 32 * 16),
])
def test_linear_fp8_scale_shape_and_exclusion_precedence(
    replica, suffixes, layer_id, coefficients, scale_elements,
):
    # Dense TP8 width1728 does not align, so use an aligned width before
    # removing packed exclusions; native merged/row-parallel constraints apply.
    model = replace(replica.model_config, dense_mlp_hidden_dim=16384)
    replica.model_config = model
    before = ParamCounter(replica)
    replica.model_config = _remove_exclusions(model, suffixes, layer_id)
    after = ParamCounter(replica)
    assert before.get_num_parameters_per_device() == after.get_num_parameters_per_device()
    assert before.get_parameter_memory_per_device_bytes() - after.get_parameter_memory_per_device_bytes() == coefficients - 4 * scale_elements


@pytest.mark.parametrize("exclude", [
    "model.layers.3.moe.experts", "model.layers.3.moe.experts.w13_weight",
    "checkpoint.model.layers.3.moe.experts.w2_weight",
])
def test_native_expert_substring_exclusion_quantizes_whole_module(replica, exclude):
    model = replica.model_config
    before = ParamCounter(replica)
    replica.model_config = replace(model, quantization_config=replace(
        model.quantization_config, ignored_layers=[*model.quantization_config.ignored_layers, exclude],
    ))
    after = ParamCounter(replica)
    # The native method matches the expert module, including either packed
    # weight's exclusion. Both matrices lose FP8 scales and become BF16.
    coefficients, scale_elements = 11 * 3 * 4096 * 1536, 11 * 1152
    assert after.get_num_parameters_per_device() == before.get_num_parameters_per_device()
    assert after.get_parameter_memory_per_device_bytes() - before.get_parameter_memory_per_device_bytes() == coefficients - 4 * scale_elements


def test_shared_prefix_rename_and_exact_packed_precedence(replica):
    model = replace(replica.model_config, share_expert_dim=2048)
    suffixes = ("share_expert.gate_up_proj", "share_expert.gate_proj", "share_expert.up_proj")
    replica.model_config = _remove_exclusions(model, suffixes, 3)
    fp8 = ParamCounter(replica)
    quant = replica.model_config.quantization_config
    replica.model_config = replace(replica.model_config, quantization_config=replace(
        quant, ignored_layers=[*quant.ignored_layers, "model.layers.3.share_expert.gate_up_proj", "model.layers.3.share_expert.gate_proj"],
    ))
    bf16 = ParamCounter(replica)
    assert bf16.get_num_parameters_per_device() == fp8.get_num_parameters_per_device()
    assert bf16.get_parameter_memory_per_device_bytes() - fp8.get_parameter_memory_per_device_bytes() == 512 * 4096 - 4 * 4 * 32


@pytest.mark.parametrize(("suffixes", "message"), [
    (("self_attn.qkv_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "linear partitions"),
    (("self_attn.g_proj",), "linear partitions"),
    (("mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj"), "linear partitions"),
    (("mlp.down_proj",), "linear partitions"),
])
def test_native_unaligned_fp8_linear_storage_is_rejected(replica, suffixes, message):
    replica.model_config = _remove_exclusions(replica.model_config, suffixes, 0)
    with pytest.raises(NotImplementedError, match=message):
        ParamCounter(replica)


def test_partial_packed_exclusion_does_not_silently_change_precision(replica):
    replica.model_config = _remove_exclusions(
        replica.model_config, ("self_attn.qkv_proj", "self_attn.q_proj"), 0,
    )
    with pytest.raises(ValueError, match="mixed quantization exclusions"):
        ParamCounter(replica)


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, ClusterType.TRANS])
def test_source_counts_do_not_claim_pd_or_partial_role_ownership(replica, role):
    with pytest.raises(NotImplementedError, match="MONOLITHIC"):
        ParamCounter(replica, role)


@pytest.mark.parametrize(("attribute", "value", "message"), [
    ("num_pipeline_stages", 2, "PP1"),
    ("attn_tensor_parallel_size", True, "positive int"),
    ("attn_dp", 0, "positive int"),
    ("moe_tensor_parallel_size", 0, "positive int"),
    ("moe_expert_parallel_size", True, "positive int"),
    ("moe_tensor_parallel_size", 2, "flattened TP/DP or full EP"),
    ("moe_expert_parallel_size", 16, "flattened TP/DP or full EP"),
])
def test_unsupported_or_invalid_parallel_domains_fail_before_counts(replica, attribute, value, message):
    setattr(replica, attribute, value)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        ParamCounter(replica)


def test_tp16_auto_refined_scales_match_retained_native_shapes(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    counter = ParamCounter(_replica(tp=16, dp=1, moe_tp=16, ep=1))
    # g2_execution_proposal.md:205-212 records these resident FP32 shapes.
    # This checks constructor storage, not executed kernel or scale contents.
    w13_elements = 352 * 6 * 128
    w2_elements = 352 * 128 * 3
    expert_coefficients = 352 * 3 * 4096 * 96
    shared_coefficients = 3 * 4096 * 96
    router_coefficients = 352 * 4096 + 352
    non_scale_bytes = expert_coefficients + 2 * shared_coefficients + 4 * router_coefficients
    scale_bytes = 4 * (w13_elements + w2_elements)
    for layer_id in range(3, 88):
        assert counter._get_step5_mlp_layer_totals(layer_id) == (
            expert_coefficients + shared_coefficients + router_coefficients,
            non_scale_bytes + scale_bytes,
        )
    assert 85 * scale_bytes == 137_871_360
    assert counter.get_num_parameters_per_device() == 37_750_873_536
    assert counter.get_attention_parameter_memory_per_device_bytes() == 1_740_394_176
    assert counter.get_mlp_parameter_memory_per_device_bytes() == 38_724_543_488
    assert counter._get_step5_other_totals()[1] == 135_020_544
    assert counter.get_parameter_memory_per_device_bytes() == 40_599_958_208


def test_expert_shards_without_supported_unpadded_refinement_are_explicit(replica):
    replica = _replica(tp=16, dp=1, moe_tp=16, ep=1)
    replica.model_config = replace(replica.model_config, routed_mlp_hidden_dim=1280)
    with pytest.raises(NotImplementedError, match="unpadded block refinement of at least 32"):
        ParamCounter(replica)


@pytest.mark.parametrize("tp", [32, 64])
def test_unsupported_sparse_head_partition_is_rejected_before_source_counts(tp, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    replica = _replica(tp=tp, dp=1, moe_tp=tp, ep=1)
    with pytest.raises(ValueError, match="local Q heads per KV group"):
        ParamCounter(replica)


def test_non_block_aligned_expert_hidden_width_is_not_claimed(replica):
    replica.model_config = replace(replica.model_config, embedding_dim=4128)
    with pytest.raises(NotImplementedError, match="unpadded hidden block alignment"):
        ParamCounter(replica)


def test_replicated_sp_shared_contract_is_not_treated_as_ordinary_tp(replica, monkeypatch):
    model = replica.model_config
    profile = model.get_model_architecture_profile()
    contracts = tuple(
        replace(contract, tensor_parallel_mode=TensorParallelMode.REPLICATED)
        if contract.layer_kind is LayerKind.SHARED else contract
        for contract in profile.layer_contracts
    )
    monkeypatch.setattr(model, "get_model_architecture_profile", lambda: replace(profile, layer_contracts=contracts))
    with pytest.raises(NotImplementedError, match="MegaMoE/SP"):
        ParamCounter(replica)


@pytest.mark.parametrize(("field", "value"), [
    ("vocab_size", 128897), ("vocab_size", True), ("torch_dtype", "float16"),
    ("expert_dtype", "fp4"), ("activation", ActivationType.GELU),
    ("use_gated_mlp", False),
])
def test_unsupported_model_storage_metadata_is_explicit(replica, field, value):
    with pytest.raises((ValueError, NotImplementedError), match="Step"):
        replica.model_config = replace(replica.model_config, **{field: value})
        ParamCounter(replica)


@pytest.mark.parametrize(("changes", "message"), [
    ({"quant_method": "fp4", "weight_block_size": None}, "serialized dynamic"),
    ({"quant_method": None, "weight_block_size": None}, "serialized dynamic"),
    ({"activation_scheme": "static", "weight_block_size": None}, "serialized dynamic"),
    ({"is_checkpoint_fp8_serialized": False, "weight_block_size": None}, "serialized dynamic"),
    ({"scale_fmt": "ue8m0"}, "initial FP32 scales"),
    ({"weight_block_size": (64, 128)}, "128x128"),
    ({"quantized_operations": ["moe_grouped_gemm"]}, "initial FP32 scales"),
])
def test_unrepresented_quantized_layouts_fail_before_coefficient_getters(replica, changes, message):
    model = replica.model_config
    replica.model_config = replace(model, quantization_config=replace(model.quantization_config, **changes))
    with pytest.raises(NotImplementedError, match=message):
        ParamCounter(replica)


@pytest.mark.parametrize("mutation", ["missing", "extra_mtp", "wrong_family", "wrong_variant", "index_owner"])
def test_backbone_binding_inventory_must_be_complete_and_canonical(replica, monkeypatch, mutation):
    model = replica.model_config
    specs = model.get_layer_attention_specs()
    if mutation == "missing":
        changed = specs[:-1]
    elif mutation == "extra_mtp":
        changed = (*specs, replace(specs[-1], global_layer_id=92))
    elif mutation == "wrong_family":
        changed = (replace(specs[0], family_id="deepseek_v4_attention"), *specs[1:])
    elif mutation == "wrong_variant":
        changed = (replace(specs[0], variant_id="sparse_gqa"), *specs[1:])
    else:
        changed = (*specs[:3], replace(specs[3], index_producer_layer_id=3), *specs[4:])
    monkeypatch.setattr(model, "get_layer_attention_specs", lambda: changed)
    with pytest.raises(ValueError, match="Step"):
        ParamCounter(replica)


def test_mtp_runtime_is_rejected_even_with_backbone_metadata(replica):
    replica.speculative_decoding_config.enabled = True
    with pytest.raises(NotImplementedError, match="speculative decoding/MTP"):
        ParamCounter(replica)


def test_mixed_step_inventory_uses_layer_metadata_without_homogeneous_api(replica, monkeypatch):
    def forbidden():
        raise AssertionError("Step mixed layers must not select a homogeneous family")

    monkeypatch.setattr(replica.model_config, "get_attention_family", forbidden)
    counter = ParamCounter(replica)
    metadata = describe_step5_layer(replica.model_config, 3, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=32)
    assert metadata.geometry.local_provider_groups == 1
    assert metadata.geometry.provider_replication == 2
    assert metadata.router_precision is PrecisionType.FP32
    assert counter.get_parameter_memory_per_device_bytes() == 23_158_461_632


def test_h200_does_not_change_source_required_weight_storage(replica):
    before = ParamCounter(replica).get_parameter_memory_per_device_bytes()
    replica.device = "h200"
    assert ParamCounter(replica).get_parameter_memory_per_device_bytes() == before
