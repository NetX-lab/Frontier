"""CPU contracts for Step linear profiling outside the attention physical scope."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.model_architectures import (
    AttentionLinearOpImplementation,
    AttentionLinearOpProfile,
    ModelArchitectureProfile,
    ModelArchitectureRegistry,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.profiling.utils.replicated_ops import (
    deduplicate_tp1_rows,
    split_replicated_result,
)
from frontier.operators.families import STEP5_FFN_FAMILY
from frontier.operators.typed_contracts import (
    parse_typed_operator_contracts,
    serialize_typed_operator_contracts,
)


@pytest.fixture(
    params=[ModelConfig.from_model_name, BaseModelConfig.create_from_name],
    ids=["profiling", "runtime"],
)
def config(request):
    return request.param("stepfun-ai/Step-5-Preview")


def test_attention_linear_profile_allows_no_attention_operators():
    profile = AttentionLinearOpProfile(
        sharded_impl=AttentionLinearOpImplementation.GENERIC,
        sharded_ops=(),
    )

    profile.require_enabled_for_profiling()
    assert profile.sharded_ops == profile.replicated_ops == ()


def test_attention_linear_profile_rejects_replicated_only_operators():
    with pytest.raises(ValueError, match="replicated_ops require"):
        AttentionLinearOpProfile(
            sharded_impl=AttentionLinearOpImplementation.GENERIC,
            sharded_ops=(),
            replicated_ops=("attn_pre_proj_qkv",),
        )


def test_step_attention_scope_owns_projections_without_enabling_execution(config):
    profile = config.get_model_architecture_profile()

    assert profile.attention_linear_ops.sharded_ops == ()
    assert profile.attention_linear_ops.replicated_ops == ()
    profile.attention_linear_ops.require_enabled_for_profiling()
    profile.validate_structural_requirements(config)
    assert STEP5_ATTENTION_FAMILY.disjoint_model_projection_attrs == ()
    assert STEP5_ATTENTION_FAMILY.execution_enabled is False
    assert STEP5_ATTENTION_FAMILY.structural_execution_enabled is True


def test_next_profile_requires_attention_physical_projection_ownership(config):
    profile = replace(
        ModelArchitectureProfile.generic(),
        profile_id="unit_physical_attention_owner",
        attention_linear_ops=AttentionLinearOpProfile(
            sharded_impl=AttentionLinearOpImplementation.GENERIC,
            sharded_ops=(),
        ),
        match=lambda _config: True,
    )
    registry = ModelArchitectureRegistry()
    registry.register(profile)
    selected = registry.resolve(config)
    selected.validate_structural_requirements(config)
    with pytest.raises(ValueError, match="unit_physical_attention_owner.*physical scope"):
        selected.validate_structural_requirements(SimpleNamespace(
            model_type="generic", use_mla=False, use_mfa=False,
            num_q_heads=8, num_kv_heads=8,
        ))


@pytest.mark.parametrize("is_moe", [False, True])
def test_step_linear_plan_keeps_ffn_memory_and_typed_branch_merge(config, is_moe):
    plan = build_profiling_plan(
        config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=is_moe,
    )

    assert plan["attn_sharded_enabled"] is False
    assert plan["attn_enabled"] is True
    assert plan["ffn_sharded_enabled"] is True
    assert set(plan["enabled_ops"]) == {
        "input_layernorm", "post_attention_layernorm", "emb",
        "share_expert_up_proj", "share_expert_down_proj", "share_expert_act",
        "step5_ffn_merge",
    } | (set() if is_moe else {"mlp_up_proj", "mlp_down_proj", "mlp_act"})
    assert set(plan["typed_operator_contracts"]) == set(plan["enabled_ops"])
    assert plan["padded_n_expanded_embd"] == (1536 if is_moe else 13824)
    merge = plan["typed_operator_contracts"]["step5_ffn_merge"]
    assert merge["operator_family_ids"] == ["step5_ffn"]
    assert merge["layer_kind"] is None
    assert merge["effective_ffn_width"] is None
    assert merge["selected_padded_ffn_width"] is None
    assert merge["tensor_parallel_mode"] == "replicated"
    assert merge["tensor_parallel_sizes"] == [1]
    assert merge["selected_tensor_parallel_size"] == 1
    assert plan["replicated_ops"] == [
        "input_layernorm", "post_attention_layernorm", "emb", "step5_ffn_merge",
    ]


def test_step_ffn_merge_obeys_disable_replicated_like_memory(config):
    plan = build_profiling_plan(
        config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1],
        is_moe=True, disable_replicated=True,
    )

    assert plan["attn_enabled"] is False
    assert plan["replicated_ops"] == []
    assert set(plan["enabled_ops"]) == {
        "share_expert_up_proj", "share_expert_down_proj", "share_expert_act",
    }
    assert "step5_ffn_merge" in plan["disabled_ops"]


def test_step_ffn_merge_serializes_in_the_full_hidden_tp1_row(config):
    plan = build_profiling_plan(
        config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1], is_moe=True,
    )
    sharded, replicated = split_replicated_result(
        {
            "num_tensor_parallel_workers": 8,
            "padded_n_embd": 4096,
            "padded_n_expanded_embd": 1536,
            "time_stats": dict.fromkeys(plan["enabled_ops"], 1.0),
            "typed_operator_contracts": plan["typed_operator_contracts"],
        },
        set(plan["replicated_ops"]),
        unpadded_n_embd=config.embedding_dim,
    )

    assert sharded["num_tensor_parallel_workers"] == 8
    assert "step5_ffn_merge" not in sharded["time_stats"]
    assert "step5_ffn_merge" not in sharded["typed_operator_contracts"]
    assert replicated["num_tensor_parallel_workers"] == 1
    assert replicated["padded_n_embd"] == 4096
    assert "step5_ffn_merge" in replicated["time_stats"]
    contracts = parse_typed_operator_contracts(
        serialize_typed_operator_contracts(replicated["typed_operator_contracts"])
    )
    assert contracts["step5_ffn_merge"] == plan["typed_operator_contracts"]["step5_ffn_merge"]
    assert STEP5_FFN_FAMILY.execution_enabled is False


@pytest.mark.parametrize("share_expert", [False, True])
def test_profile_resolves_step_ffn_activation_without_changing_generic(config, share_expert):
    profile = config.get_model_architecture_profile()
    assert profile.resolve_ffn_activation(config, share_expert=share_expert) == ("swiglustep", 7.0)
    assert ModelArchitectureProfile.generic().resolve_ffn_activation(
        config, share_expert=share_expert,
    ) == ("silu", None)


@pytest.mark.parametrize("share_expert", [False, True])
def test_step_ffn_activation_rejects_heterogeneous_limits(config, share_expert):
    changed = deepcopy(config)
    limits = list(changed.swiglu_limits_shared)
    limits[3 if share_expert else 0] = 5.0
    changed.swiglu_limits_shared = tuple(limits)

    with pytest.raises(ValueError, match="heterogeneous.*SwiGLU"):
        changed.get_model_architecture_profile().resolve_ffn_activation(
            changed, share_expert=share_expert,
        )


@pytest.mark.parametrize("share_expert", [False, True])
def test_step_ffn_activation_uses_only_its_dense_or_shared_partition(config, share_expert):
    changed = deepcopy(config)
    limits = list(changed.swiglu_limits_shared)
    for layer_id in ((0, 1) if share_expert else (3, 4)):
        limits[layer_id] = 5.0 + layer_id
    changed.swiglu_limits_shared = tuple(limits)

    assert changed.get_model_architecture_profile().resolve_ffn_activation(
        changed, share_expert=share_expert,
    ) == ("swiglustep", 7.0)


def test_normalization_implementation_is_classified_only_on_the_profile(config):
    profile = config.get_model_architecture_profile()
    assert profile.linear_norm_impl.value == "step5"
    assert ModelArchitectureProfile.generic().linear_norm_impl.value == "generic"
    assert ModelArchitectureProfile.step3_text().linear_norm_impl.value == "generic"
    assert ModelArchitectureProfile.qwen3_5_moe().linear_norm_impl.value == "generic"
    assert profile.experimental is True


@pytest.mark.parametrize("share_expert", [False, True])
def test_step_ffn_activation_supports_an_unclamped_partition(config, share_expert):
    changed = deepcopy(config)
    changed.swiglu_limits_shared = tuple(0.0 for _ in changed.swiglu_limits_shared)

    assert changed.get_model_architecture_profile().resolve_ffn_activation(
        changed, share_expert=share_expert,
    ) == ("silu", None)


def test_next_profile_reuses_activation_and_norm_classification_without_caller_changes(config):
    profile = replace(
        config.get_model_architecture_profile(),
        profile_id="unit_step_activation",
        match=lambda _config: True,
    )
    registry = ModelArchitectureRegistry()
    registry.register(profile)
    selected = registry.resolve(config)

    assert selected.profile_id == "unit_step_activation"
    assert selected.resolve_ffn_activation(config, share_expert=True) == ("swiglustep", 7.0)
    assert selected.linear_norm_impl.value == "step5"


def test_real_step_planned_profile_retains_all_replicated_merge_rows(tmp_path):
    pytest.importorskip("torch")
    from frontier.profiling.linear_op.step5_job import planned_profile

    profile = planned_profile(tmp_path)
    merge_rows = [
        row for row in profile["planned_csv_rows"]
        if "step5_ffn_merge" in row["operators"]
    ]

    assert len(merge_rows) == 3
    assert {row["num_tokens"] for row in merge_rows} == {1, 16, 64}
    assert all(row["num_tensor_parallel_workers"] == 1 for row in merge_rows)
    assert all(
        row["typed_operator_contracts"]["step5_ffn_merge"]["tensor_parallel_mode"]
        == "replicated"
        for row in merge_rows
    )


@pytest.fixture
def replicated_observations(config):
    plan = build_profiling_plan(
        config, tp_size=8, attn_tp=[8], ffn_tp=[8], moe_tp=[1],
    )
    contracts = plan["typed_operator_contracts"]
    first = {
        "num_tokens": 16,
        "num_tensor_parallel_workers": 1,
        "model_arch": config.model_arch,
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "measurement_type": "CUDA_EVENT",
        "n_embd": 4096,
        "padded_n_embd": 4096,
        "padded_n_expanded_embd": 1536,
        "time_stats": {"input_layernorm": {"mean": 1.0, "kernel_names": ["first_norm"]}},
        "typed_operator_contracts": {"input_layernorm": contracts["input_layernorm"]},
        "linear_backends": json.dumps({"input_layernorm": ["first_norm_backend"]}),
    }
    later = {
        **deepcopy(first),
        "padded_n_expanded_embd": 13824,
        "time_stats": {
            "input_layernorm": {"mean": 9.0, "kernel_names": ["later_norm"]},
            "step5_ffn_merge": {"mean": 2.0, "kernel_names": ["native_add_fp32"]},
        },
        "typed_operator_contracts": {
            name: contracts[name] for name in ("input_layernorm", "step5_ffn_merge")
        },
        "linear_backends": json.dumps({
            "input_layernorm": ["later_norm_backend"],
            "step5_ffn_merge": ["native_add_fp32_backend"],
        }),
    }
    return deepcopy(first), deepcopy(later)


def test_tp1_dedup_preserves_disjoint_observations_and_their_metadata(replicated_observations):
    first, later = replicated_observations
    original = deepcopy([first, later])

    rows = deduplicate_tp1_rows([first, later])

    assert len(rows) == 1
    assert rows[0]["time_stats"] == {
        "input_layernorm": {"mean": 1.0, "kernel_names": ["first_norm"]},
        "step5_ffn_merge": {"mean": 2.0, "kernel_names": ["native_add_fp32"]},
    }
    assert rows[0]["typed_operator_contracts"] == later["typed_operator_contracts"]
    assert rows[0]["padded_n_expanded_embd"] == 1536
    assert json.loads(rows[0]["linear_backends"]) == {
        "input_layernorm": ["first_norm_backend"],
        "step5_ffn_merge": ["native_add_fp32_backend"],
    }
    assert [first, later] == original


@pytest.mark.parametrize("typed", [False, True])
def test_tp1_duplicate_observations_keep_the_first_sample(replicated_observations, typed):
    first, later = replicated_observations
    later["time_stats"].pop("step5_ffn_merge")
    later["typed_operator_contracts"].pop("step5_ffn_merge")
    if not typed:
        first.pop("typed_operator_contracts")
        later.pop("typed_operator_contracts")
        later["padded_n_expanded_embd"] = first["padded_n_expanded_embd"]

    rows = deduplicate_tp1_rows([first, later])

    assert rows == [first]
    assert rows[0] is first


@pytest.mark.parametrize("field,value", [
    ("n_embd", 8192),
    ("padded_n_embd", 8192),
    ("measurement_type", "RECORD_FUNCTION"),
    ("model_architecture_profile", "unit_other_profile"),
])
def test_tp1_dedup_does_not_combine_conflicting_row_context(
    replicated_observations, field, value,
):
    first, later = replicated_observations
    later[field] = value

    assert deduplicate_tp1_rows([first, later]) == [first, later]


def test_tp1_dedup_does_not_combine_conflicting_overlap_contracts(replicated_observations):
    first, later = replicated_observations
    later["typed_operator_contracts"]["input_layernorm"]["selected_tensor_parallel_size"] = 2

    assert deduplicate_tp1_rows([first, later]) == [first, later]


def test_tp1_dedup_does_not_union_disjoint_profiles(replicated_observations):
    first, later = replicated_observations
    later["time_stats"].pop("input_layernorm")
    later["typed_operator_contracts"].pop("input_layernorm")
    later["typed_operator_contracts"]["step5_ffn_merge"]["profile_id"] = "unit_other_profile"

    assert deduplicate_tp1_rows([first, later]) == [first, later]


@pytest.mark.parametrize("same_operator", [False, True])
def test_tp1_dedup_does_not_misattribute_active_ffn_shape_sources(
    config, replicated_observations, same_operator,
):
    first, later = replicated_observations
    plan = build_profiling_plan(
        config, tp_size=1, attn_tp=[1], ffn_tp=[1], moe_tp=[1],
    )
    first["time_stats"] = {"mlp_up_proj": {"mean": 1.0}}
    first["typed_operator_contracts"] = {
        "mlp_up_proj": plan["typed_operator_contracts"]["mlp_up_proj"],
    }
    name = "mlp_up_proj" if same_operator else "mlp_down_proj"
    later["time_stats"] = {name: {"mean": 2.0}}
    later["typed_operator_contracts"] = {
        name: {
            **plan["typed_operator_contracts"][name],
            "effective_ffn_width": 1536,
            "selected_padded_ffn_width": 1536,
        },
    }
    later["padded_n_expanded_embd"] = first["padded_n_expanded_embd"]

    assert deduplicate_tp1_rows([first, later]) == [first, later]


def test_tp1_dedup_adds_only_the_new_operators_backend(replicated_observations):
    first, later = replicated_observations
    first.pop("linear_backends")

    row, = deduplicate_tp1_rows([first, later])

    assert json.loads(row["linear_backends"]) == {
        "step5_ffn_merge": ["native_add_fp32_backend"],
    }


def test_tp1_duplicate_memory_observations_keep_first_active_ffn_row(
    config, replicated_observations,
):
    first, later = replicated_observations
    plan = build_profiling_plan(
        config, tp_size=1, attn_tp=[1], ffn_tp=[1], moe_tp=[1],
    )
    first["time_stats"]["mlp_up_proj"] = {"mean": 1.0}
    first["typed_operator_contracts"]["mlp_up_proj"] = plan["typed_operator_contracts"]["mlp_up_proj"]
    first["padded_n_expanded_embd"] = 13824
    later["time_stats"].pop("step5_ffn_merge")
    later["typed_operator_contracts"].pop("step5_ffn_merge")
    later["padded_n_expanded_embd"] = 1536

    assert deduplicate_tp1_rows([first, later]) == [first]


def test_dedup_leaves_sharded_rows_unchanged(replicated_observations):
    first, later = replicated_observations
    first["num_tensor_parallel_workers"] = 8
    later["num_tensor_parallel_workers"] = 8

    assert deduplicate_tp1_rows([first, later]) == [first, later]


def test_real_linear_sweep_materializes_deduplicates_and_serializes_every_merge(config):
    pytest.importorskip("torch")
    import pandas as pd
    from frontier.profiling.linear_op import main
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper

    args = SimpleNamespace(
        attn_tp=[8], ffn_tp=[8], moe_tp=[1], num_tensor_parallel_workers=[8],
        disable_replicated=False, is_moe=False, include_target_embedded_mtp=False,
        sequence_parallel_moe_topology=None,
    )
    rows = []
    for method in ("CUDA_EVENT", "RECORD_FUNCTION"):
        for plan in main._build_profiling_plans(args, config):
            wrapper = object.__new__(LinearOpWrapper)
            wrapper.model_config = config
            wrapper.num_tensor_parallel_workers = plan["tp_size"]
            wrapper.profiling_plan = plan
            for tokens in (1, 16, 64):
                result = wrapper._build_profile_result({
                    name: {"mean": float(plan["tp_size"]), "kernel_names": [name]}
                    for name in plan["enabled_ops"]
                }, num_tokens=tokens)
                if "step5_ffn_merge" in plan["enabled_ops"]:
                    result["linear_backends"] = {
                        "step5_ffn_merge": ["native_add_fp32_backend"],
                        "share_expert_up_proj": ["linear_backend"],
                    }
                rows.extend(main._materialize_profile_result_rows(
                    result, measurement_type=method,
                    should_split=bool(plan["tp_size"] > 1 and plan["replicated_ops"]),
                    replicated_op_names=set(plan["replicated_ops"]), model_config=config,
                ))
    rows = deduplicate_tp1_rows(
        rows, tp1_key_fields=("num_tokens", "model_arch", "measurement_type"),
    )

    assert len(rows) == 12
    replicated = [row for row in rows if row["num_tensor_parallel_workers"] == 1]
    assert len(replicated) == 6
    assert {(row["measurement_type"], row["num_tokens"]) for row in replicated} == {
        (method, tokens)
        for method in ("CUDA_EVENT", "RECORD_FUNCTION") for tokens in (1, 16, 64)
    }
    for row in replicated:
        assert row["time_stats"]["input_layernorm"]["mean"] == 1.0
        assert row["time_stats"]["step5_ffn_merge"]["mean"] == 8.0
        assert row["padded_n_embd"] == 4096
        assert all(metadata["layer_kind"] is None for metadata in row["typed_operator_contracts"].values())
        assert json.loads(row["linear_backends"]) == {
            "step5_ffn_merge": ["native_add_fp32_backend"],
        }
    serialized = main._serialize_linear_op_output(pd.DataFrame(rows))
    for record in serialized.to_dict("records"):
        contracts = parse_typed_operator_contracts(record["typed_operator_contracts"])
        assert set(contracts) == set(record["time_stats"])
        assert ("step5_ffn_merge" in contracts) == (record["num_tensor_parallel_workers"] == 1)
