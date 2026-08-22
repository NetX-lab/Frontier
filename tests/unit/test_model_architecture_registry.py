from __future__ import annotations

import ast
import copy
import json
import logging
import pickle
from contextlib import contextmanager
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from frontier.config.model_config import BaseModelConfig, MoEModelConfig, ModelArch
from frontier.config.utils import dataclass_to_dict
from frontier.model_architectures import (
    ExpertParallelCollective,
    LinearAttentionImplementation,
    LinearAttentionProfile,
    MODEL_ARCHITECTURE_REGISTRY,
    ModelArchitectureProfile,
    ModelArchitectureRegistry,
    ResidualAddPolicy,
    StructuralRequirement,
    get_model_architecture_profile,
)
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ActivationType, ClusterType, NormType


class _LogRecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_model_architecture_warnings() -> Iterator[list[logging.LogRecord]]:
    target_logger = logging.getLogger("frontier.model_architectures")
    collector = _LogRecordCollector()
    previous_level = target_logger.level
    target_logger.addHandler(collector)
    target_logger.setLevel(logging.WARNING)
    try:
        yield collector.records
    finally:
        target_logger.removeHandler(collector)
        target_logger.setLevel(previous_level)


def _generic_fallback_records(
    records: list[logging.LogRecord],
) -> list[logging.LogRecord]:
    return [
        record
        for record in records
        if record.name == "frontier.model_architectures"
        and record.getMessage().startswith(
            "Model architecture profile fallback selected generic"
        )
    ]


class _RawProfileResolutionCallCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        function_aliases: set[str],
        module_aliases: set[str],
        registry_aliases: set[str],
    ) -> None:
        self._function_aliases = function_aliases
        self._module_aliases = module_aliases
        self._registry_aliases = registry_aliases
        self._scope: list[str] = []
        self.call_counts: dict[tuple[str, str], int] = {}

    def _visit_scoped(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        call_kind: str | None = None
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self._function_aliases
        ):
            call_kind = "helper"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_model_architecture_profile"
            and ast.unparse(node.func.value) in self._module_aliases
        ):
            call_kind = "helper"
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            registry_expression = node.func.value
            if (
                isinstance(registry_expression, ast.Name)
                and registry_expression.id in self._registry_aliases
            ):
                call_kind = "registry.resolve"
            elif (
                isinstance(registry_expression, ast.Attribute)
                and registry_expression.attr == "MODEL_ARCHITECTURE_REGISTRY"
                and ast.unparse(registry_expression.value) in self._module_aliases
            ):
                call_kind = "registry.resolve"

        if call_kind is not None:
            scope = ".".join(self._scope) or "<module>"
            key = (scope, call_kind)
            self.call_counts[key] = self.call_counts.get(key, 0) + 1

        self.generic_visit(node)


def _profiling_config(**overrides):
    values = dict(
        model_type="unit_custom_model",
        model_arch="generic",
        model_architecture_profile=None,
        embedding_dim=128,
        mlp_hidden_dim=256,
        num_q_heads=8,
        num_kv_heads=4,
        no_tensor_parallel=False,
        is_moe=True,
        post_attn_norm=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    cfg = SimpleNamespace(**values)
    cfg.supports_share_expert = lambda: bool(get_model_architecture_profile(cfg).supports_share_expert(cfg))
    return cfg


def _runtime_model_config(**overrides) -> BaseModelConfig:
    values = dict(
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=4,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=32000,
        model_type="unit_custom_model",
        model_arch=ModelArch.GENERIC,
        is_moe=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    return BaseModelConfig(**values)


def _real_profiling_model_config(**overrides) -> ProfilingModelConfig:
    values = dict(
        name="unit-model",
        num_layers=2,
        num_q_heads=8,
        num_kv_heads=4,
        embedding_dim=128,
        mlp_hidden_dim=256,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation="silu",
        norm="rms_norm",
        post_attn_norm=True,
        vocab_size=32000,
        model_type="unit_custom_model",
        model_arch=ModelArch.GENERIC,
        is_moe=True,
        share_expert_dim=64,
    )
    values.update(overrides)
    return ProfilingModelConfig(**values)


def _step3_mfa_overrides() -> dict[str, int | bool]:
    return {
        "num_kv_heads": 1,
        "use_mfa": True,
        "share_q_dim": 16,
        "head_dim": 16,
    }


def test_explicit_profile_id_reuses_step3_contract_for_new_model_name() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )

    profile = get_model_architecture_profile(cfg)

    assert profile.profile_id == "step3_text"
    assert profile.linear_attention.sharded_impl is LinearAttentionImplementation.STEP3_TEXT
    assert profile.linear_attention.replicated_ops == (
        "attn_pre_proj_qkv",
        "attn_pre_proj_q_norm",
    )
    assert profile.expert_parallel_collective is ExpertParallelCollective.ALLTOALL
    assert profile.uses_expert_parallel_alltoall(ClusterType.MONOLITHIC, expected_ep_size=2)


def test_explicit_step3_profile_drives_profiling_plan_without_model_type_branch() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
    )

    plan = build_profiling_plan(
        cfg,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
    )

    assert "attn_pre_proj_qkv" in plan["replicated_ops"]
    assert "attn_pre_proj_q_norm" in plan["replicated_ops"]
    assert "attn_pre_proj_wq" in plan["enabled_ops"]
    assert "share_expert_up_proj" in plan["enabled_ops"]


def test_explicit_step2_profile_drives_profiling_plan_without_model_type_branch() -> None:
    cfg = _profiling_config(
        model_type="unit_new_step2_like",
        model_architecture_profile="step2_mini",
    )

    plan = build_profiling_plan(
        cfg,
        tp_size=2,
        attn_tp=[2],
        ffn_tp=[2],
        is_moe=True,
    )

    assert "attn_inter_norm" in plan["enabled_ops"]
    assert "attn_wq_proj" in plan["enabled_ops"]
    assert "share_expert_up_proj" in plan["enabled_ops"]


def test_local_registry_can_plugin_custom_profile_without_global_model_branch() -> None:
    registry = ModelArchitectureRegistry()
    registry.register(ModelArchitectureProfile.generic())
    registry.register(
        ModelArchitectureProfile.step3_text(
            profile_id="unit_step3_plugin",
            match=lambda cfg: getattr(cfg, "model_type", None) == "unit_plugin_model",
        )
    )

    profile = registry.resolve(_profiling_config(model_type="unit_plugin_model"))

    assert profile.profile_id == "unit_step3_plugin"
    assert profile.linear_attention.sharded_ops == (
        "attn_pre_proj",
        "attn_rope",
        "attn_post_proj",
        "attn_pre_proj_wq",
    )


def test_model_architecture_profile_requires_explicit_ep_collective_policy() -> None:
    expert_parallel_collective = next(
        field
        for field in fields(ModelArchitectureProfile)
        if field.name == "expert_parallel_collective"
    )

    assert expert_parallel_collective.default is MISSING


@pytest.mark.parametrize(
    "profile",
    MODEL_ARCHITECTURE_REGISTRY.iter_profiles(),
    ids=lambda profile: profile.profile_id,
)
@pytest.mark.parametrize("expected_ep_size", (1, 2))
def test_canonical_profiles_use_alltoall_for_every_ep_size(
    profile: ModelArchitectureProfile,
    expected_ep_size: int,
) -> None:
    assert profile.expert_parallel_collective is ExpertParallelCollective.ALLTOALL
    assert profile.uses_expert_parallel_alltoall(
        ClusterType.MONOLITHIC,
        expected_ep_size=expected_ep_size,
    )


def test_registry_rejects_non_alltoall_collective_policy() -> None:
    registry = ModelArchitectureRegistry()
    invalid_profile = ModelArchitectureProfile(
        profile_id="unit_invalid_ep_collective",
        display_name="Invalid EP Collective",
        linear_attention=LinearAttentionProfile(
            sharded_impl=LinearAttentionImplementation.GENERIC,
            sharded_ops=(
                "attn_pre_proj",
                "attn_rope",
                "attn_post_proj",
            ),
        ),
        expert_parallel_collective=ExpertParallelCollective.ALLGATHER,
    )

    with pytest.raises(
        ValueError,
        match="must declare expert_parallel_collective=ALLTOALL",
    ):
        registry.register(invalid_profile)


def test_unknown_model_uses_generic_profile_with_warning(caplog) -> None:
    cfg = SimpleNamespace(
        model_type="unit_unknown_transformer",
        model_arch="generic",
        model_architecture_profile=None,
    )
    target_logger = logging.getLogger("frontier.model_architectures")

    target_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=target_logger.name):
            profile = get_model_architecture_profile(cfg)
    finally:
        target_logger.removeHandler(caplog.handler)

    assert profile.profile_id == "generic"
    assert "unit_unknown_transformer" in caplog.text
    assert "generic" in caplog.text


def test_runtime_model_config_resolves_implicit_profile_once() -> None:
    with _capture_model_architecture_warnings() as records:
        cfg = _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile=None,
        )
        assert len(_generic_fallback_records(records)) == 1

        profiles = [cfg.get_model_architecture_profile() for _ in range(5)]
        share_expert_results = [cfg.supports_share_expert() for _ in range(5)]

        assert len(_generic_fallback_records(records)) == 1

    generic_profile = MODEL_ARCHITECTURE_REGISTRY.get("generic")
    assert cfg.model_architecture_profile is None
    assert all(profile is generic_profile for profile in profiles)
    assert share_expert_results == [True] * 5


def test_ep_collective_resolver_reuses_runtime_resolved_identity() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    with _capture_model_architecture_warnings() as records:
        cfg = _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile=None,
        )
        assert len(_generic_fallback_records(records)) == 1

        collectives = [
            resolve_ep_collective_kind(
                cfg,
                ClusterType.MONOLITHIC,
                expected_ep_size=2,
            )
            for _ in range(5)
        ]

        assert len(_generic_fallback_records(records)) == 1

    assert collectives == [ExpertParallelCollective.ALLTOALL] * 5


def test_ep_collective_resolver_uses_runtime_snapshot_after_identity_mutation() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    cfg = _runtime_model_config(
        model_type="unit_unknown_transformer",
        model_architecture_profile=None,
    )
    generic_profile = MODEL_ARCHITECTURE_REGISTRY.get("generic")
    step3_profile = MODEL_ARCHITECTURE_REGISTRY.get("step3_text")

    assert cfg.get_model_architecture_profile() is generic_profile
    assert get_model_architecture_profile(cfg) is generic_profile

    cfg.model_type = "step3_text"
    collective_after_mutation = resolve_ep_collective_kind(
        cfg,
        ClusterType.MONOLITHIC,
        expected_ep_size=2,
    )
    reclassified = replace(cfg, **_step3_mfa_overrides())

    assert cfg.get_model_architecture_profile() is generic_profile
    assert get_model_architecture_profile(cfg) is step3_profile
    assert collective_after_mutation is ExpertParallelCollective.ALLTOALL
    assert reclassified.get_model_architecture_profile() is step3_profile
    assert get_model_architecture_profile(reclassified) is step3_profile


def test_ep_collective_resolver_requires_runtime_profile_accessor() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    with pytest.raises(
        TypeError,
        match=r"model_config\.get_model_architecture_profile\(\)",
    ):
        resolve_ep_collective_kind(
            _profiling_config(model_architecture_profile="generic"),
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )


def test_ep_collective_resolver_rejects_invalid_runtime_profile() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    model_config = SimpleNamespace(
        model_type="unit_invalid_runtime_profile",
        model_arch="generic",
        model_architecture_profile="generic",
        get_model_architecture_profile=lambda: object(),
    )

    with pytest.raises(
        TypeError,
        match=r"get_model_architecture_profile\(\) must return",
    ):
        resolve_ep_collective_kind(
            model_config,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )


def test_ep_collective_resolver_rejects_unsupported_cluster_role() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    model_config = _runtime_model_config(model_architecture_profile="generic")

    with pytest.raises(
        ValueError,
        match=r"profile generic does not support EP collectives for DECODE_ATTN",
    ):
        resolve_ep_collective_kind(
            model_config,
            ClusterType.DECODE_ATTN,
            expected_ep_size=2,
        )


def test_attention_trainer_reuses_runtime_resolved_identity() -> None:
    from frontier.training.attention_trainer import AttentionTrainer

    with _capture_model_architecture_warnings() as records:
        model_config = _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile=None,
        )
        trainer = AttentionTrainer.__new__(AttentionTrainer)
        trainer.model_config = model_config

        required_column_sets = [
            trainer._get_required_compute_dataset_columns() for _ in range(5)
        ]

        assert len(_generic_fallback_records(records)) == 1

    assert all(columns == required_column_sets[0] for columns in required_column_sets)


def test_raw_model_profile_resolution_callsites_are_allowlisted() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_call_counts = {
        # Construction binds the runtime snapshot from declarative config state.
        ("frontier/config/model_config.py", "BaseModelConfig.__post_init__", "helper"): 1,
        # Config-like prediction adapters may not own a BaseModelConfig snapshot.
        (
            "frontier/execution_time_predictor/shared_prediction_model_manager.py",
            "_resolve_model_architecture_profile",
            "helper",
        ): 1,
        (
            "frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py",
            "SklearnDisaggregationExecutionTimePredictor._resolve_model_architecture_profile_for_config",
            "helper",
        ): 1,
        # Metrics accepts both runtime and structural/profiling config-like objects.
        (
            "frontier/metrics/capability_context.py",
            "CapabilityContext.from_replica_config",
            "helper",
        ): 1,
        # This is the single implementation boundary for raw registry resolution.
        (
            "frontier/model_architectures.py",
            "get_model_architecture_profile",
            "registry.resolve",
        ): 1,
        # Operator binding accepts structural/profiling config-like objects.
        (
            "frontier/operators/binding.py",
            "_get_model_architecture_profile",
            "helper",
        ): 1,
        # Profiling configs intentionally resolve their current declarative state.
        (
            "frontier/profiling/common/model_config.py",
            "ModelConfig.get_model_architecture_profile",
            "helper",
        ): 1,
        (
            "frontier/profiling/linear_op/linear_op_impl.py",
            "build_linear_op_attention_module",
            "helper",
        ): 1,
        (
            "frontier/profiling/linear_op/profiling_plan.py",
            "build_profiling_plan",
            "helper",
        ): 1,
        (
            "frontier/profiling/utils/confirmation.py",
            "build_linear_op_config_sections",
            "helper",
        ): 1,
        # The MTP adapter wraps a profiling config rather than runtime state.
        (
            "frontier/spec_decode/mtp_runtime.py",
            "StructuralModelConfigAdapter.get_model_architecture_profile",
            "helper",
        ): 1,
    }
    observed_call_counts: dict[tuple[str, str, str], int] = {}

    for path in (repo_root / "frontier").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_aliases: set[str] = set()
        module_aliases: set[str] = set()
        registry_aliases: set[str] = (
            {"MODEL_ARCHITECTURE_REGISTRY"}
            if path == repo_root / "frontier" / "model_architectures.py"
            else set()
        )

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "frontier.model_architectures"
            ):
                function_aliases.update(
                    imported.asname or imported.name
                    for imported in node.names
                    if imported.name == "get_model_architecture_profile"
                )
                registry_aliases.update(
                    imported.asname or imported.name
                    for imported in node.names
                    if imported.name == "MODEL_ARCHITECTURE_REGISTRY"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "frontier":
                module_aliases.update(
                    imported.asname or imported.name
                    for imported in node.names
                    if imported.name == "model_architectures"
                )
            elif isinstance(node, ast.Import):
                module_aliases.update(
                    imported.asname or imported.name
                    for imported in node.names
                    if imported.name == "frontier.model_architectures"
                )

        collector = _RawProfileResolutionCallCollector(
            function_aliases=function_aliases,
            module_aliases=module_aliases,
            registry_aliases=registry_aliases,
        )
        collector.visit(tree)
        relative_path = str(path.relative_to(repo_root))
        for (scope, call_kind), call_count in collector.call_counts.items():
            observed_call_counts[(relative_path, scope, call_kind)] = call_count

    assert observed_call_counts == expected_call_counts


def test_runtime_model_config_replace_re_resolves_implicit_identity() -> None:
    with _capture_model_architecture_warnings() as records:
        cfg = _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile=None,
        )
        reclassified = replace(
            cfg,
            model_type="step3_text",
            **_step3_mfa_overrides(),
        )
        profile = reclassified.get_model_architecture_profile()

    assert len(_generic_fallback_records(records)) == 1
    assert profile is MODEL_ARCHITECTURE_REGISTRY.get("step3_text")
    assert cfg.model_architecture_profile is None
    assert reclassified.model_architecture_profile is None


@pytest.mark.parametrize("model_name", ["Qwen3-235B-A22B", "llama3.3-70b"])
def test_benchmark_model_metadata_declares_generic_profile(model_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (repo_root / "data" / "config" / "models" / f"{model_name}.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["model_architecture_profile"] == "generic"


@pytest.mark.parametrize("model_name", ["Qwen3-235B-A22B", "llama3.3-70b"])
def test_known_generic_model_config_avoids_fallback_warning(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)

    with _capture_model_architecture_warnings() as records:
        cfg = BaseModelConfig.create_from_name(model_name)
        profile = cfg.get_model_architecture_profile()
        supports_share_expert = cfg.supports_share_expert()

    assert _generic_fallback_records(records) == []
    assert cfg.model_architecture_profile == "generic"
    assert profile is MODEL_ARCHITECTURE_REGISTRY.get("generic")
    assert supports_share_expert is False


def test_runtime_model_profile_cache_preserves_copy_protocols() -> None:
    with _capture_model_architecture_warnings() as records:
        cfg = _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile=None,
        )
        round_tripped = pickle.loads(pickle.dumps(cfg))
        copied = copy.deepcopy(cfg)
        replaced = replace(cfg)

        configs = (cfg, round_tripped, copied, replaced)
        profiles = [config.get_model_architecture_profile() for config in configs]

    assert len(_generic_fallback_records(records)) == 2
    assert all(config.model_architecture_profile is None for config in configs)
    assert all(
        profile is MODEL_ARCHITECTURE_REGISTRY.get("generic")
        for profile in profiles
    )


def test_runtime_model_profile_cache_stays_out_of_dataclass_schema() -> None:
    cfg = _runtime_model_config(model_architecture_profile="generic")
    internal_name = "_resolved_model_architecture_profile_id"

    assert internal_name not in {field.name for field in fields(BaseModelConfig)}
    assert internal_name not in asdict(cfg)
    assert internal_name not in dataclass_to_dict(cfg)
    assert cfg.get_model_architecture_profile() is MODEL_ARCHITECTURE_REGISTRY.get(
        "generic"
    )


def test_profiling_model_config_from_known_model_ignores_runtime_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)

    cfg = ProfilingModelConfig.from_model_name("Qwen3-235B-A22B")

    assert cfg.model_architecture_profile == "generic"
    assert cfg.get_model_architecture_profile() is MODEL_ARCHITECTURE_REGISTRY.get(
        "generic"
    )


def test_runtime_config_explicit_unknown_profile_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown model architecture profile"):
        _runtime_model_config(
            model_type="unit_unknown_transformer",
            model_architecture_profile="typo_step3_text",
        )


def test_explicit_unknown_profile_fails_fast_without_generic_downgrade() -> None:
    cfg = SimpleNamespace(
        model_type="unit_unknown_transformer",
        model_arch="generic",
        model_architecture_profile="typo_step3_text",
    )

    with pytest.raises(ValueError, match="Unknown model architecture profile"):
        get_model_architecture_profile(cfg)


def test_profiles_no_longer_expose_model_identity_booleans() -> None:
    for profile in (
        ModelArchitectureProfile.generic(),
        ModelArchitectureProfile.step2_mini(),
        ModelArchitectureProfile.step3_text(),
    ):
        assert not hasattr(profile, "step2_mini_compatible")
        assert not hasattr(profile, "step3_text_compatible")


def test_config_objects_no_longer_expose_model_identity_accessors() -> None:
    runtime_config = _runtime_model_config(model_architecture_profile="generic")
    profiling_config = _real_profiling_model_config(model_architecture_profile="generic")

    for config in (runtime_config, profiling_config):
        assert not hasattr(config, "is_step2_mini")
        assert not hasattr(config, "is_step3_text")


def test_linear_op_wrapper_metadata_does_not_emit_identity_field() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "frontier/profiling/linear_op/linear_op_wrapper.py"
    ).read_text(encoding="utf-8")

    assert '"model_architecture_profile"' in source
    assert '"is_step2_mini"' not in source


def test_runtime_config_explicit_profile_uses_profile_api_without_identity_accessors() -> None:
    cfg = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        **_step3_mfa_overrides(),
    )

    assert cfg.get_model_architecture_profile().profile_id == "step3_text"
    assert not hasattr(cfg, "is_step3_text")
    assert not hasattr(cfg, "is_step2_mini")
    assert cfg.supports_share_expert()
    assert cfg.get_attention_family().family_id == "dense_attention"


def test_profiling_config_explicit_profile_uses_profile_api_without_identity_accessors() -> None:
    cfg = _real_profiling_model_config(
        model_type="unit_new_step2_like",
        model_arch=ModelArch.GENERIC,
        model_architecture_profile="step2_mini",
    )

    assert cfg.get_model_architecture_profile().profile_id == "step2_mini"
    assert not hasattr(cfg, "is_step2_mini")
    assert not hasattr(cfg, "is_step3_text")
    assert cfg.supports_share_expert()


def test_ep_collective_resolver_uses_profile_not_step3_model_type() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    step3_alias_cfg = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        **_step3_mfa_overrides(),
    )
    generic_named_step3_cfg = _runtime_model_config(
        model_type="step3_text",
        model_architecture_profile="generic",
    )

    assert (
        resolve_ep_collective_kind(
            step3_alias_cfg,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )
        is ExpertParallelCollective.ALLTOALL
    )
    assert (
        resolve_ep_collective_kind(
            generic_named_step3_cfg,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )
        is ExpertParallelCollective.ALLTOALL
    )


@pytest.mark.parametrize(
    ("expected_ep_size", "expected_collective"),
    [
        (2, ExpertParallelCollective.ALLTOALL),
        (1, ExpertParallelCollective.ALLTOALL),
    ],
)
def test_step_moe_noquant_real_config_resolves_decode_ffn_collective(
    expected_ep_size: int,
    expected_collective: ExpertParallelCollective,
) -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    model_config = BaseModelConfig.create_from_name("step-moe-noquant")

    assert model_config.get_name() == "step-moe-noquant"
    assert model_config.model_type == "step3_text"
    assert model_config.embedding_dim == 7168
    assert model_config.get_model_architecture_profile().profile_id == "step3_text"
    assert (
        resolve_ep_collective_kind(
            model_config,
            ClusterType.DECODE_FFN,
            expected_ep_size=expected_ep_size,
        )
        is expected_collective
    )


def test_ep_collective_resolver_fails_fast_without_model_config() -> None:
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        resolve_ep_collective_kind,
    )

    with pytest.raises(
        ValueError,
        match="EP collective resolution requires replica_config.model_config",
    ):
        resolve_ep_collective_kind(
            None,
            ClusterType.MONOLITHIC,
            expected_ep_size=2,
        )


def test_phase2_consumers_do_not_directly_branch_on_step2_step3_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    consumer_paths = (
        repo_root / "frontier/profiling/linear_op/linear_op_impl.py",
        repo_root / "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py",
        repo_root / "frontier/metrics/metrics_store.py",
    )
    forbidden_snippets = (
        'config.model_type == "step3_text"',
        'config.model_type == "step2_mini"',
        'model_config.model_type == "step3_text"',
        "model_config.is_step3_text()",
    )

    violations = []
    for path in consumer_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append(f"{path.relative_to(repo_root)}: {snippet}")

    assert violations == []





def test_step3_profile_declares_residual_add_policy_capability() -> None:
    assert ModelArchitectureProfile.generic().residual_add_policy is ResidualAddPolicy.STANDARD
    assert (
        ModelArchitectureProfile.step3_text().residual_add_policy
        is ResidualAddPolicy.FFN_RESIDUAL_ONLY
    )


def test_predictor_metadata_validates_architecture_profile_id() -> None:
    import pandas as pd

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    predictor._model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
        **_step3_mfa_overrides(),
    )
    df = pd.DataFrame(
        {
            "profiling_precision": ["fp16"],
            "model_arch": ["generic"],
            "model_architecture_profile": ["step3_text"],
            "quant_signature": [predictor._model_config.get_quant_signature()],
            "measurement_type": ["cuda_event"],
        }
    )

    metadata = predictor._get_profiling_metadata(df, "unit.csv")

    assert metadata.model_arch == "generic"
    assert metadata.model_architecture_profile == "step3_text"


def test_predictor_metadata_rejects_profile_mismatch() -> None:
    import pandas as pd
    import pytest

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    predictor._model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
        **_step3_mfa_overrides(),
    )
    df = pd.DataFrame(
        {
            "profiling_precision": ["fp16"],
            "model_arch": ["generic"],
            "model_architecture_profile": ["generic"],
            "quant_signature": [predictor._model_config.get_quant_signature()],
            "measurement_type": ["cuda_event"],
        }
    )

    with pytest.raises(ValueError, match="model_architecture_profile mismatch"):
        predictor._get_profiling_metadata(df, "unit.csv")


def test_predictor_rejects_invalid_architecture_profile_contract() -> None:
    from typing import cast

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = object.__new__(_ConcretePredictor)
    # This test intentionally injects a malformed structural config.
    predictor._model_config = cast(
        BaseModelConfig,
        SimpleNamespace(get_model_architecture_profile=lambda: object()),
    )

    try:
        predictor._get_model_architecture_profile()
    except TypeError as exc:
        assert "must return ModelArchitectureProfile" in str(exc)
    else:
        raise AssertionError("Expected invalid architecture profile contract to fail")


def test_moe_predictor_has_no_visibility_scaling_hook() -> None:
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
        SklearnMoEExecutionTimePredictor,
    )

    assert not hasattr(
        SklearnMoEExecutionTimePredictor,
        "_apply_share_expert_tp_allreduce_overlap",
    )


def test_step3_profile_requires_moe_runtime_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires? is_moe=True"):
        _runtime_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=False,
            share_expert_dim=64,
        )


def test_step3_profile_requires_moe_profiling_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires? is_moe=True"):
        _real_profiling_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=False,
            share_expert_dim=64,
        )


def test_step3_profile_requires_mfa_runtime_attention_contract() -> None:
    with pytest.raises(ValueError, match="step3_text.*use_mfa=True"):
        _runtime_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=True,
            share_expert_dim=64,
            use_mfa=False,
        )


def test_step3_profile_accepts_mfa_dense_runtime_attention_family() -> None:
    cfg = _runtime_model_config(
        model_type="unit_step3_like_mfa",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
        **_step3_mfa_overrides(),
    )

    assert cfg.get_model_architecture_profile().profile_id == "step3_text"
    assert cfg.use_mfa is True
    assert cfg.get_attention_family().family_id == "dense_attention"
    assert cfg.get_attention_family().supported_variants == ("gqa", "mha", "mqa")


def test_step3_profile_requires_mfa_profiling_attention_contract() -> None:
    with pytest.raises(ValueError, match="step3_text.*use_mfa=True"):
        _real_profiling_model_config(
            model_type="unit_invalid_step3_like",
            model_architecture_profile="step3_text",
            is_moe=True,
            share_expert_dim=64,
            use_mfa=False,
        )


def test_step3_profiling_model_config_to_dict_preserves_mfa_contract() -> None:
    cfg = _real_profiling_model_config(
        model_type="unit_step3_like_mfa",
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
        **_step3_mfa_overrides(),
    )

    serialized = cfg.to_dict()
    restored = ProfilingModelConfig(**serialized)

    assert serialized["use_mfa"] is True
    assert restored.use_mfa is True
    assert restored.get_model_architecture_profile().profile_id == "step3_text"


def test_mla_attention_shape_profile_requires_latent_mla_attention_family() -> None:
    profile = ModelArchitectureProfile(
        profile_id="unit_mla_profile",
        display_name="Unit MLA Profile",
        linear_attention=LinearAttentionProfile(
            sharded_impl=LinearAttentionImplementation.GENERIC,
            sharded_ops=(
                "attn_pre_proj",
                "attn_rope",
                "attn_post_proj",
            ),
        ),
        expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
        attention_shape_log_kind="mla",
    )

    with pytest.raises(ValueError, match="unit_mla_profile.*latent_mla_attention"):
        profile.validate_structural_requirements(
            SimpleNamespace(
                model_type="unit_dense",
                model_arch="generic",
                use_mla=False,
                use_mfa=False,
                num_q_heads=8,
                num_kv_heads=8,
            )
        )

    profile.validate_structural_requirements(
        SimpleNamespace(
            model_type="unit_mla",
            model_arch="generic",
            use_mla=True,
            use_mfa=False,
            num_q_heads=8,
            num_kv_heads=1,
            kv_lora_rank=4,
            qk_nope_head_dim=3,
            qk_rope_head_dim=2,
            qk_head_dim=5,
            v_head_dim=4,
        )
    )


def test_structural_requirement_wraps_predicate_value_error_with_profile_context() -> None:
    profile = ModelArchitectureProfile(
        profile_id="unit_wrapped_error_profile",
        display_name="Unit Wrapped Error Profile",
        linear_attention=LinearAttentionProfile(
            sharded_impl=LinearAttentionImplementation.GENERIC,
            sharded_ops=(
                "attn_pre_proj",
                "attn_rope",
                "attn_post_proj",
            ),
        ),
        expert_parallel_collective=ExpertParallelCollective.ALLTOALL,
        structural_requirements=(
            StructuralRequirement(
                name="requires_unit_contract",
                predicate=lambda config: (_ for _ in ()).throw(
                    ValueError("low-level binding failed")
                ),
                message=lambda profile, config: (
                    f"{profile.profile_id} requires unit structural contract"
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="unit_wrapped_error_profile requires unit structural contract",
    ):
        profile.validate_structural_requirements(SimpleNamespace(model_type="unit_invalid"))


def test_moe_model_config_runs_base_structural_validation_for_step3_profile() -> None:
    with pytest.raises(ValueError, match="step3_text.*use_mfa=True"):
        MoEModelConfig(
            num_layers=2,
            num_q_heads=8,
            num_kv_heads=1,
            embedding_dim=128,
            mlp_hidden_dim=256,
            max_position_embeddings=4096,
            use_gated_mlp=True,
            use_bias=False,
            use_qkv_bias=False,
            activation=ActivationType.SILU,
            norm=NormType.RMS_NORM,
            post_attn_norm=True,
            vocab_size=32000,
            model_type="step3_text",
            model_architecture_profile="step3_text",
            is_moe=True,
            num_experts=8,
            num_experts_per_tok=2,
            share_expert_dim=64,
            use_mfa=False,
        )


def test_existing_step3_json_preserves_mfa_dense_attention_contract() -> None:
    from frontier.attention.model_binding import bind_attention_family

    cfg = BaseModelConfig.create_from_name("step-moe-noquant-small")
    binding = bind_attention_family(cfg)

    assert cfg.get_model_architecture_profile().profile_id == "step3_text"
    assert cfg.use_mfa is True
    assert cfg.use_mla is False
    assert binding.family_id == "dense_attention"
    assert binding.variant_id == "mqa"


def test_mtp_structural_adapter_uses_explicit_architecture_profile() -> None:
    from frontier.spec_decode.mtp_runtime import StructuralModelConfigAdapter

    profiling_config = _real_profiling_model_config(
        model_type="unit_new_step3_like",
        model_arch=ModelArch.GENERIC,
        model_architecture_profile="step3_text",
        is_moe=True,
        share_expert_dim=64,
        **_step3_mfa_overrides(),
    )
    adapter = StructuralModelConfigAdapter(profiling_config)

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert not hasattr(adapter, "is_step3_text")
    assert not hasattr(adapter, "is_step2_mini")
    assert adapter.supports_share_expert()


def test_mtp_json_fallback_preserves_explicit_architecture_profile(
    monkeypatch,
    tmp_path,
) -> None:
    import json

    from frontier.spec_decode.mtp_runtime import (
        _load_structural_model_config_from_json,
    )

    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "unit-json-step3-like.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "hidden_size": 128,
                "intermediate_size": 256,
                "max_position_embeddings": 4096,
                "vocab_size": 32000,
                "hidden_act": "silu",
                "model_type": "unit_new_step3_like",
                "model_arch": "generic",
                "model_architecture_profile": "step3_text",
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "share_expert_dim": 64,
                "share_q_dim": 16,
                "head_dim": 16,
                "use_mfa": True,
                "torch_dtype": "float16",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    adapter = _load_structural_model_config_from_json("unit-json-step3-like")

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert not hasattr(adapter, "is_step3_text")
    assert not hasattr(adapter, "is_step2_mini")
    assert adapter.supports_share_expert()
    assert adapter.use_mfa is True
    assert adapter.get_attention_family().family_id == "dense_attention"


def test_mtp_structural_loader_does_not_mask_internal_profiling_errors(
    monkeypatch,
) -> None:
    import pytest

    from frontier.spec_decode import mtp_runtime

    def raise_internal_error(model_name: str):  # noqa: ARG001
        raise RuntimeError("profiling registry exploded")

    monkeypatch.setattr(
        mtp_runtime.ProfilingModelConfig,
        "from_model_name",
        staticmethod(raise_internal_error),
    )

    with pytest.raises(RuntimeError, match="profiling registry exploded"):
        mtp_runtime.load_mtp_structural_model_config("unit-json-step3-like")


def test_mtp_structural_loader_preserves_json_fallback_for_value_error(
    monkeypatch,
    tmp_path,
) -> None:
    import json

    from frontier.spec_decode import mtp_runtime

    def raise_model_lookup_error(model_name: str):  # noqa: ARG001
        raise ValueError("profiling model config unavailable")

    config_dir = tmp_path / "data" / "config" / "models"
    config_dir.mkdir(parents=True)
    (config_dir / "unit-json-step3-like.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "hidden_size": 128,
                "intermediate_size": 256,
                "max_position_embeddings": 4096,
                "vocab_size": 32000,
                "hidden_act": "silu",
                "model_type": "unit_new_step3_like",
                "model_arch": "generic",
                "model_architecture_profile": "step3_text",
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "share_expert_dim": 64,
                "share_q_dim": 16,
                "head_dim": 16,
                "use_mfa": True,
                "torch_dtype": "float16",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mtp_runtime.ProfilingModelConfig,
        "from_model_name",
        staticmethod(raise_model_lookup_error),
    )

    adapter = mtp_runtime.load_mtp_structural_model_config("unit-json-step3-like")

    assert adapter.get_model_architecture_profile().profile_id == "step3_text"
    assert adapter.supports_share_expert()
    assert adapter.use_mfa is True
    assert adapter.get_attention_family().family_id == "dense_attention"


def test_param_counter_share_expert_uses_profile_for_new_model_name() -> None:
    from types import SimpleNamespace
    from typing import cast

    from frontier.config import ReplicaConfig
    from frontier.utils.param_counter import ParamCounter

    model_config = _runtime_model_config(
        model_type="unit_new_step3_like",
        model_architecture_profile="step3_text",
        embedding_dim=128,
        share_expert_dim=64,
        use_gated_mlp=True,
        is_moe=True,
        **_step3_mfa_overrides(),
    )
    replica_config = cast(
        ReplicaConfig,
        SimpleNamespace(
            model_config=model_config,
            attn_tensor_parallel_size=1,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=1,
            num_pipeline_stages=1,
        ),
    )

    counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)

    assert counter._get_share_expert_params_per_layer(tensor_parallel_size=2) == 12288


def test_phase2_predictor_consumers_do_not_use_step2_step3_identity_wrappers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    consumer_paths = (
        repo_root / "frontier/execution_time_predictor/sklearn_execution_time_predictor.py",
        repo_root / "frontier/execution_time_predictor/shared_prediction_model_manager.py",
        repo_root / "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py",
        repo_root / "frontier/utils/param_counter.py",
        repo_root / "frontier/profiling/utils/confirmation.py",
    )
    forbidden_snippets = (
        "is_step2_mini()",
        "is_step3_text()",
        "model_type == \"step3_text\"",
        "model_type not in {\"step2_mini\", \"step3_text\"}",
        "model_arch == \"step2_mini\"",
        "step3_text_compatible",
        "_log_step3_attention_shape",
        "[STEP3_SHAPE]",
    )

    violations = []
    for path in consumer_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append(f"{path.relative_to(repo_root)}: {snippet}")

    assert violations == []
