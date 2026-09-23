"""Checks that specifically protect the four module splits.

The fidelity matrix proves that supported configurations produce the same
numbers. It cannot see the failure modes a move-heavy refactor actually has,
because those break at import or construction time, before any simulation runs:
an annotation that no longer resolves in its defining module, a name that used
to be re-exported, a mixin whose method the owning class no longer reaches, a
pickled estimator whose module path moved.

Each test here was run as a task-local command during the split. Committing them
is the difference between a check that happened once and a check that keeps
happening.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import sys
import typing
from pathlib import Path

import pytest

import frontier.config
from frontier.config.config import SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from tests.frontier_sources import iter_frontier_sources


# --- annotation resolution in each defining module --------------------------


def _config_modules() -> list[str]:
    package = frontier.config
    return sorted(
        f"frontier.config.{info.name}"
        for info in pkgutil.iter_modules(package.__path__)
        if not info.ispkg
    )


#: ``ClusterConfig.cc_backend_config`` is annotated ``BaseCCBackendConfig``,
#: which the module does not import at runtime. This predates the split: the
#: same lookup fails on ``1f694f7``, where the class still lived in the single
#: ``config.py``. It is pinned rather than fixed so that a *new* unresolvable
#: annotation, which is what the split could plausibly introduce, fails here.
KNOWN_UNRESOLVED_CONFIG_ANNOTATIONS = {"frontier.config.cluster_config.ClusterConfig"}


def test_no_new_config_dataclass_loses_its_annotations() -> None:
    """``flat_dataclass`` resolves string annotations in the defining module.

    ``from __future__ import annotations`` makes every annotation a string, and
    the CLI generator resolves each one against the namespace of the module
    that defines the dataclass. Splitting one module into twelve therefore
    means each new module must import the names its own annotations mention,
    not merely the names its code calls. A missing import is invisible until
    something asks for the type.
    """

    unresolved: dict[str, str] = {}
    for module_name in _config_modules():
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if not dataclasses.is_dataclass(value) or value.__module__ != module_name:
                continue
            try:
                typing.get_type_hints(value)
            except Exception as error:  # noqa: BLE001 - the message is the finding
                unresolved[f"{module_name}.{name}"] = f"{type(error).__name__}: {error}"

    new = {key: reason for key, reason in unresolved.items()
           if key not in KNOWN_UNRESOLVED_CONFIG_ANNOTATIONS}
    assert not new, "annotations that no longer resolve:\n" + "\n".join(
        f"  {key}: {reason}" for key, reason in sorted(new.items())
    )
    fixed = KNOWN_UNRESOLVED_CONFIG_ANNOTATIONS - set(unresolved)
    assert not fixed, (
        f"these now resolve, so drop them from the pinned set: {sorted(fixed)}"
    )


def test_the_flat_cli_can_still_be_generated() -> None:
    """The end-to-end consequence of the test above, through the real generator."""

    flat = create_flat_dataclass(SimulationConfig)
    assert len(dataclasses.fields(flat)) > 700


# --- public re-exports ------------------------------------------------------


def _names_imported_from_frontier_config(repo_root: Path) -> dict[str, set[str]]:
    """Collect what other modules import, keyed by the module they import from.

    ``frontier.config`` and ``frontier.config.config`` are different entry
    points with different contents: ``PrecisionType`` and the quantization
    helpers live on the package, never on ``config.py``. Each import site is
    therefore checked against the module it actually names.
    """

    import ast

    wanted: dict[str, set[str]] = {"frontier.config": set(), "frontier.config.config": set()}
    for path in sorted(iter_frontier_sources(repo_root)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in wanted:
                wanted[node.module].update(
                    alias.name for alias in node.names if alias.name != "*"
                )
    return wanted


def test_public_config_names_resolve_from_the_entry_point_that_is_imported() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    holders = {
        "frontier.config": frontier.config,
        "frontier.config.config": importlib.import_module("frontier.config.config"),
    }

    missing: list[str] = []
    for module_name, names in _names_imported_from_frontier_config(repo_root).items():
        for name in sorted(names):
            if not hasattr(holders[module_name], name):
                missing.append(f"{module_name}.{name}")
    assert not missing, "names other modules import but cannot reach: " + ", ".join(missing)


def test_the_split_modules_stay_reachable_through_the_package() -> None:
    """Everything ``config.py`` re-exports must also come off the package.

    Callers use both spellings, and the split moved the definitions out from
    under both of them at once.
    """

    config_module = importlib.import_module("frontier.config.config")
    exported = [
        name for name, value in vars(config_module).items()
        if not name.startswith("_") and dataclasses.is_dataclass(value)
    ]
    assert len(exported) > 20, "the re-export surface collapsed"
    unreachable = [name for name in exported if not hasattr(frontier.config, name)]
    assert not unreachable, (
        "config.py exports these but the package does not: " + ", ".join(sorted(unreachable))
    )


# --- mixin method resolution ------------------------------------------------


def test_vllm_v1_scheduler_reaches_every_extracted_mixin() -> None:
    from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
        BaseReplicaScheduler,
    )
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler,
    )

    mro = VLLMv1EngineReplicaScheduler.__mro__
    names = [cls.__name__ for cls in mro]
    expected_order = [
        "VLLMv1EngineReplicaScheduler",
        "IterationSchedulingPolicy",
        "KvBlockAllocation",
        "PrefixCacheLedger",
        "TargetEmbeddedMtpWaitPolicy",
        "DecodeAttentionCohort",
        "DisaggregatedRoleScheduling",
    ]
    assert names[: len(expected_order)] == expected_order
    # The base must stay behind every mixin, or a mixin override would lose to
    # the base implementation it was extracted from.
    assert names.index("BaseReplicaScheduler") > names.index("DisaggregatedRoleScheduling")
    assert issubclass(VLLMv1EngineReplicaScheduler, BaseReplicaScheduler)


def test_sglang_still_reaches_the_extracted_decision_log_helper() -> None:
    """The donor branch deleted this method while this caller kept calling it."""

    from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
        SGLangStyleReplicaScheduler,
    )

    method = getattr(SGLangStyleReplicaScheduler, "_get_num_waiting_reqs_for_decision_log")
    assert method.__module__ == (
        "frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy"
    ), "the helper moved; check that both decision-log callers still resolve it"


@pytest.mark.parametrize(
    ("class_path", "expected_head"),
    (
        (
            "frontier.execution_time_predictor.shared_prediction_model_manager."
            "ExecutionTimePredictionModelManager",
            ["ExecutionTimePredictionModelManager", "PredictionFamilyTrainers",
             "ProfilingDataFrameLoaders", "PredictionModelRegistry",
             "LayerContractResolution"],
        ),
        (
            "frontier.execution_time_predictor.sklearn_moe_execution_time_predictor."
            "SklearnMoEExecutionTimePredictor",
            ["SklearnMoEExecutionTimePredictor", "MoeOperatorTimes", "MoeRoutingWorkload",
             "MoeDatasetTraining", "MoeMtpReplay", "SklearnExecutionTimePredictor"],
        ),
        (
            "frontier.config.cluster_config.ClusterConfig",
            ["ClusterConfig", "ClusterRoleConfigBuilder", "ClusterTopologySummary"],
        ),
    ),
)
def test_split_classes_keep_their_mixin_order(class_path: str, expected_head: list[str]) -> None:
    module_name, _, class_name = class_path.rpartition(".")
    owner = getattr(importlib.import_module(module_name), class_name)
    assert [cls.__name__ for cls in owner.__mro__][: len(expected_head)] == expected_head


def test_cluster_config_fields_come_only_from_the_owning_class() -> None:
    """The two extracted mixins hold behavior, not state."""

    from frontier.config.cluster_config import ClusterConfig
    from frontier.config.cluster_role_config import ClusterRoleConfigBuilder
    from frontier.config.cluster_topology_summary import ClusterTopologySummary

    for mixin in (ClusterRoleConfigBuilder, ClusterTopologySummary):
        assert not dataclasses.is_dataclass(mixin), (
            f"{mixin.__name__} became a dataclass, so it now contributes fields "
            "to ClusterConfig and changes the generated CLI"
        )
    assert len(dataclasses.fields(ClusterConfig)) == 181


# --- estimator cache loading ------------------------------------------------


def test_a_cached_estimator_loads_into_a_fresh_registry(tmp_path: Path) -> None:
    """Cache *names* matching does not prove a cached object still loads.

    A pickle records the module path of the class it holds. Moving code is
    exactly what invalidates that, and a cold run that retrains from scratch
    would never notice, because it writes the file it then reads.
    """

    from sklearn.tree import DecisionTreeRegressor

    from frontier.execution_time_predictor.prediction_model_registry import (
        PredictionModelRegistry,
    )

    class _Holder(PredictionModelRegistry):
        def __init__(self, cache_dir: str) -> None:  # noqa: D107 - test scaffold
            self._cache_dir = cache_dir

    estimator = DecisionTreeRegressor(random_state=0).fit([[0.0], [1.0]], [0.0, 2.0])

    writer = _Holder(str(tmp_path))
    writer._store_model_in_cache("attn_prefill", "abc123", estimator)

    reader = _Holder(str(tmp_path))
    loaded = reader._load_model_from_cache("attn_prefill", "abc123")

    assert loaded is not None, "a fresh registry could not read what another wrote"
    assert loaded.predict([[1.0]]).tolist() == estimator.predict([[1.0]]).tolist()
    assert reader._load_model_from_cache("attn_prefill", "not_this_hash") is None


def test_every_split_module_imports_in_a_fresh_interpreter_order() -> None:
    """Import each new module first, with nothing else loaded from its package.

    The config split's one real defect was a name available only under
    ``TYPE_CHECKING`` while a method constructed it at runtime. Importing a
    module in isolation is what surfaces that class of mistake.
    """

    split_modules = [
        *_config_modules(),
        "frontier.scheduler.replica_scheduler.vllm_v1_decision_log",
        "frontier.scheduler.replica_scheduler.vllm_v1_decode_attn_cohort",
        "frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy",
        "frontier.scheduler.replica_scheduler.vllm_v1_kv_allocation",
        "frontier.scheduler.replica_scheduler.vllm_v1_mtp_wait",
        "frontier.scheduler.replica_scheduler.vllm_v1_prefix_cache",
        "frontier.scheduler.replica_scheduler.vllm_v1_role_schedules",
        "frontier.execution_time_predictor.layer_contract_resolution",
        "frontier.execution_time_predictor.moe_dataset_training",
        "frontier.execution_time_predictor.moe_mtp_replay",
        "frontier.execution_time_predictor.moe_operator_times",
        "frontier.execution_time_predictor.moe_predictor_helpers",
        "frontier.execution_time_predictor.moe_routing_workload",
        "frontier.execution_time_predictor.prediction_family_trainers",
        "frontier.execution_time_predictor.prediction_model_identity",
        "frontier.execution_time_predictor.prediction_model_registry",
        "frontier.execution_time_predictor.profiling_dataframe_loaders",
    ]
    failures: list[str] = []
    for module_name in split_modules:
        try:
            importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - the message is the finding
            failures.append(f"{module_name}: {type(error).__name__}: {error}")
    assert not failures, "modules that do not import:\n" + "\n".join(failures)


def test_runtime_only_names_are_not_hidden_behind_type_checking() -> None:
    """The I1 defect, pinned: a runtime constructor needs a runtime import."""

    from frontier.config.cluster_role_config import ClusterRoleConfigBuilder

    source = Path(
        sys.modules[ClusterRoleConfigBuilder.__module__].__file__
    ).read_text(encoding="utf-8")
    builder = "get_cluster_configs_for_disaggregation"
    assert builder in source
    body = source[source.index(f"def {builder}"):]
    assert "from frontier.config.cluster_config import ClusterConfig" in body, (
        "this method constructs ClusterConfig at runtime, so the name has to be "
        "imported outside TYPE_CHECKING"
    )
