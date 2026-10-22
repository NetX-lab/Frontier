"""CPU correctness checks against the real native registry and Frontier counts."""

import ast
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from frontier.moe_ep_workload import (
    generate_moe_routing_ratios,
    materialize_expert_token_counts,
)
from frontier.profiling.moe.balanced_routing import balanced_topk, register_balanced_topk

ROOT = Path(__file__).resolve().parents[2]
MODEL = json.loads((ROOT / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json").read_text())


@pytest.fixture
def native(monkeypatch):
    source_root = os.environ.get("VLLM_BS_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("Set VLLM_BS_SOURCE_ROOT for the pinned native registry checks")
    source = Path(source_root) / "vllm"
    router = ast.parse((source / "model_executor/layers/fused_moe/router/routing_simulator_router.py").read_text())
    context = ast.parse((source / "forward_context.py").read_text())
    names = {"RoutingStrategy", "DistributionBasedRouting", "RoutingSimulator", "_compute_sp_num_tokens", "DPMetadata"}
    nodes = [node for tree in (router, context) for node in tree.body if getattr(node, "name", None) in names]
    namespace = dict(torch=torch, ABC=ABC, abstractmethod=abstractmethod, Any=Any,
                     dataclass=dataclass, contextmanager=contextmanager, ParallelConfig=Any,
                     logger=SimpleNamespace(warning_once=lambda *args: None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    ctx = SimpleNamespace(dp_metadata=None, cudagraph_runtime_mode="NONE", is_padding=None)
    modules = {
        "vllm.distributed": dict(get_ep_group=lambda: group),
        "vllm.forward_context": dict(get_forward_context=lambda: ctx),
        "vllm.model_executor.layers.fused_moe.router.routing_simulator_router": namespace,
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    simulator = namespace["RoutingSimulator"]
    assert simulator.get_available_strategies() == ["uniform_random", "normal_routing"]
    environment = dict(os.environ)
    register_balanced_topk()
    assert dict(os.environ) == environment
    assert isinstance(simulator._routing_strategies["balanced_topk"], namespace["RoutingStrategy"])
    return simulator, namespace["DPMetadata"], group, ctx


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("indices_type", [None, torch.int32, torch.int64])
@pytest.mark.parametrize("widths,sp,mode", [([6144], 1, "NONE"), ([9, 25], 8, "NONE"),
                                         ([0, 9], 8, "NONE"), ([8, 8], 8, "FULL"),
                                         ([24, 24], 8, "PIECEWISE")])
def test_native_chunks_match_frontier(native, dtype, indices_type, widths, sp, mode):
    simulator, metadata_type, group, ctx = native
    metadata = metadata_type(torch.tensor(widths, dtype=torch.int32))
    group.world_size = len(widths) * sp
    ctx.dp_metadata = metadata if group.world_size > 1 else None
    ctx.cudagraph_runtime_mode = mode
    experts, top_k = MODEL["n_routed_experts"], MODEL["num_experts_per_tok"]
    chunks = []
    with metadata.sp_local_sizes(sp) as sizes:
        for rank, rows in enumerate(sizes):
            group.rank_in_group = rank
            hidden = torch.zeros(rows, MODEL["hidden_size"], dtype=dtype)
            logits = torch.full((rows, experts), float("nan"), dtype=dtype)
            ctx.is_padding = torch.ones(rows, dtype=torch.bool)
            weights, ids = simulator.simulate_routing(hidden, logits, "balanced_topk", top_k, indices_type)
            assert ids.dtype == (torch.long if indices_type is None else indices_type)
            assert ids.shape == weights.shape == (rows, top_k)
            assert ids.device == weights.device == hidden.device
            assert weights.dtype == torch.float32
            torch.testing.assert_close(weights, torch.full((rows, top_k), 1 / top_k))
            assert torch.all(ids.sort(dim=1).values.diff(dim=1) > 0)
            chunks.append(ids.long())
    assert metadata.local_sizes is None
    ids = torch.cat(chunks)
    counts = torch.bincount(ids.flatten(), minlength=experts).tolist()
    expected = materialize_expert_token_counts(
        routing_ratios=generate_moe_routing_ratios(total_expert_num=experts,
                                                  distribution_type="balanced", seed=0, layer_id=0),
        total_routed_assignments=sum(sizes) * top_k, total_expert_num=experts,
    )
    assert counts == list(expected.values())
    assert sum(counts) == sum(sizes) * top_k
    _, unchunked = balanced_topk(torch.empty(sum(sizes), 1), torch.empty(sum(sizes), experts), top_k)
    torch.testing.assert_close(ids, unchunked)
    if widths == [6144]:
        assert counts == [144] * experts


@pytest.mark.parametrize("sizes,world,rows,error", [(None, 2, 2, "metadata"),
    (None, 1, 2, "local_sizes"), ([2], 2, 2, "local_sizes"), ([1, 3], 2, 2, "router rows")])
def test_missing_or_mismatched_native_metadata_rejects(native, sizes, world, rows, error):
    simulator, _, group, ctx = native
    group.world_size = world
    ctx.dp_metadata = None if world == 2 and sizes is None else SimpleNamespace(local_sizes=sizes)
    with pytest.raises(ValueError, match=error):
        simulator.simulate_routing(torch.empty(rows, 1), torch.empty(rows, 8), "balanced_topk", 3)


@pytest.mark.parametrize("rows,experts,top_k,offset", [(0, 8, 3, 0), (5, 5, 5, 7), (9, 7, 1, 2)])
def test_tensor_boundary_shapes(rows, experts, top_k, offset):
    _, ids = balanced_topk(torch.empty(rows, 1), torch.empty(rows, experts), top_k, token_offset=offset)
    assert ids.tolist() == [[((offset + row) * top_k + k) % experts for k in range(top_k)] for row in range(rows)]


@pytest.mark.parametrize("top_k,offset,error", [(0, 0, "top_k"), (9, 0, "top_k"),
                                              (3, -1, "token_offset")])
def test_invalid_tensor_routing_inputs_reject(top_k, offset, error):
    with pytest.raises(ValueError, match=error):
        balanced_topk(torch.empty(2, 1), torch.empty(2, 8), top_k, token_offset=offset)


def test_flat_discovery_has_no_runtime_imports_or_environment_changes():
    path = ROOT / "frontier/profiling/moe/balanced_routing.py"
    script = f"""import importlib.util, os, sys
before = dict(os.environ)
spec = importlib.util.spec_from_file_location('balanced_routing', {str(path)!r})
spec.loader.exec_module(importlib.util.module_from_spec(spec))
assert dict(os.environ) == before
assert not any(n == 'torch' or n.startswith(('torch.', 'vllm')) for n in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("distribution", ["random", "skewed", "zipf"])
def test_balanced_selector_rejects_incompatible_distribution(distribution):
    from frontier.config import ReplicaConfig

    with pytest.raises(ValueError, match="balanced_topk requires"):
        ReplicaConfig(moe_routing_distribution_type=distribution,
                      moe_gating_routing_runtime_path="balanced_topk")


def test_selector_validation_and_default_serialization():
    from frontier.config import ReplicaConfig
    from frontier.config.utils import dataclass_to_dict
    from frontier.moe_routing_runtime import resolve_moe_gating_routing_runtime_path as resolve

    assert "moe_gating_routing_runtime_path" not in dataclass_to_dict(ReplicaConfig())
    selected = ReplicaConfig(moe_gating_routing_runtime_path=" balanced_topk ")
    assert dataclass_to_dict(selected)["moe_gating_routing_runtime_path"] == "balanced_topk"
    assert resolve("balanced", SimpleNamespace(moe_gating_routing_runtime_path="standard_fused_topk"),
                   selected.moe_gating_routing_runtime_path) == "balanced_topk"
    with pytest.raises(ValueError, match="Unsupported routing_runtime_path"):
        ReplicaConfig(moe_gating_routing_runtime_path="uniform_random")


def test_producer_executes_shared_tensor_inside_routing_timer(monkeypatch):
    from frontier.profiling.moe import moe_impl

    scopes, calls = [], []
    @contextmanager
    def timer(name):
        scopes.append(name)
        yield
        scopes.pop()
    def route(*args):
        calls.append(scopes[-1])
        return balanced_topk(*args)
    monkeypatch.setattr(moe_impl, "CudaTimer", timer)
    monkeypatch.setattr(moe_impl, "balanced_topk", route)
    gate = moe_impl.MoEGatingNetwork(4, 8, 3, use_vllm_fused_topk=False,
                                   routing_runtime_path="balanced_topk")
    weights, ids = gate(torch.ones(5, 4))
    assert calls == ["moe_gating_routing_topk"]
    assert torch.bincount(ids.flatten(), minlength=8).tolist() == [2] * 7 + [1]
    assert weights.dtype == torch.float32 and ids.dtype == torch.long
    assert moe_impl.get_routing_runtime_metadata("balanced_topk") == {
        "routing_runtime_path": "balanced_topk", "routing_assignment_policy": "global_round_robin_balanced",
        "routing_weight_policy": "uniform_1_over_topk", "routing_uses_router_logits": False,
    }


def test_predictor_initialization_retains_selected_runtime(monkeypatch):
    from frontier.config import ReplicaConfig
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor

    class ParentBoundary(Exception):
        pass
    def stop(*args, **kwargs):
        raise ParentBoundary
    monkeypatch.setattr(SklearnExecutionTimePredictor, "__init__", stop)
    class Predictor(SklearnMoEExecutionTimePredictor):
        _get_estimator = lambda self: None
        _get_grid_search_params = lambda self: {}
    predictor = object.__new__(Predictor)
    replica = ReplicaConfig(moe_gating_routing_runtime_path="balanced_topk")
    with pytest.raises(ParentBoundary):
        predictor.__init__(None, replica, None, None)
    predictor._get_model_architecture_profile = replica.model_config.get_model_architecture_profile
    assert predictor._moe_gating_routing_runtime_path == "balanced_topk"
    assert predictor._get_requested_moe_gating_routing_runtime_path() == "balanced_topk"


@pytest.mark.parametrize("order", [("standard_fused_topk", "balanced_topk"), ("balanced_topk", "standard_fused_topk")])
def test_selected_producer_rows_train_and_reuse_distinct_cache(tmp_path, order):
    import pandas as pd
    from frontier.config import ClusterConfig, ReplicaConfig
    from frontier.config.execution_time_predictor_config import RandomForrestExecutionTimePredictorConfig
    from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
    from frontier.execution_time_predictor.prediction_model_identity import _serialize_selected_layer_cache_identity
    from frontier.moe_load_imbalance import MoELoadImbalanceInput
    from frontier.operators.families import MOE_FAMILY, SHARE_EXPERT_FAMILY, get_family_profiling_names
    from frontier.operators.typed_contracts import serialize_typed_operator_contracts
    from frontier.profiling.moe.moe_impl import get_routing_runtime_metadata
    from frontier.types import ClusterType, MeasurementType

    op = "moe_gating_routing_topk"
    configs = {role: ClusterConfig(cluster_type=role, replica_config=ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800", attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=1, moe_gating_routing_runtime_path=path))
        for role, path in zip((ClusterType.PREFILL, ClusterType.DECODE), order)}
    model = configs[ClusterType.PREFILL].replica_config.model_config
    profile = model.get_model_architecture_profile()
    metadata = {name: profile.resolve_layer_contract(model, operator_name=name,
        moe_tp_size=1, expert_parallel_size=1).typed_metadata_identity()
        for name in get_family_profiling_names(MOE_FAMILY)}
    dimensions = dict(num_experts=MODEL["n_routed_experts"], router_topk=MODEL["num_experts_per_tok"],
                      hidden_dim=MODEL["hidden_size"], expert_hidden_dim=MODEL["moe_intermediate_size"],
                      num_tensor_parallel_workers=1, expert_parallel_size=1)
    features = MoELoadImbalanceInput(8, model.num_experts, model.embedding_dim,
                                    model.routed_mlp_hidden_dim, model.num_experts_per_tok).to_features_dict()
    rows = pd.DataFrame([dict(dimensions, num_tokens=8, profiling_precision="FP16", measurement_type="CUDA_EVENT",
                             gating_runtime_context="standalone_legacy", gating_runtime_context_impl="none",
                             model_architecture_profile=profile.profile_id,
                             typed_operator_contracts=serialize_typed_operator_contracts(metadata),
                             **get_routing_runtime_metadata(path), **{key: value for key, value in features.items() if key not in dimensions},
                             **{f"time_stats.{name}.median": value if name == op else 1.0 for name in metadata})
                         for path, value in [("standard_fused_topk", 2.0), ("balanced_topk", 7.0)]])
    source, linear = tmp_path / "moe.csv", tmp_path / "linear.csv"
    rows.to_csv(source, index=False)
    shared_width = profile.resolve_layer_contract(model, operator_name="share_expert_up_proj", ffn_tp_size=1).width
    pd.DataFrame([dict(num_tokens=8, num_tensor_parallel_workers=1, n_expanded_embd=shared_width,
        profiling_precision="FP16", measurement_type="CUDA_EVENT",
        **{f"time_stats.{name}.median": 1.0 for name in ["post_attention_layernorm",
            *get_family_profiling_names(SHARE_EXPERT_FAMILY)]})]).to_csv(linear, index=False)
    manager = ExecutionTimePredictionModelManager({}, SimpleNamespace(cache_dir=str(tmp_path)))
    manager._cluster_configs, manager._all_dummy_mode = configs, False
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    config, signatures = RandomForrestExecutionTimePredictorConfig(num_estimators=[1], max_depth=[1], min_samples_split=[2]), set()
    def train(role):
        return manager._train_ffn_models_for_cluster(role, configs[role].replica_config, config,
            str(linear), str(source), True, signatures)[op]
    cold = {role: train(role) for role in configs}
    contracts = {role: manager._resolve_cluster_model_contract(role, op) for role in configs}
    assert all(contract.profile_id == "deepseek_v4" for contract in contracts.values())
    assert {json.loads(_serialize_selected_layer_cache_identity(contract))["routing_runtime_path"]
            for contract in contracts.values()} == set(order)
    assert len({model._frontier_model_hash for model in cold.values()}) == 2
    for role, path in zip(configs, order):
        selected = manager.get_model(op, layer_contract=contracts[role])
        assert selected.predict(pd.DataFrame({"num_tokens": [8]}))[0] == {"standard_fused_topk": 2.0, "balanced_topk": 7.0}[path]
        assert manager.get_model(op, layer_contract=contracts[role]) is cold[role]
        assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
        assert cold[role].predict(pd.DataFrame({"num_tokens": [8]}))[0] == {"standard_fused_topk": 2.0, "balanced_topk": 7.0}[path]
    manager._create_estimator_and_params = lambda *_: pytest.fail("Warm lookup must not fit")
    signatures.clear()
    warm = {role: train(role) for role in configs}
    for role in configs:
        assert warm[role]._frontier_model_hash == cold[role]._frontier_model_hash
        assert manager.get_model(op, layer_contract=contracts[role]) is warm[role]
        assert manager.get_models_for_cluster(role)["eager"][op] is warm[role]
        assert warm[role].predict(pd.DataFrame({"num_tokens": [8]}))[0] == cold[role].predict(pd.DataFrame({"num_tokens": [8]}))[0]
    for wrong in [rows.iloc[:1], rows.drop(columns="routing_runtime_path")]:
        wrong.to_csv(source, index=False)
        signatures.clear()
        with pytest.raises(ValueError, match="routing.*path"):
            train(next(role for role in configs if configs[role].replica_config.moe_gating_routing_runtime_path == "balanced_topk"))


@pytest.mark.parametrize("decode_roles", [dict(decode_cluster_num_replicas=1),
                                         dict(decode_attn_cluster_num_replicas=1, decode_ffn_cluster_num_replicas=1,
                                              decode_attn_af_pipeline_num_micro_batch=1, decode_ffn_af_pipeline_num_micro_batch=1)])
def test_v4_disaggregated_roles_and_copy_preserve_selector(decode_roles):
    from frontier.config import ClusterConfig, ReplicaConfig

    replica = ReplicaConfig(model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
                            moe_gating_routing_runtime_path="balanced_topk")
    cluster = ClusterConfig(replica_config=replica, prefill_cluster_num_replicas=1, **decode_roles)
    for role in cluster.get_cluster_configs_for_disaggregation().values():
        assert role.replica_config.moe_gating_routing_runtime_path == "balanced_topk"
    assert ClusterConfig(replica_config=replica)._create_replica_config_copy().moe_gating_routing_runtime_path == "balanced_topk"
