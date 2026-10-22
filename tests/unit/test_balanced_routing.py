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
