"""Validate routed sorting ownership with CPU tensors at the AITER boundary."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.experimental.sglang import moe


@pytest.mark.parametrize("counts", [(1, 0), (31, 1), (32, 0), (33, 31), (64, 33, 0)])
def test_sorting_validates_every_block_of_each_expert(monkeypatch, counts):
    torch = pytest.importorskip("torch")
    tensor, full = torch.tensor, torch.full

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return tensor(*args, **kwargs)

    def cpu_full(*args, **kwargs):
        kwargs.pop("device", None)
        return full(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch, "full", cpu_full)

    def sorted_routes(ids, weights, num_experts, hidden_size, dtype, block_size, **kwargs):
        size = len(ids)
        packed_ids, packed_weights, owners = [], [], []
        for expert in range(num_experts):
            routes = [(token, slot) for token, row in enumerate(ids.tolist())
                      for slot, owner in enumerate(row) if owner == expert]
            if not routes:
                continue
            blocks = (len(routes) + block_size - 1) // block_size
            owners.extend([expert] * blocks)
            packed_ids.extend(token | (slot << 24) for token, slot in routes)
            packed_weights.extend(float(weights[token, slot]) for token, slot in routes)
            padding = blocks * block_size - len(routes)
            packed_ids.extend([size] * padding)
            packed_weights.extend([0.0] * padding)
        return (
            tensor(packed_ids, dtype=torch.int32),
            tensor(packed_weights, dtype=torch.float32),
            tensor(owners, dtype=torch.int32),
            tensor([len(packed_ids), size], dtype=torch.int32),
            torch.zeros(size, hidden_size, dtype=dtype),
        )

    aiter = ModuleType("aiter")
    fused_moe = ModuleType("aiter.fused_moe")
    fused_moe.moe_sorting = sorted_routes
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fused_moe)
    model = SimpleNamespace(
        is_moe=True, routed_mlp_hidden_dim=256, embedding_dim=64,
        num_experts=len(counts), num_experts_per_tok=1,
    )

    call, expected, mutable, backend, spec, workload = moe.make_moe_sorting_primitive(
        sum(counts), counts, model, SimpleNamespace(world_size=1)
    )

    torch.testing.assert_close(call(), expected, rtol=0, atol=0)
    assert workload["sorted_token_blocks"] == sum((count + 31) // 32 for count in counts)
    assert mutable == ()
    assert backend == "aiter.moe_sorting/opus"
