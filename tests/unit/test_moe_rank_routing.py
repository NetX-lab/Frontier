"""Rank selection and gate/routed row plumbing with CPU tensors only."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from frontier.moe_ep_workload import materialize_balanced_ep_workload
from frontier.profiling.moe.balanced_routing import balanced_topk
from frontier.profiling.moe.moe_wrapper import MoEWrapper
from frontier.profiling.moe.moe_impl import get_routing_runtime_metadata
from frontier.profiling.moe.moe_profile_identity import balanced_routing_columns
from test_moe_rank_plan import EXPECTED


@pytest.mark.parametrize("rank,ep", [(-1, 16), (16, 16), (0, 0)])
def test_constructor_rejects_invalid_ep_rank_before_runtime_setup(rank, ep):
    with pytest.raises(ValueError, match="EP rank"):
        MoEWrapper(SimpleNamespace(is_moe=True), 1, ep, "cuda_event", rank, "/unused")


def cpu_wrapper(ep, rank):
    wrapper = MoEWrapper.__new__(MoEWrapper)
    wrapper.rank = rank
    wrapper.expert_parallel_size = ep
    wrapper.num_experts = 256
    wrapper.num_experts_per_device = 256 // ep
    wrapper.router_topk = 6
    wrapper.routing_runtime_path = "balanced_topk"
    wrapper.routing_runtime_metadata = {**get_routing_runtime_metadata("balanced_topk"),
                                        **balanced_routing_columns()}
    return wrapper


@pytest.mark.parametrize("row", EXPECTED)
def test_each_planned_class_uses_its_rank_and_balanced_routing(monkeypatch, row):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ep, gathered, rank_class, local, rank = row
    wrapper = cpu_wrapper(ep, rank)
    routing = wrapper._prepare_routing_inputs(gathered, "balanced", seed=0)
    workload = materialize_balanced_ep_workload(
        routing_token_count=gathered, router_topk=6, total_expert_num=256,
        moe_expert_parallel_size=ep,
    )
    assert routing["expert_token_counts"] == list(workload.per_ep_per_expert_tokens[rank].values())
    assert sum(routing["expert_token_counts"]) == rank_class[0]
    expert_map = routing["expert_map"]
    assert (expert_map >= 0).nonzero().flatten().tolist() == list(
        range(rank * wrapper.num_experts_per_device, (rank + 1) * wrapper.num_experts_per_device))
    weights, ids = balanced_topk(torch.empty(gathered, 1), torch.empty(gathered, 256), 6,
                                 token_offset=0)
    torch.testing.assert_close(routing["topk_ids"], ids)
    torch.testing.assert_close(routing["topk_weights"], weights)
    assert routing["topk_weights"].dtype == torch.float32
    assert routing["topk_ids"].shape[0] == gathered


@pytest.mark.parametrize("local", [None, 1, 2, 768])
def test_profile_gate_uses_local_rows_and_reuses_the_gathered_sample(monkeypatch, local):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = cpu_wrapper(16, 3)
    wrapper.num_tensor_parallel_workers = 1
    wrapper.hidden_dim = 1
    wrapper.expert_hidden_dim = 1
    wrapper.use_gated = True
    wrapper.gating_runtime_context_metadata = {}
    wrapper._resolve_routed_layer_contract = lambda **kwargs: SimpleNamespace(
        typed_metadata_identity=lambda: {}, effective_ffn_width=1)
    calls = []
    samples = []
    def gate(rows):
        hidden = torch.empty(rows, 1)
        calls.append(hidden.shape[0])
        return {"time_stats": {"moe_gating_linear": {}, "moe_gating_routing_topk": {}},
                "num_tokens": rows}
    def routed(rows, **kwargs):
        sample = kwargs["routing_inputs"]
        assert sample["topk_ids"].shape[0] == rows == 6152
        samples.append(sample)
        return {"time_stats": {"moe_shuffling": {}, "moe_grouped_gemm": {}}}
    wrapper.profile_gating = gate
    wrapper.profile_shuffling = routed
    wrapper.profile_grouped_gemm = routed
    result = wrapper.profile(6152, "balanced", seed=0, gating_num_tokens=local)
    assert calls == [6152 if local is None else local]
    assert samples[0] is samples[1]
    assert result["num_tokens"] == 6152
    assert result["gating_num_tokens"] == calls[0]
    assert result["ep_rank"] == 3
    assert [result[name] for name in ("local_routed_rows", "local_experts_with_extra_row",
                                     "base_rows_per_expert")] == [2304, 0, 144]
    assert set(result["time_stats"]) == {"moe_gating_linear", "moe_gating_routing_topk",
                                         "moe_shuffling", "moe_grouped_gemm"}
    assert result["routing_strategy_sha256"] == balanced_routing_columns()["routing_strategy_sha256"]


def test_generated_counts_must_match_the_algebra(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = cpu_wrapper(16, 0)
    wrapper._compute_local_expert_token_counts = lambda ids: [0] * 16
    with pytest.raises(ValueError, match="disagree with EP algebra"):
        wrapper._prepare_routing_inputs(16, "balanced")


def test_balanced_signature_rejects_another_load_distribution():
    with pytest.raises(ValueError, match="balanced load_distribution"):
        cpu_wrapper(16, 0)._prepare_routing_inputs(16, "uniform")


def test_balanced_strategy_identity_is_pinned_and_mismatch_fails(monkeypatch):
    from frontier.profiling.moe import moe_profile_identity, moe_wrapper
    path = Path(__file__).resolve().parents[2] / "frontier/profiling/moe/balanced_routing.py"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert balanced_routing_columns() == {"routing_strategy_sha256": expected}
    monkeypatch.setattr(moe_profile_identity, "BALANCED_ROUTING_SHA256", "wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        balanced_routing_columns()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        config = SimpleNamespace(is_moe=True, name="deepseek-ai/DeepSeek-V4-Flash",
            get_model_architecture_profile=lambda: SimpleNamespace(moe_gating_routing_runtime_path="balanced_topk"))
        monkeypatch.setattr(moe_wrapper, "configure_quantization_manager_for_model_name", lambda name: None)
        MoEWrapper(config, 1, 16, "cuda_event", 0, "/unused")


def test_multiprocessing_worker_passes_task_rank_and_gate_rows(monkeypatch):
    from frontier.profiling.moe import main as moe_main
    seen = {}
    class Wrapper:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def profile(self, **kwargs):
            seen["profile"] = kwargs
            return {"result": True}
    monkeypatch.setattr(moe_main, "_get_moe_wrapper_class", lambda: Wrapper)
    monkeypatch.setattr(moe_main, "ModelConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(moe_main, "_ensure_torch_available", lambda: SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda device: seen.update(device=device))))
    monkeypatch.setattr(moe_main, "_CUDA_INITIALIZED", False)
    monkeypatch.setattr(moe_main, "_CUDA_GPU_ID", None)
    result = moe_main._worker_profile_task((0, 0, {}, dict(
        num_tensor_parallel_workers=1, expert_parallel_size=16, rank=3,
        profile_method="cuda_event", output_dir="/unused", use_vllm_kernel=True,
    ), dict(num_tokens=6152, gating_num_tokens=1, load_distribution="balanced", seed=0)))
    assert result == {"result": True}
    assert seen["rank"] == 3
    assert seen["profile"]["num_tokens"] == 6152
    assert seen["profile"]["gating_num_tokens"] == 1


@pytest.mark.parametrize("ray_mode", [False, True])
def test_runtime_scheduler_uses_the_same_rank_tasks_as_the_plan(monkeypatch, ray_mode):
    import sys
    from frontier.profiling.moe import main as moe_main
    from test_moe_rank_plan import SPEC
    monkeypatch.setattr(sys, "argv", ["moe.main", *SPEC["common_args"],
        "--expert_parallel_sizes", "16", "--dp_lane_local_rows", "768,1"])
    args, _ = moe_main.parse_args()
    args.disable_ray = not ray_mode
    model = moe_main.ModelConfig.from_model_name(SPEC["model"])
    expected = moe_main._lane_rank_tasks(args, model)
    seen = []
    class Wrapper:
        def __init__(self, *positional, **kwargs):
            self.rank = positional[4] if positional else kwargs["rank"]
        def profile(self, **kwargs):
            seen.append((self.rank, kwargs["num_tokens"], kwargs["gating_num_tokens"]))
            return dict(num_tokens=kwargs["num_tokens"], expert_parallel_size=16,
                        num_tensor_parallel_workers=1, time_stats={name: {"median": 1.0}
                        for name in ("moe_gating_linear", "moe_gating_routing_topk",
                                     "moe_shuffling", "moe_grouped_gemm")})
    class ActorFactory:
        def options(self, **kwargs):
            return self
        def remote(self, *args):
            wrapper = Wrapper(*args)
            return SimpleNamespace(profile=SimpleNamespace(remote=wrapper.profile))
    monkeypatch.setattr(moe_main, "_get_moe_wrapper_class", lambda: Wrapper)
    monkeypatch.setattr(moe_main, "_get_available_gpus", lambda num: [0])
    monkeypatch.setattr(moe_main, "set_process_visible_device", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(moe_main, "RAY_AVAILABLE", True)
    monkeypatch.setattr(moe_main, "ray", SimpleNamespace(
        remote=lambda **kwargs: lambda cls: ActorFactory(), get=lambda promises: promises))
    frame = moe_main.profile_model(args, SPEC["model"], [6152],
                                   SimpleNamespace(update=lambda count: None), True)
    assert len(frame) == len(expected)
    assert sorted(seen) == sorted((task["ep_rank"], task["num_tokens"], task["gating_num_tokens"])
                                 for task in expected)
