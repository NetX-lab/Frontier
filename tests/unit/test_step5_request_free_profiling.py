"""Native producer seams for the one-slot request-free DP workload."""

from contextlib import contextmanager, nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5.inputs import Step5ProfileInput, read_profile_inputs
from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper


@pytest.fixture(scope="module")
def config():
    return ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")


def test_request_free_native_forward_receives_no_attention_metadata_or_slot_mapping(monkeypatch):
    import torch

    calls = []
    module = ModuleType("vllm.forward_context")

    @contextmanager
    def set_forward_context(metadata, native, **kwargs):
        calls.append((metadata, kwargs))
        yield

    module.set_forward_context = set_forward_context
    module.BatchDescriptor = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    native_config = ModuleType("vllm.config")
    native_config.CUDAGraphMode = {"NONE": "NONE"}
    monkeypatch.setitem(sys.modules, native_config.__name__, native_config)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.native = torch, object()
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: zeros(*a, **{**k, "device": "cpu"}))
    wrapper._set_current = lambda _: nullcontext()
    wrapper.builder = SimpleNamespace(build=lambda *a, **k: pytest.fail("Request-free must not build metadata"))
    wrapper.layer = SimpleNamespace(attn=SimpleNamespace(layer_name="layer0.attn"))
    cell = Step5ProfileInput(0, (1,), (0,), "decode", has_attention_metadata=False)
    prepared = wrapper._metadata(cell)
    assert prepared[0].tolist() == [0]
    assert prepared[1:] == (None, None, "none")
    hidden = torch.ones((1, 4))
    output = wrapper._forward(hidden, prepared, layer=lambda positions, values: values + 1)
    assert torch.equal(output, hidden + 1)
    assert calls[0][0] is None
    assert calls[0][1]["slot_mapping"] is None


@pytest.mark.parametrize("layer", [0, 3])
def test_request_free_input_matches_existing_consumer_workload_and_scopes(config, layer):
    from frontier.attention.step5 import describe_step5_layer

    cell = Step5ProfileInput(layer, (1,), (0,), "decode", has_attention_metadata=False)
    workload = cell.workload(config, block_size=16, max_model_len=7168)
    assert workload.has_attention_metadata is False
    assert workload.block_table_columns == workload.score_regions == 0
    metadata = describe_step5_layer(config, layer, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    scopes = metadata.attention_scopes(workload)
    assert scopes == metadata.request_free_attention_scopes()
    work = {item for _, items in scopes for item in items}
    assert {"q_projection", "qk_norm", "output_zero_fill", "output_projection"} <= work
    assert not {"raw_kv_write", "summary_update", "selector_scoring", "sparse_gqa"} & work


@pytest.mark.parametrize("values", [
    {"graph_mode": "full"}, {"context_lens": (8,)}, {"query_lens": (1, 1), "context_lens": (0, 0)},
    {"phase": "prefill", "query_lens": (16,)}, {"has_attention_metadata": 0},
    {"block_table_columns": 448},
])
def test_request_free_rejects_graph_history_multiple_slots_and_ambiguous_identity(values):
    with pytest.raises(ValueError):
        Step5ProfileInput(**{**dict(global_layer_id=0, query_lens=(1,), context_lens=(0,),
                                   phase="decode", has_attention_metadata=False), **values})


def test_explicit_request_free_inputs_do_not_inherit_decode_full_graph(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps([
        dict(global_layer_id=0, query_lens=[1], context_lens=[0], phase="decode", has_attention_metadata=False),
        dict(global_layer_id=0, query_lens=[1], context_lens=[6144], phase="decode"),
    ]))
    cells = read_profile_inputs(path, cudagraph_capture_sizes=(1, 2))
    assert [cell.graph_mode for cell in cells] == ["none", "full"]


@pytest.mark.parametrize("layer", [0, 3])
def test_request_free_rows_are_accepted_by_unchanged_consumer_not_regular_decode(tmp_path, config, layer):
    from frontier.attention.step5 import describe_step5_layer
    from frontier.execution_time_predictor.step5 import load_step5_attention_rows, step5_row_features
    from frontier.profiling.step5.rows import OPERATOR_NAMES, build_row, rows_to_dataframe

    cell = Step5ProfileInput(layer, (1,), (0,), "decode", has_attention_metadata=False)
    metadata = describe_step5_layer(config, layer, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    active = [name for name, _ in metadata.request_free_attention_scopes()]
    stats = {name: dict(min=1. if name in active else 0., max=1. if name in active else 0.,
                       mean=1. if name in active else 0., median=1. if name in active else 0., std=0., count=2)
             for name in OPERATOR_NAMES}
    stats["step5_layer_e2e"] = dict(min=float(len(active)), max=float(len(active)),
        mean=float(len(active)), median=float(len(active)), std=0., count=2, kernel_count=len(active))
    row = build_row(config, cell, tp=8, block_size=16, max_model_len=7168,
        runtime_signature="cpu_producer_seam_not_native_measurement", device="h800", time_stats=stats,
        measurement_type="KERNEL_ONLY", evidence={"attention_backend": "native_cpu_boundary",
            "active_operators": json.dumps(active), "timing_basis": "correlated_kernel_plus_memcpy_duration_sum"})
    path = tmp_path / "idle.csv"
    frame = rows_to_dataframe([row], measurement_type="KERNEL_ONLY")
    frame.to_csv(path, index=False)
    replica = SimpleNamespace(model_config=config, device="h800", attn_tensor_parallel_size=8)
    scheduler = SimpleNamespace(block_size=16, max_model_len=7168)
    loaded = load_step5_attention_rows(path, replica, scheduler, "KERNEL_ONLY")
    assert len(loaded) == 1 and loaded.iloc[0].has_attention_metadata == 0
    assert loaded.iloc[0]["time_stats.step5_layer_e2e.kernel_count"] == len(active)
    assert loaded.iloc[0]["time_stats.step5_layer_e2e.mean"] > 0
    regular = build_row(config, Step5ProfileInput(layer, (1,), (0,), "decode"), tp=8,
        block_size=16, max_model_len=7168, runtime_signature="cpu_fixture", device="h800", time_stats=stats,
        evidence={"attention_backend": "native_cpu_boundary"})
    assert step5_row_features(row) != step5_row_features(regular)


def test_native_metadata_can_preserve_full_serving_table_extent(monkeypatch):
    import torch

    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *a, **k: arange(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    backend = ModuleType("vllm.v1.attention.backend")
    backend.CommonAttentionMetadata = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.block_size, wrapper.max_model_len = torch, 16, 7168
    wrapper.native, wrapper._set_current = object(), lambda _: nullcontext()
    wrapper.builder = SimpleNamespace(build=lambda _, common, **kw: common)
    cell = Step5ProfileInput(3, (1, 1), (6144, 6656), "decode", block_table_columns=448)
    _, slots, metadata, _ = wrapper._metadata(cell)
    assert metadata.block_table_tensor.shape == (2, 448)
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [6145, 6657]
    assert slots.tolist() == [6160, 13840]
    one = Step5ProfileInput(3, (1,), (6656,), "decode", block_table_columns=448)
    _, offset_slots, offset_metadata, _ = wrapper._metadata(one, request_offset=1)
    assert offset_slots.tolist() == [13840]
    assert torch.equal(offset_metadata.block_table_tensor[0], metadata.block_table_tensor[1])


def test_variable_decode_prefix_and_full_reference_respect_native_step_budget(config, monkeypatch):
    import torch
    from frontier.attention.step5 import describe_step5_layer

    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **k: randn(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cell = Step5ProfileInput(0, (1, 1), (6144, 6656), "decode", block_table_columns=448)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.config = torch, config
    wrapper.native = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=8192))
    wrapper._set_current = lambda _: nullcontext()
    wrapper.layer_id, wrapper.graph_mode, wrapper.max_batch_size = 0, "none", 128
    wrapper.block_size, wrapper.max_model_len, wrapper.profile_index, wrapper.seed = 16, 7168, 0, 0
    wrapper.layer = SimpleNamespace(use_dsa_backend=False)
    wrapper.state, wrapper.empty_state = [torch.zeros(1)], (torch.zeros(1),)
    wrapper.description = describe_step5_layer(config, 0, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16)
    active = dict(wrapper.description.attention_scopes(cell.workload(config, block_size=16, max_model_len=7168)))
    wrapper.scopes = SimpleNamespace(observe=lambda _: nullcontext(), times=lambda: active)
    wrapper._bindings = lambda: []
    wrapper._metadata = lambda current, request_offset=0: (current, request_offset)
    calls = []

    def native(values, prepared):
        current, offset = prepared
        assert len(values) == current.num_tokens <= 8192
        calls.append((current.num_tokens, offset))
        return values

    wrapper._forward = native

    def timing_boundary(*args, **kwargs):
        raise RuntimeError("CPU test reached the timing boundary")

    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.capture_native_replay", timing_boundary)
    with pytest.raises(RuntimeError, match="reached the timing boundary"):
        wrapper.profile(cell, warmup_iterations=1, profile_iterations=2, max_time_cv=.05)
    assert calls[:2] == [(6144, 0), (6656, 1)]
    assert (6145, 0) in calls and (6657, 1) in calls


def test_request_free_profile_rejects_any_cache_mutation(config, monkeypatch):
    import torch

    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **k: randn(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cache = torch.zeros(1)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.config, wrapper.native = torch, config, object()
    wrapper.layer_id, wrapper.graph_mode, wrapper.max_batch_size = 0, "none", 128
    wrapper.block_size, wrapper.max_model_len, wrapper.profile_index, wrapper.seed = 16, 7168, 0, 0
    wrapper.layer = SimpleNamespace(use_dsa_backend=False)
    wrapper._set_current = lambda _: nullcontext()
    wrapper.state, wrapper.empty_state = [cache], (cache.clone(),)
    wrapper._metadata = lambda _: (torch.zeros(1), None, None, "none")

    def invalid_native(hidden, prepared):
        cache.add_(1)
        return hidden

    wrapper._forward = invalid_native
    cell = Step5ProfileInput(0, (1,), (0,), "decode", has_attention_metadata=False)
    with pytest.raises(AssertionError):
        wrapper.profile(cell, warmup_iterations=1, profile_iterations=2, max_time_cv=.05)


def test_explicit_eager_reference_does_not_relabel_graph_dispatch(tmp_path):
    path = tmp_path / "inputs.json"
    values = dict(global_layer_id=3, query_lens=[16], context_lens=[6128], phase="prefill")
    path.write_text(json.dumps([{**values, "graph_mode": "none"}, values]))
    cells = read_profile_inputs(path, cudagraph_capture_sizes=(1, 2, 4, 8, 16))
    assert [cell.graph_mode for cell in cells] == ["none", "piecewise"]


def test_p0_packages_are_independent_h800_role_measurements(tmp_path, config):
    from pathlib import Path
    from frontier.profiling.step5.job import describe_p0_profiles

    for role, rows, gpu_rows in (("PREFILL", 4, 4), ("DECODE", 20, 4)):
        manifest = describe_p0_profiles(tmp_path / role, "p0_" + role.lower(), role=role)
        assert (manifest["gpu_tag"], manifest["charged_group"], manifest["gpus"], manifest["preemptible"]) == (
            "H800", "codesign", 1, False)
        assert manifest["measurement_role"] == role
        assert manifest["expected_attention_event_rows"] == rows
        assert manifest["expected_attention_gpu_active_rows"] == gpu_rows
        assert len(manifest["producer_commands"]) == 5
        assert "--expert_parallel_sizes 16" in manifest["worker_command"]
        assert manifest["measurement_plan"]["serving_topology_per_role"]["attention_dp"] == 2
        cells = read_profile_inputs(Path(manifest["attention_arguments"][
            manifest["attention_arguments"].index("--input-file") + 1]),
            cudagraph_capture_sizes=manifest["measurement_plan"]["cudagraph_capture_sizes"])
        assert sum(not c.has_attention_metadata for c in cells) == 2
        for cell in cells:
            workload = cell.workload(config, block_size=16, max_model_len=7168)
            assert workload.block_table_columns == (448 if cell.has_attention_metadata else 0)
        assert not manifest["admitted"] and manifest["blocks"]


@pytest.mark.parametrize("flag,value", [("--max-num-seqs", "1"), ("--max-num-batched-tokens", "1")])
def test_cli_capacity_smaller_than_grid_is_rejected(tmp_path, flag, value):
    from frontier.profiling.step5.main import main

    with pytest.raises(ValueError, match="must cover"):
        main(["--model-path", str(tmp_path), "--max-model-len", "7168", "--output-dir", str(tmp_path),
              "--decode-batches", "2", "--prefill-tokens", "16", flag, value, "--dry-run"])
