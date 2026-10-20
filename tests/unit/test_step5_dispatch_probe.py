"""CPU diagnostics tests; none of these results qualify GPU dispatch schemas."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.profiling.collectives import step5_dispatch_probe as probe


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model_path(tmp_path):
    directory = tmp_path / "model_metadata"
    directory.mkdir()
    source = ROOT / "data/config/models/stepfun-ai__Step-5-Preview.json"
    (directory / "config.json").write_text(source.read_text())
    return directory


def arguments(model_path, tmp_path, *extra):
    return probe.parse_args(["--model-path", str(model_path),
                             "--output-dir", str(tmp_path / "output"), *extra])


def test_plan_has_two_native_layers_and_no_cost_rows(model_path, tmp_path):
    plan = probe.probe_plan(arguments(model_path, tmp_path))
    assert plan["parallelism"] == {"tp": 4, "dp": 2, "ep": 8}
    assert [(c["layer_id"], c["precision"], c["design_row_bytes"]) for c in plan["cases"]] == [
        (3, "fp8", 4288), (88, "bf16", 8256)]
    assert plan["cases"][0]["expected_columns_dtypes"] == [
        (4096, "float8_e4m3fn"), (8, "float32"), (8, "int32"), (32, "float32")]
    assert plan["cost_data"] is False
    assert plan["timing_rows"] == 0
    assert plan["full_model_verified"] is False
    assert plan["model_runner_verified"] is False


def test_dry_run_command_needs_no_torch_or_vllm(model_path, tmp_path):
    command = [sys.executable, "-S", "-m", probe.__name__, "--model-path", str(model_path),
               "--output-dir", str(tmp_path / "unused"), "--tokens-per-dp-rank", "3", "1",
               "--disable-custom-all-reduce", "--dry-run"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "not_run"
    assert report["tokens_per_dp_rank"] == [3, 1]
    assert report["disable_custom_all_reduce"] is True
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("tokens", [(0, 1), (1, -1)])
def test_reject_zero_or_negative_tokens(model_path, tmp_path, tokens):
    with pytest.raises(SystemExit):
        arguments(model_path, tmp_path, "--tokens-per-dp-rank", *map(str, tokens))


def test_model_metadata_path_is_required():
    with pytest.raises(SystemExit):
        probe.parse_args(["--output-dir", "/unused"])


def test_metadata_mismatch_is_not_silently_replaced(model_path, tmp_path):
    path = model_path / "config.json"
    metadata = json.loads(path.read_text())
    metadata["text_config"]["model_type"] = "other"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Step-5 metadata"):
        probe.probe_plan(arguments(model_path, tmp_path))


def descriptor(rows, columns, dtype):
    byte_size = {"float8_e4m3fn": 1, "bfloat16": 2, "float32": 4, "int32": 4}[dtype]
    return {"shape": [rows, columns], "dtype": dtype, "stride": [columns, 1],
            "element_size": byte_size, "numel": rows * columns,
            "nbytes": rows * columns * byte_size, "device": "cpu"}


def recorded_calls(case, rows, sizes):
    columns = case["expected_columns_dtypes"]
    dispatch = {"op": "all_gatherv", "group_ranks": [0, 4], "sizes": list(sizes),
                "dim": 0, "completed": True,
                "inputs": [descriptor(rows, c, d) for c, d in columns],
                "outputs": [descriptor(sum(sizes), c, d) for c, d in columns]}
    combine = {"op": "reduce_scatterv_into_output", "group_ranks": [0, 4],
               "sizes": list(sizes), "dim": 0, "completed": True,
               "inputs": [descriptor(sum(sizes), 4096, "bfloat16")],
               "outputs": [descriptor(rows, 4096, "bfloat16")]}
    reduction = {"op": "all_reduce", "group_ranks": [0, 1, 2, 3], "dim": None,
                 "sizes": None, "completed": True,
                 "inputs": [descriptor(rows, 4096, "bfloat16")],
                 "outputs": [descriptor(rows, 4096, "bfloat16")]}
    return [deepcopy(reduction), dispatch, combine, deepcopy(reduction)]


def compare(case, calls, rows=1, sizes=(1, 1)):
    return probe.compare_dispatch(case, calls, rows=rows, sizes=list(sizes),
                                  dp_ranks=[0, 4], tp_ranks=[0, 1, 2, 3], hidden=4096)


@pytest.mark.parametrize("case_index", [0, 1])
@pytest.mark.parametrize("sizes", [(1, 1), (3, 1), (4, 8)])
def test_compare_actual_schemas_and_row_bytes(model_path, tmp_path, case_index, sizes):
    case = probe.probe_plan(arguments(model_path, tmp_path))["cases"][case_index]
    result = compare(case, recorded_calls(case, sizes[0], sizes), sizes[0], sizes)
    assert result == {"status": "match", "expected_row_bytes": case["design_row_bytes"],
                      "observed_row_bytes": [case["design_row_bytes"]], "differences": []}


@pytest.mark.parametrize("fault", ["missing", "scale_layout", "scale_dtype", "hidden_dtype",
                                   "wrong_group", "wrong_sizes", "incomplete", "no_combine",
                                   "wrong_output", "missing_tp_reduce"])
def test_discrepancies_never_become_a_pass(model_path, tmp_path, fault):
    case = probe.probe_plan(arguments(model_path, tmp_path))["cases"][0]
    calls = recorded_calls(case, 1, (1, 1))
    dispatch = calls[1]
    if fault == "missing":
        calls.pop(1)
    elif fault == "scale_layout":
        dispatch["inputs"][-1] = descriptor(32, 1, "float32")
    elif fault == "scale_dtype":
        dispatch["inputs"][-1] = descriptor(1, 32, "bfloat16")
    elif fault == "hidden_dtype":
        dispatch["inputs"][0] = descriptor(1, 4096, "bfloat16")
    elif fault == "wrong_group":
        dispatch["group_ranks"] = list(range(8))
    elif fault == "wrong_sizes":
        dispatch["sizes"] = [2, 1]
    elif fault == "incomplete":
        dispatch["completed"] = False
    elif fault == "no_combine":
        calls.pop(2)
    elif fault == "wrong_output":
        dispatch["outputs"][0]["shape"] = [1, 4096]
    else:
        calls.pop(0)
    verdict = compare(case, calls)
    assert verdict["status"] == "discrepancy"
    assert verdict["differences"]


class CpuCoordinator:
    def __init__(self, events):
        self.ranks = [0, 4]
        self.events = events

    def all_gatherv(self, input_, dim=0, sizes=None):
        self.events.append({"event": "original", "input_id": id(input_)})
        return input_

    def reduce_scatterv(self, input_, dim=-1, sizes=None):
        return input_

    def reduce_scatterv_into_output(self, input_, output, dim=-1, sizes=None):
        output.copy_(input_)
        return output

    def all_reduce(self, input_):
        return input_


def test_observation_precedes_original_and_preserves_arguments_and_result():
    torch = pytest.importorskip("torch")
    events = []
    original = CpuCoordinator.all_gatherv
    group = CpuCoordinator(events)
    observer = probe.DispatchObserver(events.append)
    tensors = [torch.zeros(1, 4096, dtype=torch.bfloat16)]
    with observer.collectives(CpuCoordinator):
        result = group.all_gatherv(tensors, 0, [1, 1])
    assert result is tensors
    assert [event["event"] for event in events] == ["collective_enter", "original", "collective_exit"]
    assert events[1]["input_id"] == id(tensors)
    assert observer.calls[0]["completed"] is True
    assert CpuCoordinator.all_gatherv is original


def test_failed_collective_keeps_entry_and_restores_original(monkeypatch):
    torch = pytest.importorskip("torch")

    def fail(self, input_, dim=0, sizes=None):
        raise RuntimeError("native collective error")

    monkeypatch.setattr(CpuCoordinator, "all_gatherv", fail)
    events = []
    observer = probe.DispatchObserver(events.append)
    with pytest.raises(RuntimeError, match="native collective error"):
        with observer.collectives(CpuCoordinator):
            CpuCoordinator(events).all_gatherv([torch.empty(1, 2)])
    assert [event["event"] for event in events] == ["collective_enter"]
    assert observer.calls[0]["completed"] is False
    assert CpuCoordinator.all_gatherv is fail


def test_into_output_signature_does_not_treat_output_as_dim():
    torch = pytest.importorskip("torch")
    observer = probe.DispatchObserver(lambda event: None)
    group = CpuCoordinator([])
    source, output = torch.ones(1, 4), torch.empty(1, 4)
    with observer.collectives(CpuCoordinator):
        result = group.reduce_scatterv_into_output(source, output, 0, [1, 1])
    assert result is output
    assert torch.equal(output, source)
    assert observer.calls[0]["dim"] == 0
    assert observer.calls[0]["sizes"] == [1, 1]
    assert observer.calls[0]["outputs"] == [probe.tensor_schema(output)]


def test_records_runtime_fallback_without_changing_its_selection():
    class TritonExperts:
        def apply(self, token):
            return token

    class DeepGemmExperts:
        def apply(self, token):
            raise AssertionError("The unselected backend must not run")

    class FallbackExperts:
        def __init__(self):
            self.experts = DeepGemmExperts()
            self.fallback_experts = TritonExperts()

        def apply(self, token):
            return self.fallback_experts.apply(token)

    experts = FallbackExperts()
    observer = probe.DispatchObserver(lambda event: None)
    token = object()
    with observer.experts(experts):
        assert experts.apply(token) is token
    assert observer.expert_calls == ["FallbackExperts", "TritonExperts"]
    assert "apply" not in experts.__dict__
    assert "apply" not in experts.fallback_experts.__dict__


def test_tensor_schema_retains_actual_scale_strides():
    torch = pytest.importorskip("torch")
    scales = torch.empty(32, 3).t()
    schema = probe.tensor_schema(scales)
    assert schema["shape"] == [3, 32]
    assert schema["stride"] == [1, 3]
    assert schema["nbytes"] == 384
    assert schema["dtype"] == "float32"


def test_official_config_and_dummy_loader_invocation_offline(
    model_path, tmp_path, monkeypatch, cpu_vllm_platform
):
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("vllm.models.step5.nvidia.model")
    from vllm.model_executor.model_loader import dummy_loader, utils

    monkeypatch.setenv("RANK", "0")
    config = probe.create_native_config(arguments(model_path, tmp_path))
    assert config.model_config.hf_config.model_type == "step5"
    assert config.parallel_config.enable_expert_parallel
    assert config.parallel_config.all2all_backend == "allgather_reducescatter"
    assert config.parallel_config.tensor_parallel_size == 4
    assert config.parallel_config.data_parallel_size == 2
    assert config.kernel_config.moe_backend == "auto"
    assert "model.layers.88.moe.experts.w13_weight" in config.quant_config.ignored_layers
    assert ".moe.share_expert." in next(x for x in config.quant_config.ignored_layers
                                       if "layers.3.moe.share_expert." in x)
    events = []
    layer = torch.nn.Identity()

    def initialize(vllm_config, *, model_class, prefix):
        assert model_class is native.Step5MoE
        assert vllm_config is config
        events.append(("initialize", prefix))
        return layer

    monkeypatch.setattr(utils, "initialize_model", initialize)
    monkeypatch.setattr(dummy_loader.DummyModelLoader, "load_weights",
                        lambda self, model, metadata: events.append(("dummy", model)))
    monkeypatch.setattr(utils, "process_weights_after_loading",
                        lambda model, metadata, device: events.append(("post_load", model)))
    case = probe.probe_plan(arguments(model_path, tmp_path))["cases"][1]
    assert probe.load_native_layer(config, case, torch.device("cpu")) is layer
    assert events == [("initialize", "model.layers.88.moe"), ("dummy", layer), ("post_load", layer)]
    assert layer.training is False


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_official_layer_prepare_finalize_and_group_calls_on_cpu(
    precision, monkeypatch, cpu_vllm_platform
):
    """Run native Python dispatch; GPU quantization, GEMMs and transport are stand-ins."""
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("vllm.models.step5.nvidia.model")
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.device_communicators import all2all
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP

    class CpuTransport:
        def all_gatherv(self, tensors, dim, sizes):
            return [torch.empty((sum(sizes), *t.shape[1:]), dtype=t.dtype) for t in tensors]

        def reduce_scatterv_into_output(self, input_, output, dim, sizes):
            output.zero_()
            return output

    group = object.__new__(ps.GroupCoordinator)
    group.ranks, group.world_size, group.rank_in_group = [0, 4], 2, 0
    group.device_communicator = CpuTransport()
    manager = object.__new__(all2all.AgRsAll2AllManager)
    manager.dp_world_size, manager.use_ep = 2, True
    manager.tp_group = SimpleNamespace(world_size=4)
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3, 1])
    monkeypatch.setattr(all2all, "get_dp_group", lambda: group)
    monkeypatch.setattr(all2all, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(all2all, "get_forward_context", lambda: SimpleNamespace(dp_metadata=metadata))
    monkeypatch.setattr(naive_dp_ep, "get_ep_group", lambda: manager)

    def quantize(input_, input_scale, **kwargs):
        if precision == "fp8":
            return input_.to(torch.float8_e4m3fn), torch.ones(input_.shape[0], 32)
        return input_, None

    monkeypatch.setattr(naive_dp_ep, "moe_kernel_quantize_input", quantize)
    prepare = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular()
    sequence = []

    class SharedExpert(torch.nn.Module):
        def forward(self, inputs):
            sequence.append("shared")
            return inputs

    class Gate(torch.nn.Module):
        def forward(self, inputs):
            sequence.append("gate")
            return torch.zeros(inputs.shape[0], 352), None

    class Experts(torch.nn.Module):
        def forward(self, hidden_states, router_logits):
            sequence.append("experts")
            assert router_logits.dtype == torch.float32
            rows = hidden_states.shape[0]
            ids = torch.zeros(rows, 8, dtype=torch.int32)
            weights = torch.ones(rows, 8, dtype=torch.float32)
            dispatched, scales, _, ids, weights = prepare.prepare(
                hidden_states, weights, ids, 352, None, False, FusedMoEQuantConfig.make())
            assert dispatched.shape == (4, 4096)
            assert (scales is not None) == (precision == "fp8")
            output = torch.empty_like(hidden_states)
            prepare.finalize(output, torch.zeros(4, 4096, dtype=torch.bfloat16),
                             weights, ids, False, TopKWeightAndReduceNoOP())
            return output

    layer = native.Step5MoE.__new__(native.Step5MoE)
    torch.nn.Module.__init__(layer)
    layer.share_expert, layer.gate, layer.experts = SharedExpert(), Gate(), Experts()
    observer = probe.DispatchObserver(lambda event: None)
    with observer.collectives(ps.GroupCoordinator):
        shared, routed = layer(torch.ones(3, 4096, dtype=torch.bfloat16))
    assert sequence == ["shared", "gate", "experts"]
    assert shared.shape == routed.shape == (3, 4096)
    assert [c["op"] for c in observer.calls] == ["all_gatherv", "reduce_scatterv_into_output"]
    assert sum(t["nbytes"] for t in observer.calls[0]["inputs"]) // 3 == (
        4288 if precision == "fp8" else 8256)
    assert observer.calls[0]["group_ranks"] == [0, 4]
    assert observer.calls[0]["sizes"] == [3, 1]
    assert all(call["completed"] for call in observer.calls)
