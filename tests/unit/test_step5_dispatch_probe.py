"""CPU diagnostics tests; none of these results qualify GPU dispatch schemas."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
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


@pytest.mark.parametrize("limits", [None, {"max_model_len": 7168,
    "max_num_batched_tokens": 8192, "max_num_seqs": 128, "unrelated": "excluded"}])
def test_native_config_binds_only_supplied_limits_before_text_submodel(model_path, tmp_path, monkeypatch, limits):
    calls=[];events=[];metadata=SimpleNamespace(max_position_embeddings=1048576)
    names=("max_model_len", "max_num_batched_tokens", "max_num_seqs")
    defaults=dict(zip(names,(1048576,2048,256)))
    expected=defaults if limits is None else {name:limits[name] for name in names}
    class EngineArgs:
        def __init__(self, **kwargs): calls.append(kwargs)
        def create_engine_config(self):
            kwargs=calls[-1];events.append("create")
            values={name:kwargs.get(name,defaults[name]) for name in names}
            def with_hf_config(hf, *, architectures):
                assert hf is metadata and architectures == ["Step5ForCausalLM"]
                events.append("text_submodel")
                return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf,max_model_len=values[names[0]]),
                    scheduler_config=SimpleNamespace(**{name:values[name] for name in names[1:]}),quant_config=object())
            return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=metadata),with_hf_config=with_hf_config)
    model_class=object()
    def quant(config, cls):
        assert cls is model_class
        events.append("quant")
    modules={"vllm.engine.arg_utils":SimpleNamespace(EngineArgs=EngineArgs),
        "vllm.model_executor.model_loader.utils":SimpleNamespace(configure_quant_config=quant),
        "vllm.models.step5.nvidia.model":SimpleNamespace(Step5ForCausalLM=model_class)}
    for name,module in modules.items(): monkeypatch.setitem(sys.modules,name,module)
    args=arguments(model_path,tmp_path)
    config=(probe.create_native_config(args) if limits is None else
            probe.create_native_config(args,engine_limits=limits))
    assert events == ["create","text_submodel","quant"] and len(calls) == 1
    assert {name:calls[0][name] for name in names if name in calls[0]} == ({} if limits is None else expected)
    assert "unrelated" not in calls[0] and calls[0]["enforce_eager"] is True
    assert {k:v for k,v in calls[0].items() if k not in names} == dict(model=str(model_path.resolve()),
        load_format="dummy",skip_tokenizer_init=True,language_model_only=True,tensor_parallel_size=4,
        data_parallel_size=2,enable_expert_parallel=True,all2all_backend="allgather_reducescatter",
        distributed_executor_backend="external_launcher",enforce_eager=True,disable_custom_all_reduce=False)
    assert config.model_config.max_model_len == expected["max_model_len"]
    assert vars(config.scheduler_config) == {name:expected[name] for name in names[1:]}
    assert config.model_config.hf_config is metadata and metadata.max_position_embeddings == 1048576


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


@pytest.mark.parametrize("tp,dp,nodes", [(4, 2, 1), (8, 4, 4), (8, 2, 2)])
def test_expected_topology_covers_every_rank_and_node(tp, dp, nodes):
    topology = probe.expected_topology(tp, dp, 8)
    assert topology["world_size"] == tp * dp
    assert topology["node_count"] == nodes
    assert topology["gpus_per_node"] == 8
    assert len(topology["ranks"]) == tp * dp
    for rank, expected in enumerate(topology["ranks"]):
        assert expected == {
            "rank": rank, "local_rank": rank % 8, "node_rank": rank // 8,
            "dp_rank": rank // tp,
            "group_ranks": {"tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
                            "dp": list(range(rank % tp, tp * dp, tp)),
                            "ep": list(range(tp * dp))}}
        assert {member // 8 for member in expected["group_ranks"]["tp"]} == {rank // 8}
    tp_groups = {tuple(r["group_ranks"]["tp"]) for r in topology["ranks"]}
    assert len(tp_groups) == dp
    assert len(tp_groups) // nodes == (2 if tp == 4 else 1)


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
def test_plan_derives_ep_and_default_tokens_from_tp_dp(model_path, tmp_path, tp, dp):
    args = arguments(model_path, tmp_path, "--tensor-parallel-size", str(tp),
                     "--data-parallel-size", str(dp), "--gpus-per-node", "8")
    plan = probe.probe_plan(args)
    assert plan["parallelism"] == {"tp": tp, "dp": dp, "ep": tp * dp}
    assert plan["topology"] == probe.expected_topology(tp, dp, 8)
    assert plan["tokens_per_dp_rank"] == [1] * dp
    assert plan["timing_rows"] == 0 and plan["cost_data"] is False
    assert plan["full_model_verified"] is False and plan["model_runner_verified"] is False
    assert [case["layer_id"] for case in plan["cases"]] == [3, 88]


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
def test_dry_run_command_needs_no_torch_or_vllm(model_path, tmp_path, tp, dp):
    tokens = [3, *([1] * (dp - 1))]
    command = [sys.executable, "-S", "-m", probe.__name__, "--model-path", str(model_path),
               "--output-dir", str(tmp_path / "unused"), "--tensor-parallel-size", str(tp),
               "--data-parallel-size", str(dp), "--gpus-per-node", "8",
               "--tokens-per-dp-rank", *map(str, tokens),
               "--disable-custom-all-reduce", "--dry-run"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "not_run"
    assert report["parallelism"] == {"tp": tp, "dp": dp, "ep": tp * dp}
    assert report["topology"]["node_count"] == tp * dp // 8
    assert report["tokens_per_dp_rank"] == tokens
    assert report["disable_custom_all_reduce"] is True
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("tokens", [(0, 1), (1, -1)])
def test_reject_zero_or_negative_tokens(model_path, tmp_path, tokens):
    with pytest.raises(SystemExit):
        arguments(model_path, tmp_path, "--tokens-per-dp-rank", *map(str, tokens))


@pytest.mark.parametrize("dp,tokens", [(2, [1]), (2, [1, 2, 3]),
                                      (4, [1, 1]), (4, [1, 1, 1, 1, 1])])
def test_reject_token_count_different_from_dp(model_path, tmp_path, capsys, dp, tokens):
    with pytest.raises(SystemExit):
        arguments(model_path, tmp_path, "--tensor-parallel-size", "8",
                  "--data-parallel-size", str(dp), "--tokens-per-dp-rank", *map(str, tokens))
    assert "one count per DP rank" in capsys.readouterr().err


@pytest.mark.parametrize("tp,dp,gpus", [(0, 2, 8), (4, -1, 8), (4, 2, 0),
                                       (16, 2, 8), (3, 8, 8), (4, 3, 8)])
def test_reject_invalid_node_topology(model_path, tmp_path, tp, dp, gpus):
    with pytest.raises(SystemExit):
        arguments(model_path, tmp_path, "--tensor-parallel-size", str(tp),
                  "--data-parallel-size", str(dp), "--gpus-per-node", str(gpus))


def test_no_independent_ep_argument(model_path, tmp_path):
    with pytest.raises(SystemExit):
        arguments(model_path, tmp_path, "--expert-parallel-size", "16")


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
def test_launcher_validates_world_separately_from_local_world(tp, dp):
    topology = probe.expected_topology(tp, dp, 8)
    probe.validate_launcher(topology, {"WORLD_SIZE": str(tp * dp), "LOCAL_WORLD_SIZE": "8"})
    with pytest.raises(ValueError, match="WORLD_SIZE=64; expected"):
        probe.validate_launcher(topology, {"WORLD_SIZE": "64", "LOCAL_WORLD_SIZE": "8"})
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE=4; expected 8"):
        probe.validate_launcher(topology, {"WORLD_SIZE": str(tp * dp), "LOCAL_WORLD_SIZE": "4"})


def placement_records(topology):
    return [{"rank": r["rank"], "local_rank": r["local_rank"], "host": f"node-{r['node_rank']}"}
            for r in topology["ranks"]]


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
def test_gathered_placement_requires_exact_node_count_and_local_ranks(tp, dp):
    topology = probe.expected_topology(tp, dp, 8)
    assert probe.compare_placement(topology, placement_records(topology)) == {
        "status": "match", "differences": []}


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
@pytest.mark.parametrize("fault", ["extra_host", "duplicate_local_rank", "swapped_local_ranks",
                                   "missing_rank", "duplicate_rank", "rank_order"])
def test_placement_mismatch_is_diagnostic(tp, dp, fault):
    topology = probe.expected_topology(tp, dp, 8)
    ranks = placement_records(topology)
    if fault == "extra_host":
        ranks[-1]["host"] = "unexpected-host"
    elif fault == "duplicate_local_rank":
        ranks[-1]["local_rank"] = 0
    elif fault == "swapped_local_ranks":
        ranks[0]["local_rank"], ranks[1]["local_rank"] = 1, 0
    elif fault == "missing_rank":
        ranks.pop()
    elif fault == "duplicate_rank":
        ranks[-1]["rank"] = 0
    else:
        ranks.reverse()
    result = probe.compare_placement(topology, ranks)
    assert result["status"] == "discrepancy"
    assert result["differences"]
    if fault == "extra_host":
        assert any(d["field"].startswith("tp_group") for d in result["differences"])


@pytest.mark.parametrize("dp", [2, 4])
def test_placement_rejects_merged_nodes_and_cross_node_tp(dp):
    topology = probe.expected_topology(8, dp, 8)
    ranks = placement_records(topology)
    ranks[-1]["host"], ranks[0]["host"] = ranks[0]["host"], ranks[-1]["host"]
    verdict = probe.compare_placement(topology, ranks)
    assert verdict["status"] == "discrepancy"
    assert sum(d["field"].startswith("tp_group") for d in verdict["differences"]) == 2
    ranks = placement_records(topology)
    for rank in ranks:
        rank["host"] = "one-host"
    verdict = probe.compare_placement(topology, ranks)
    assert verdict["status"] == "discrepancy"
    assert {d["field"] for d in verdict["differences"]} >= {"node_count", "host[one-host].gpu_count"}


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


def test_bound_observer_uses_actual_branch_policy_without_blocking_native_errors(monkeypatch):
    from frontier.profiling.collectives import vllm_collective_benchmark as benchmark
    tensor=SimpleNamespace(shape=(1,2),stride=lambda:(2,1),dtype="torch.bfloat16",
        element_size=lambda:2,numel=lambda:2,device="cpu",data_ptr=lambda:1)
    calls=[];error=RuntimeError("actual_native_error")
    def native(*a,**kw):
        calls.append("native")
        if len(calls)>1: raise error
        return tensor
    pynccl=SimpleNamespace(reduce_scatterv=native)
    group=CpuCoordinator([]);group.device_communicator=SimpleNamespace(pynccl_comm=pynccl)
    monkeypatch.setattr(CpuCoordinator,"reduce_scatterv_into_output",lambda group,input_,output,dim=0,sizes=None:pynccl.reduce_scatterv(input_,output))
    import types
    allocator=types.ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    allocator.is_symmetric_memory_tensor=lambda tensor:False
    monkeypatch.setitem(sys.modules,allocator.__name__,allocator)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(distributed=SimpleNamespace(),ops=SimpleNamespace(vllm=SimpleNamespace())))
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST","cpu_fixture")
    observer=probe.DispatchObserver(lambda event:None);observer.runtime_policy=lambda:{"cpu_fixture":True}
    with observer.collectives(CpuCoordinator):
        assert group.reduce_scatterv_into_output(tensor,tensor,0,[1,1]) is tensor
        with pytest.raises(RuntimeError) as caught: group.reduce_scatterv_into_output(tensor,tensor,0,[1,1])
    assert caught.value is error and calls == ["native","native"]
    policy=observer.calls[0]["execution_policy"]
    assert policy["branches"] == ["pynccl_comm.reduce_scatterv"] and policy["runtime"] == observer.runtime_policy()


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


@pytest.mark.parametrize("operation", ["all_gatherv", "reduce_scatterv",
                                        "reduce_scatterv_into_output", "all_reduce"])
@pytest.mark.parametrize("fault", ["schema_enter", "schema_exit", "emit_enter", "emit_exit"])
@pytest.mark.parametrize("native_error", [False, True])
def test_collective_diagnostics_never_block_or_replace_native_call(
    monkeypatch, operation, fault, native_error
):
    tensor = SimpleNamespace(shape=(1, 2), stride=lambda: (2, 1), dtype="torch.bfloat16",
                             element_size=lambda: 2, numel=lambda: 2, device="cpu")
    calls, events = [], []
    error = RuntimeError("native collective failed")

    def original(group, input_, dim=0, sizes=None):
        calls.append((group, input_, dim, sizes))
        if native_error:
            raise error
        return input_

    monkeypatch.setattr(CpuCoordinator, operation, original)
    schema = probe.tensor_schema
    schema_calls = []

    def describe(input_):
        schema_calls.append(input_)
        if fault == ("schema_enter" if len(schema_calls) == 1 else "schema_exit"):
            raise ValueError("diagnostic schema failed")
        return schema(input_)

    def emit(event):
        if fault == ("emit_enter" if event["event"] == "collective_enter" else "emit_exit"):
            raise OSError("diagnostic emit failed")
        events.append(event)

    monkeypatch.setattr(probe, "tensor_schema", describe)
    observer = probe.DispatchObserver(emit)
    group = CpuCoordinator([])
    inputs, sizes = [tensor], [1, 1]
    with observer.collectives(CpuCoordinator):
        if native_error:
            with pytest.raises(RuntimeError) as caught:
                getattr(group, operation)(inputs, 0, sizes)
            assert caught.value is error
        else:
            assert getattr(group, operation)(inputs, 0, sizes) is inputs
    assert calls == [(group, inputs, 0, sizes)]
    if not native_error or fault.endswith("enter"):
        assert observer.errors
        assert "diagnostic" in observer.errors[0]["error"]
    assert getattr(CpuCoordinator, operation) is original


@pytest.mark.parametrize("native_error", [False, True])
@pytest.mark.parametrize("broken_stderr", [False, True])
def test_expert_emit_error_does_not_block_or_replace_native_apply(native_error, broken_stderr, monkeypatch):
    calls = []
    error = RuntimeError("native expert failed")

    class Experts:
        def apply(self, token):
            calls.append(token)
            if native_error:
                raise error
            return token

    def emit(event):
        raise OSError("expert diagnostic emit failed")

    if broken_stderr:
        monkeypatch.setattr(probe, "sys", SimpleNamespace(stderr=SimpleNamespace(write=emit)))
    observer = probe.DispatchObserver(emit)
    experts, token = Experts(), object()
    with observer.experts(experts):
        if native_error:
            with pytest.raises(RuntimeError) as caught:
                experts.apply(token)
            assert caught.value is error
        else:
            assert experts.apply(token) is token
    assert calls == [token]
    assert observer.errors
    assert "expert diagnostic emit failed" in observer.errors[0]["error"]
    assert "apply" not in experts.__dict__


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


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
@pytest.mark.parametrize("fault", [None, "group", "placement", "native_error", "output_directory",
                                   "log_open", "log_write", "log_close", "rank_write", "aggregate_write",
                                   "observer_schema", "backend", "comparison", "placement_comparison",
                                   "report_stdout", "native_error_rank_write", "native_error_log_close"])
def test_probe_report_gather_and_diagnostic_verdict_with_cpu_standins(
    model_path, tmp_path, monkeypatch, capsys, tp, dp, fault
):
    """Exercise report orchestration only, not native layer or GPU execution."""
    torch = pytest.importorskip("torch")
    args = arguments(model_path, tmp_path, "--tensor-parallel-size", str(tp),
                     "--data-parallel-size", str(dp))
    plan = probe.probe_plan(args)
    topology = plan["topology"]
    expected = topology["ranks"][0]
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0),
                             model_config=SimpleNamespace(dtype=torch.bfloat16,
                                                          hf_text_config=SimpleNamespace(hidden_size=4096)))
    groups = {name: SimpleNamespace(ranks=list(ranks))
              for name, ranks in expected["group_ranks"].items()}
    if fault == "group":
        groups["dp"].ranks.reverse()
    cpu_group = object()
    lifecycle, executed_layers, attempted_writes = [], [], []
    native_error = RuntimeError("native execution failed")
    ps = SimpleNamespace(
        GroupCoordinator=CpuCoordinator,
        get_tp_group=lambda: groups["tp"], get_dp_group=lambda: groups["dp"],
        get_ep_group=lambda: groups["ep"],
        get_world_group=lambda: SimpleNamespace(cpu_group=cpu_group),
        destroy_model_parallel=lambda: lifecycle.append("model_parallel_destroyed"),
        destroy_distributed_environment=lambda: lifecycle.append("distributed_destroyed"))
    modules = {
        "vllm": SimpleNamespace(__version__="cpu-standin"),
        "vllm.config": SimpleNamespace(set_current_vllm_config=lambda config: nullcontext()),
        "vllm.distributed": SimpleNamespace(parallel_state=ps),
        "vllm.forward_context": SimpleNamespace(set_forward_context=lambda *a, **kw: nullcontext()),
        "vllm.v1.worker.gpu_worker": SimpleNamespace(
            init_worker_distributed_environment=lambda *a: lifecycle.append("initialized")),
        "vllm.v1.worker.workspace": SimpleNamespace(init_workspace_manager=lambda *a: None)}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": str(tp * dp),
                        "LOCAL_WORLD_SIZE": "8"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "node-0")
    monkeypatch.setattr(probe, "create_native_config", lambda args: config)
    monkeypatch.setattr(probe, "native_backend", lambda layer: {"experts_class": "cpu-standin"})
    monkeypatch.setattr(probe, "compare_dispatch", lambda *a, **kw: {"status": "match"})

    def fail_diagnostic(*a, **kw):
        raise OSError(f"diagnostic {fault} failed")

    if fault == "observer_schema":
        monkeypatch.setattr(probe, "tensor_schema", fail_diagnostic)
    elif fault == "backend":
        monkeypatch.setattr(probe, "native_backend", fail_diagnostic)
    elif fault == "comparison":
        monkeypatch.setattr(probe, "compare_dispatch", fail_diagnostic)
    elif fault == "placement_comparison":
        monkeypatch.setattr(probe, "compare_placement", fail_diagnostic)
    elif fault == "report_stdout":
        def print_report(*a, **kw):
            if a[0] == "STEP5_DISPATCH_PROBE_REPORT":
                fail_diagnostic()
            return print(*a, **kw)

        monkeypatch.setattr(probe, "print", print_report, raising=False)

    mkdir, open_file, write_text = Path.mkdir, Path.open, Path.write_text

    def mkdir_directory(path, *a, **kw):
        if path == args.output_dir and fault == "output_directory":
            fail_diagnostic()
        return mkdir(path, *a, **kw)

    class Log:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            if fault == "log_write":
                fail_diagnostic()
            return self.stream.write(text)

        def close(self):
            self.stream.close()
            if fault in ("log_close", "native_error_log_close"):
                fail_diagnostic()

    def open_log(path, *a, **kw):
        if path == args.output_dir / "rank_0.jsonl":
            if fault == "log_open":
                fail_diagnostic()
            return Log(open_file(path, *a, **kw))
        return open_file(path, *a, **kw)

    def save_report(path, text, *a, **kw):
        attempted_writes.append((path.name, json.loads(text)))
        if ((path.name == "rank_0.json" and fault in ("rank_write", "native_error_rank_write"))
                or (path.name == "dispatch_probe.json" and fault == "aggregate_write")):
            fail_diagnostic()
        return write_text(path, text, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", mkdir_directory)
    monkeypatch.setattr(Path, "open", open_log)
    monkeypatch.setattr(Path, "write_text", save_report)
    for name in ("set_device", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: randn(*a, **{**kw, "device": "cpu"}))

    class Layer:
        experts = SimpleNamespace(routed_experts=SimpleNamespace(quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(fused_experts=SimpleNamespace(apply=lambda: None)))))

        def __init__(self, layer_id):
            self.layer_id = layer_id

        def __call__(self, inputs):
            executed_layers.append(self.layer_id)
            CpuCoordinator([]).all_gatherv([inputs], 0, args.tokens_per_dp_rank)
            self.experts.routed_experts.quant_method.moe_kernel.fused_experts.apply()
            if fault in ("native_error", "native_error_rank_write", "native_error_log_close"):
                raise native_error
            return inputs, inputs

    monkeypatch.setattr(probe, "load_native_layer", lambda config, case, device: Layer(case["layer_id"]))

    def gather(ranks, report, group):
        assert group is cpu_group
        assert len(ranks) == tp * dp
        for index, expected_rank in enumerate(topology["ranks"]):
            ranks[index] = {**deepcopy(report), **expected_rank,
                            "host": f"node-{expected_rank['node_rank']}", "status": "match"}
        ranks[0] = deepcopy(report)
        if fault == "placement":
            ranks[-1]["host"] = "unexpected-host"
        lifecycle.append("gathered")

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if fault in ("native_error", "native_error_rank_write", "native_error_log_close"):
        with pytest.raises(RuntimeError, match="native execution failed") as caught:
            probe.run_probe(args, plan)
        assert caught.value is native_error
        rank_report = attempted_writes[-1][1]
        assert rank_report["status"] == "runtime_error"
        assert "native execution failed" in rank_report["error"]
        assert not (args.output_dir / "dispatch_probe.json").exists()
        assert "gathered" not in lifecycle
        assert executed_layers == [3]
        return
    report = probe.run_probe(args, plan)
    if (args.output_dir / "dispatch_probe.json").exists():
        assert json.loads((args.output_dir / "dispatch_probe.json").read_text()) == json.loads(json.dumps(report))
    assert report["status"] == ("match" if fault is None else "discrepancy")
    assert executed_layers == [3, 88]
    assert len(report["ranks"]) == tp * dp
    assert len(report["ranks"][0]["layers"]) == 2
    assert report["full_model_verified"] is False and report["model_runner_verified"] is False
    assert report["cost_data"] is False and report["timing_rows"] == 0
    assert report["ranks"][0]["group_comparison"]["status"] == (
        "discrepancy" if fault == "group" else "match")
    assert report["placement_comparison"]["status"] == (
        "unverified" if fault == "placement_comparison" else
        "discrepancy" if fault == "placement" else "match")
    if fault not in (None, "group", "placement"):
        rank_report = report["ranks"][0]
        assert (report["diagnostic_errors"] or rank_report["diagnostic_errors"]
                or any(layer["observer_errors"] for layer in rank_report["layers"]))
        streams = capsys.readouterr()
        assert "STEP5_DISPATCH_PROBE_DIAGNOSTIC_ERROR" in streams.err
        assert '"status": "match"' not in streams.out
        if fault in ("observer_schema", "backend", "comparison"):
            assert all(layer["observer_errors"] and layer["comparison"]["status"] == "unverified"
                       for layer in rank_report["layers"])
    assert lifecycle == ["initialized", "gathered", "model_parallel_destroyed", "distributed_destroyed"]


@pytest.mark.parametrize("tp,dp", [(4, 2), (8, 4), (8, 2)])
def test_official_config_and_dummy_loader_invocation_offline(
    model_path, tmp_path, monkeypatch, cpu_vllm_platform, tp, dp
):
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("vllm.models.step5.nvidia.model")
    from vllm.model_executor.model_loader import dummy_loader, utils

    monkeypatch.setenv("RANK", "0")
    args = arguments(model_path, tmp_path, "--tensor-parallel-size", str(tp),
                     "--data-parallel-size", str(dp))
    config = probe.create_native_config(args)
    assert config.model_config.hf_config.model_type == "step5"
    assert config.parallel_config.enable_expert_parallel
    assert config.parallel_config.all2all_backend == "allgather_reducescatter"
    assert config.parallel_config.tensor_parallel_size == tp
    assert config.parallel_config.data_parallel_size == dp
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
    case = probe.probe_plan(args)["cases"][1]
    assert probe.load_native_layer(config, case, torch.device("cpu")) is layer
    assert events == [("initialize", "model.layers.88.moe"), ("dummy", layer), ("post_load", layer)]
    assert layer.training is False


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
@pytest.mark.parametrize("tp,dp,dp_rank", [(4, 2, 0), (4, 2, 1), (8, 4, 0),
                                         (8, 4, 3), (8, 2, 0), (8, 2, 1)])
def test_official_layer_prepare_finalize_and_group_calls_on_cpu(
    precision, monkeypatch, cpu_vllm_platform, tp, dp, dp_rank
):
    """Run native Python dispatch; GPU quantization, GEMMs and transport are stand-ins."""
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("vllm.models.step5.nvidia.model")
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.device_communicators import all2all
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP

    topology = probe.expected_topology(tp, dp, 8)
    rank = dp_rank * tp + tp - 1
    expected = topology["ranks"][rank]
    sizes = [3, *([1] * (dp - 1))]
    rows, total_rows = sizes[dp_rank], sum(sizes)

    class CpuTransport:
        def all_gatherv(self, tensors, dim, sizes):
            return [torch.empty((sum(sizes), *t.shape[1:]), dtype=t.dtype) for t in tensors]

        def reduce_scatterv_into_output(self, input_, output, dim, sizes):
            output.zero_()
            return output

    group = object.__new__(ps.GroupCoordinator)
    group.ranks, group.world_size, group.rank_in_group = expected["group_ranks"]["dp"], dp, dp_rank
    group.device_communicator = CpuTransport()
    manager = object.__new__(all2all.AgRsAll2AllManager)
    manager.dp_world_size, manager.use_ep = dp, True
    manager.tp_group = SimpleNamespace(world_size=tp)
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: sizes)
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
            assert dispatched.shape == (total_rows, 4096)
            assert (scales is not None) == (precision == "fp8")
            output = torch.empty_like(hidden_states)
            prepare.finalize(output, torch.zeros(total_rows, 4096, dtype=torch.bfloat16),
                             weights, ids, False, TopKWeightAndReduceNoOP())
            return output

    layer = native.Step5MoE.__new__(native.Step5MoE)
    torch.nn.Module.__init__(layer)
    layer.share_expert, layer.gate, layer.experts = SharedExpert(), Gate(), Experts()
    observer = probe.DispatchObserver(lambda event: None)
    with observer.collectives(ps.GroupCoordinator):
        shared, routed = layer(torch.ones(rows, 4096, dtype=torch.bfloat16))
    assert sequence == ["shared", "gate", "experts"]
    assert shared.shape == routed.shape == (rows, 4096)
    assert [c["op"] for c in observer.calls] == ["all_gatherv", "reduce_scatterv_into_output"]
    assert sum(t["nbytes"] for t in observer.calls[0]["inputs"]) // rows == (
        4288 if precision == "fp8" else 8256)
    assert observer.calls[0]["group_ranks"] == expected["group_ranks"]["dp"]
    assert observer.calls[0]["sizes"] == sizes
    assert all(call["completed"] for call in observer.calls)
def test_bound_probe_plans_exact_input_layer_keys(model_path,tmp_path,monkeypatch):
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST",str(tmp_path/"manifest.json"))
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY","position_cyclic")
    from frontier.profiling.moe.inputs import MoESourceFrame, MoERoutingProfileInput
    monkeypatch.setattr("frontier.profiling.moe.inputs.read_routing_profile_inputs",lambda path:[
        MoERoutingProfileInput("tail16",(
            MoESourceFrame(0,(16,),(6128,),16,"standalone_legacy"),
            MoESourceFrame(1,(),(),1,"standalone_legacy")),17)])
    args=arguments(model_path,tmp_path,"--routing-input-file",str(tmp_path/"inputs.json"),
        "--moe-input-ids","tail16","--tensor-parallel-size","8")
    plan=probe.probe_plan(args)
    assert [(c["routing_input"]["moe_input_id"],c["layer_id"]) for c in plan["cases"]] == [("tail16",3),("tail16",88)]
    assert plan["verification_scope"] == "eager_native_moe_module"
    assert not plan["full_model_verified"] and "tokens_per_dp_rank" not in plan


def test_bound_forward_materializes_core_positions_once_and_preserves_native_exception(monkeypatch):
    torch=pytest.importorskip("torch")
    from contextlib import contextmanager
    from frontier.profiling.moe.inputs import MoESourceFrame,source_frame_tensors
    materialized=[];events=[]
    def materialize(frame,*,device):
        materialized.append(frame)
        return source_frame_tensors(frame,device=device)
    @contextmanager
    def forward(*args,**kwargs):
        assert kwargs["is_padding"].tolist() == [False]*16+[True]
        events.append("forward_enter")
        try: yield
        finally: events.append("forward_exit")
    @contextmanager
    def positions(value):
        assert value.dtype == torch.int64 and value.tolist() == list(range(6128,6144))+[0]
        events.append("positions_enter")
        try: yield
        finally: events.append("positions_exit")
    monkeypatch.setattr("frontier.profiling.moe.inputs.source_frame_tensors",materialize)
    monkeypatch.setitem(sys.modules,"vllm.forward_context",SimpleNamespace(set_forward_context=forward,set_routing_positions=positions))
    frame=dict(source_lane_id=0,query_lens=[16],context_lens=[6128],physical_rows=17,gating_runtime_context="standalone_legacy")
    error=RuntimeError("native_failed")
    with pytest.raises(RuntimeError) as caught:
        with probe.source_forward_context(object(),17,[17,1],"cpu",dict(source_frames=[frame]),0): raise error
    assert caught.value is error and materialized == [MoESourceFrame(**frame)]
    assert events == ["forward_enter","positions_enter","positions_exit","forward_exit"]
