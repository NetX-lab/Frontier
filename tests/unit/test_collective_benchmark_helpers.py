"""CPU checks for the pure helpers of the vLLM collective, NIXL READ and EP + NIXL benchmarks."""

import csv
import json
import sys
import types
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from frontier.profiling.collectives.ep_kv_read_overlap_benchmark import (
    ep_plan,
    expected_notifications,
    issue_load,
    load_runs,
    overlap_read,
    summarize_overlap,
)
from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVRegionLayout,
    ReadCase,
    block_descriptors,
    descriptor_ids,
    read_cases,
    request_block_ids,
    summarize_case,
)
from frontier.profiling.collectives.vllm_collective_benchmark import (
    CollectiveRunner,
    actual_communicator,
    active_dp_rails,
    forward_totals,
    main as collective_main,
    network_all_reduce_rows,
    pass_order,
    summarize,
    write_csv,
)
from frontier.profiling.collectives.vllm_forward_collectives import (
    CollectiveCall,
    ModelShape,
    ServingParallelism,
    group_members,
)

KV_CACHE = {
    "regions": [{"name": "main_mla", "count": 3, "block_bytes": 1152},
                {"name": "indexer", "count": 2, "block_bytes": 132}],
    "block_size": 64, "num_gpu_blocks": 10, "resident_bytes_per_block": 3 * 1152 + 2 * 132,
}


def _record(call="c", op="all_reduce", group="tp", graph=False, timings=None, nbytes=1024):
    timings = timings or [[1.0, 2.0, 3.0], [4.0, 1.0, 1.0], None]
    return {"pass": 1, "scenario": "s", "call": call, "op": op, "group": group, "group_size": 8,
            "cuda_graph": graph, "calls_per_forward": 3, "tokens_per_dp_rank": [4, 1],
            "useful_tokens_per_dp_rank": [4, 0], "padded_tokens_per_dp_rank": [4, 1],
            "dtypes": ["bfloat16"], "columns": [4096], "concurrent_dp_rails": None,
            "participating_ranks": [0, 1], "group_ranks": [[0, 1]],
            "max_input_bytes": nbytes, "max_output_bytes": nbytes, "kernels": ["k"],
            "per_rank_call_us": timings,
            "per_rank_launch_call_us": [r and [10 * v for v in r] for r in timings],
            "per_rank_queue_held": [r and [True, True, False] for r in timings]}


def test_summary_takes_slowest_rank_per_repeat():
    row = summarize(_record())
    assert (row["median_us"], row["min_us"], row["max_us"]) == (3.0, 2.0, 4.0)
    assert row["launch_median_us"] == 30.0
    assert row["queue_held_fraction"] == pytest.approx(2 / 3)
    assert row["tokens_per_dp_rank"] == "4/1"
    cpu = {**_record(op="cpu_all_reduce"), "per_rank_launch_call_us": [None, None, None],
           "per_rank_queue_held": [None, None, None]}
    assert summarize(cpu)["launch_median_us"] is None
    assert summarize(cpu)["queue_held_fraction"] is None
    totals = forward_totals([row, summarize(_record(call="d"))])
    assert totals == [{"pass": 1, "scenario": "s", "cuda_graph": False,
                       "tokens_per_dp_rank": "4/1", "forward_device_comm_us": 18.0,
                       "forward_host_comm_us": 0.0,
                       "c_us": 9.0, "d_us": 9.0}]
    other_pass = forward_totals([row, {**row, "pass": 2}])
    assert [t["pass"] for t in other_pass] == [1, 2]


@pytest.mark.parametrize("begin_complete", [False, True])
def test_gpu_repeat_checks_the_begin_event_not_alignment_host_time(monkeypatch, begin_complete):
    import frontier.profiling.collectives.vllm_collective_benchmark as benchmark

    runner = _runner(_parallel())
    runner.settings["calls_per_repeat"] = 5
    clock = [0.0]
    events = []
    recorded = []

    class Event:
        def __init__(self, **kwargs):
            events.append(self)

        def record(self):
            recorded.append(self)

        def synchronize(self):
            pass

        def query(self):
            assert events[-1] in recorded
            return begin_complete

        def elapsed_time(self, other):
            return 0.05 if self is events[-2] else 0.1

    runner.torch = types.SimpleNamespace(
        distributed=types.SimpleNamespace(barrier=lambda **kwargs: None),
        cuda=types.SimpleNamespace(Event=Event, _sleep=lambda cycles: None))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "_align_start", lambda group: clock.__setitem__(0, clock[0] + 0.001))

    def run():
        clock[0] += 0.000020

    result = runner.gpu_repeat(run, runner.groups["tp"], 100)
    assert result[1] is (not begin_complete)
    assert result[2] == pytest.approx(20.0)
    assert result[0] == pytest.approx(10.0)


def test_queue_sleep_budget_uses_the_measured_cpu_enqueue_window():
    runner = _runner(_parallel())
    runner.sleep_cycles_per_us = 2

    class Tensor:
        def __init__(self, values, **kwargs):
            self.value = values[0]

        def item(self):
            return self.value

    runner.torch = types.SimpleNamespace(
        tensor=Tensor, float64="float64",
        distributed=types.SimpleNamespace(ReduceOp=types.SimpleNamespace(MAX="max"),
                                          all_reduce=lambda *args, **kwargs: None))
    assert runner.queue_sleep_cycles([20.0, 100.0, 80.0], runner.groups["tp"]) == 400


def test_device_timing_calibrates_from_cpu_enqueue_and_retains_diagnostics(monkeypatch):
    runner = _runner(_parallel())
    runner.torch = types.SimpleNamespace()
    runner.settings["repeats"] = 2
    runner.sleep_cycles_per_us = 2
    samples = iter([(1.0, False, 100.0), (2.0, False, 200.0),
                    (3.0, True, 150.0), (4.0, False, 170.0)])
    sleeps = []

    def repeat(run, group, sleep_cycles):
        sleeps.append(sleep_cycles)
        return next(samples)

    def budget(enqueue, group):
        assert enqueue == [100.0, 200.0]
        return 800

    monkeypatch.setattr(runner, "gpu_repeat", repeat)
    monkeypatch.setattr(runner, "queue_sleep_cycles", budget)
    result = runner.time_call(_call(runner.parallel, captured=True), lambda: None)
    assert sleeps == [0, 0, 800, 800]
    assert result == {"call_us": [3.0, 4.0], "launch_call_us": [1.0, 2.0],
                      "queue_held": [True, False], "launch_enqueue_us": [100.0, 200.0],
                      "enqueue_us": [150.0, 170.0], "queue_sleep_us": 400.0}


@pytest.mark.parametrize("op,kernels,expected", [
    ("all_reduce", ["void flashinfer::trtllm_allreduce::oneShotAllReduceKernel<half>(...)"],
     "flashinfer_trtllm"),
    ("all_reduce", ["flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport"],
     "flashinfer_trtllm"),
    ("all_reduce", ["void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>"],
     "symmetric_memory"),
    ("all_reduce", ["void two_shot_all_reduce_kernel<half>(...)"], "symmetric_memory"),
    ("all_reduce", ["void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(...)"],
     "custom_all_reduce"),
    ("all_reduce", ["void vllm::cross_device_reduce_2stage<half, 8>(...)"], "custom_all_reduce"),
    ("all_gatherv", ["ncclDevKernel_AllGather_RING_LL(...)"], "nccl"),
    ("reduce_scatterv", ["ncclKernel_Reduce_Sum_bf16_RING_LL(...)"], "nccl"),
    ("all_reduce", ["Memcpy DtoD (Device -> Device)", "vllm::cross_device_reduce_1stage<half>"],
     "custom_all_reduce"),
    ("all_reduce", ["vllm::cross_device_reduce_1stage<half>", "multimem_all_reduce_kernel"],
     "symmetric_memory;custom_all_reduce"),
    ("cpu_all_reduce", [], "host_gloo"),
    ("all_reduce", [], "unknown"),
    ("all_reduce", ["Memcpy DtoD (Device -> Device)"], "unknown"),
    ("all_reduce", ["unrecognized_all_reduce_kernel"], "unknown"),
])
def test_actual_communicator_uses_kernel_evidence(op, kernels, expected):
    assert actual_communicator(op, kernels) == expected


def test_summary_csv_schema_and_separate_host_device_totals(tmp_path):
    device = summarize({**_record(), "kernels": ["ncclDevKernel_AllReduce"]})
    host = summarize({**_record(call="dp_sync", op="cpu_all_reduce", group="dp"),
                      "kernels": [], "per_rank_launch_call_us": [None] * 3,
                      "per_rank_queue_held": [None] * 3, "concurrent_dp_rails": 1})
    path = tmp_path / "summary.csv"
    write_csv(path, [device, host])
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["communicator"] for row in rows] == ["nccl", "host_gloo"]
    assert [row["timing_domain"] for row in rows] == ["device", "host"]
    assert rows[0]["useful_tokens_per_dp_rank"] == "4/0"
    assert rows[0]["padded_tokens_per_dp_rank"] == "4/1"
    assert rows[0]["dtypes"] == "bfloat16" and rows[0]["columns"] == "4096"
    assert json.loads(rows[0]["group_ranks"]) == [[0, 1]]
    assert rows[1]["launch_median_us"] == rows[1]["queue_held_fraction"] == ""
    (total,) = forward_totals([device, host])
    assert total["forward_device_comm_us"] == total["forward_host_comm_us"] == 9.0
    assert "forward_comm_us" not in total


@pytest.mark.parametrize("host_first", [False, True])
def test_forward_graph_identity_is_independent_of_shuffled_host_sync(host_first):
    device = summarize(_record(graph=True))
    host = summarize({**_record(call="dp_sync", op="cpu_all_reduce", group="dp"),
                      "kernels": [], "per_rank_launch_call_us": [None] * 3,
                      "per_rank_queue_held": [None] * 3})
    (result,) = forward_totals([host, device] if host_first else [device, host])
    assert result["cuda_graph"] is True


def _parallel(tp=8, dp=4):
    return ServingParallelism.from_engine({
        "tensor_parallel_size": tp, "data_parallel_size": dp, "pipeline_parallel_size": 1,
        "enable_expert_parallel": True, "dtype": "bfloat16", "async_scheduling": True})


def _call(parallel, op="all_gatherv", group="dp", captured=False):
    return CollectiveCall("call", op, group, 0, (4,), ("bfloat16",),
                          (1,) * parallel.world_size, 1, "test", captured)


def _runner(parallel, rank=0, instance=0, isolated=False):
    runner = CollectiveRunner.__new__(CollectiveRunner)
    runner.parallel, runner.instance_rank, runner.instance = parallel, rank, instance
    runner.dp_rail_isolation = isolated
    runner.settings = {"repeats": 3, "calls_per_repeat": 2, "warmup_calls": 1}
    offset = instance * parallel.world_size
    runner.groups = {name: types.SimpleNamespace(
        ranks=[offset + member for member in group_members(parallel, name, rank)],
        world_size=len(group_members(parallel, name, rank)),
        cpu_group=f"{name}_cpu", device_group=f"{name}_device")
        for name in ("tp", "dp", "ep")}
    runner.instance_cpu = "instance_cpu"
    runner.device = "cuda:0"
    return runner


@pytest.mark.parametrize("isolated,rails", [(False, None), (True, None), (False, (0,)),
                                          (False, tuple(range(8)))])
@pytest.mark.parametrize("op,group", [("all_gatherv", "dp"), ("reduce_scatterv", "dp"),
                                     ("cpu_all_reduce", "dp"), ("all_reduce", "tp")])
def test_rail_participation_preserves_native_tp_and_one_host_sync(isolated, rails, op, group):
    parallel = _parallel()
    call = _call(parallel, op, group)
    scenario = types.SimpleNamespace(active_dp_rails=rails)
    ranks = [rank for rank in range(parallel.world_size)
             if _runner(parallel, rank, isolated=isolated).participates(call, scenario)]
    only_one_rail = group == "dp" and (isolated or rails == (0,) or op == "cpu_all_reduce")
    assert ranks == (list(range(0, 32, 8)) if only_one_rail else list(range(32)))


def test_rail_selection_uses_scenario_indices_and_keeps_zero_row_skip():
    parallel = _parallel()
    call = _call(parallel)
    scenario = types.SimpleNamespace(active_dp_rails=(1, 3))
    assert active_dp_rails(parallel, call, scenario) == (1, 3)
    assert _runner(parallel, rank=1).participates(call, scenario)
    assert not _runner(parallel, rank=0).participates(call, scenario)
    empty = replace(_call(parallel, group="tp"), shard_rows=(0,) * parallel.world_size)
    assert not _runner(parallel).participates(empty)


@pytest.mark.parametrize("op", ["all_gatherv", "reduce_scatterv", "cpu_all_reduce"])
def test_inactive_rank_still_joins_instance_barrier_and_gather(op):
    runner = _runner(_parallel(), rank=1, isolated=True)
    events = []
    runner.torch = types.SimpleNamespace(distributed=types.SimpleNamespace(
        barrier=lambda **kwargs: events.append(("barrier", kwargs["group"]))))
    runner.gather = lambda local: events.append(("gather", local))
    runner.prepare = lambda call: pytest.fail("Inactive rank must not allocate or issue a DP call")
    call = _call(runner.parallel, op=op)
    assert runner.measure(types.SimpleNamespace(), call) is None
    assert runner.kernel_names(call) is None
    assert events == [("barrier", "instance_cpu"), ("gather", None),
                       ("barrier", "instance_cpu"), ("gather", {"kernels":[], "graph_events":[]})]


def test_call_record_retains_shapes_active_groups_and_useful_padding():
    parallel = _parallel()
    runner = _runner(parallel, instance=1, isolated=True)
    scenario = types.SimpleNamespace(name="idle_graph", tokens_per_dp_rank=(8,) * 4,
                                     useful_tokens_per_dp_rank=(8, 8, 8, 0),
                                     padded_tokens_per_dp_rank=(8,) * 4)
    gathered = [{"bytes": (4, 16)} if rank % 8 == 0 else None for rank in range(32)]
    record = runner.call_record(scenario, _call(parallel), gathered)
    assert record["participating_ranks"] == [32, 40, 48, 56]
    assert record["group_ranks"] == [[32, 40, 48, 56]]
    assert record["concurrent_dp_rails"] == 1
    assert record["useful_tokens_per_dp_rank"] == [8, 8, 8, 0]
    assert record["padded_tokens_per_dp_rank"] == [8] * 4
    assert record["dtypes"] == ["bfloat16"] and record["columns"] == [4]
    del scenario.useful_tokens_per_dp_rank, scenario.padded_tokens_per_dp_rank
    assert runner.call_record(scenario, _call(parallel), gathered)["useful_tokens_per_dp_rank"] == [8] * 4


def test_fp8_allocation_casts_supported_fill_dtype():
    runner = _runner(_parallel())
    allocations, casts = [], []

    def ones(shape, *, dtype, device):
        allocations.append((shape, dtype, device))
        return types.SimpleNamespace(to=lambda **kwargs: casts.append(kwargs["dtype"]) or kwargs["dtype"])

    runner.torch = types.SimpleNamespace(ones=ones, bfloat16="bf16", float8_e4m3fn="fp8",
                                        float32="fp32", int32="int32")
    call = replace(_call(runner.parallel), columns=(4096, 8, 8, 32),
                   dtypes=("float8_e4m3fn", "float32", "int32", "float32"))
    assert runner._tensors(call, 3, "cuda:0") == ["fp8", "fp32", "int32", "fp32"]
    assert allocations[0] == ((3, 4096), "bf16", "cuda:0")
    assert casts == ["fp8", "fp32", "int32", "fp32"]


@pytest.mark.parametrize("op",["reduce_scatterv_into_output","local_copy","local_add"])
def test_preallocated_issue_keeps_real_buffers_and_has_no_extra_copy(monkeypatch,op):
    allocations,copies,calls=[],[],[]
    class Tensor:
        dtype="bf16"
        def __init__(self,shape): self.shape=shape;allocations.append(self)
        def to(self,**kwargs): return self
        def stride(self): return self.shape[1],1
        def element_size(self): return 2
        def data_ptr(self): return id(self)
        def copy_(self,source): copies.append((self,source));return self
    runner=_runner(_parallel(tp=8,dp=2))
    runner.torch=types.SimpleNamespace(bfloat16="bf16",float8_e4m3fn="fp8",
        ones=lambda shape,**kwargs:Tensor(shape),empty_like=lambda tensor:Tensor(tensor.shape),
        add=lambda a,b,**kwargs:calls.append((a,b,kwargs["out"])))
    group=runner.groups["dp"]
    group.device_communicator=types.SimpleNamespace(pynccl_comm=types.SimpleNamespace(nccl_version=22902))
    def into_output(tensor,output,**kwargs):
        calls.append((output,tensor,kwargs));return output
    group.reduce_scatterv_into_output=into_output
    vllm=types.ModuleType("vllm");vllm.envs=types.SimpleNamespace(VLLM_USE_NCCL_SYMM_MEM=False,VLLM_BATCH_INVARIANT=False)
    allocator=types.ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    allocator.is_symmetric_memory_tensor=lambda tensor:False
    utils=types.ModuleType("vllm.distributed.device_communicators.all_reduce_utils")
    utils.should_nccl_symm_mem_ag_rs=lambda:False
    cuda=types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
    cuda.NCCL_DIRECT_SYMM_RS_OUTPUT_MIN_VERSION=22902
    monkeypatch.setitem(sys.modules,"vllm",vllm)
    monkeypatch.setitem(sys.modules,allocator.__name__,allocator)
    monkeypatch.setitem(sys.modules,utils.__name__,utils)
    monkeypatch.setitem(sys.modules,cuda.__name__,cuda)
    call=replace(_call(runner.parallel,op=op,group="dp"),shard_rows=(16,)*8+(1,)*8)
    issue=runner._issue(call);count=len(allocations)
    issue();issue()
    assert len(allocations) == count
    if op == "reduce_scatterv_into_output":
        receipt=issue.buffer_receipt
        assert receipt["input"]["shape"] == [17,4] and receipt["output"]["shape"] == [16,4]
        assert receipt["return_output_alias"] and not receipt["input_output_alias"]
        assert receipt["staging_copy_count"] == receipt["staging_copy_bytes"] == 0
        assert all(c[0] is calls[0][0] and c[1] is calls[0][1] for c in calls)
        assert not copies
    elif op == "local_add":
        assert len(calls) == 2 and not copies
    else:
        assert len(copies) == 2


def test_host_prepare_and_timing_never_use_cuda(monkeypatch):
    runner = _runner(_parallel())
    issues, barriers = [], []
    runner._issue = lambda call: lambda: issues.append(1)
    runner.torch = types.SimpleNamespace(distributed=types.SimpleNamespace(
        barrier=lambda **kwargs: barriers.append(kwargs["group"])))
    ticks = iter([1.0, 1.000002, 2.0, 2.000004, 3.0, 3.000006])
    monkeypatch.setattr("frontier.profiling.collectives.vllm_collective_benchmark.time.perf_counter",
                        lambda: next(ticks))
    call = _call(runner.parallel, op="cpu_all_reduce")
    issue, run = runner.prepare(call)
    timing = runner.time_call(call, run)
    assert timing["call_us"] == pytest.approx([1.0, 2.0, 3.0])
    assert timing["launch_call_us"] is timing["queue_held"] is None
    assert len(issues) == 1 + 3 * 2 and barriers == ["dp_cpu"] * 3


def test_graph_capture_enters_only_call_group():
    runner = _runner(_parallel(), isolated=True)
    events = []

    @contextmanager
    def capture():
        events.append("dp_capture")
        yield types.SimpleNamespace(stream="dp_stream")

    graph = types.SimpleNamespace(replay=lambda: events.append("replay"))
    runner.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        synchronize=lambda: None, CUDAGraph=lambda: graph,
        graph=lambda graph, **kwargs: nullcontext()))
    runner.groups["dp"].graph_capture = capture
    runner._issue = lambda call: lambda: events.append("issue")
    issue, run = runner.prepare(_call(runner.parallel, captured=True))
    run()
    assert events == ["issue", "dp_capture", "issue", "issue", "replay"]


@pytest.mark.parametrize("engine", [{}, {"disable_custom_all_reduce": False},
                                    {"disable_custom_all_reduce": True}])
def test_native_launch_flag_and_manifest_preserve_platform_nccl(monkeypatch, engine):
    parallel = _parallel(tp=2, dp=2)
    groups = _runner(parallel).groups
    for group in groups.values():
        group.device_communicator = types.SimpleNamespace(
            pynccl_comm=types.SimpleNamespace(disabled=False, nccl_version=23007), ca_comm=None,
            fi_ar_comm=types.SimpleNamespace(disabled=False))
    calls = []
    env_values = {"VLLM_ALLREDUCE_USE_FLASHINFER": True,
                  "VLLM_ALLREDUCE_USE_SYMM_MEM": True, "VLLM_USE_NCCL_SYMM_MEM": False}
    envs = types.SimpleNamespace(environment_variables=dict.fromkeys(env_values), **env_values)
    monkeypatch.delenv("VLLM_ALLREDUCE_USE_FLASHINFER", raising=False)
    monkeypatch.delenv("VLLM_USE_NCCL_SYMM_MEM", raising=False)
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_DEBUG_SUBSYS", "INIT,NET")
    monkeypatch.setenv("NCCL_IB_HCA", "platform_hca")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    ps = types.SimpleNamespace(get_tp_group=lambda: groups["tp"], get_dp_group=lambda: groups["dp"],
                               get_ep_group=lambda: groups["ep"])
    torch = types.SimpleNamespace(
        __version__="test", device=lambda *args: "cuda:0", zeros=lambda *args, **kwargs: None,
        cuda=types.SimpleNamespace(set_device=lambda device: None, synchronize=lambda: None),
        distributed=types.SimpleNamespace(
            new_group=lambda *args, **kwargs: "instance_cpu", all_reduce=lambda *args, **kwargs: None,
            all_gather_object=lambda result, host, **kwargs: result.__setitem__(slice(None), [host] * 4)))

    def worker_init(config, rank, method, local_rank):
        # Native gpu_worker.init_worker_distributed_environment performs this
        # before constructing the process groups.
        calls.append((not config.parallel_config.disable_custom_all_reduce, rank, method, local_rank))

    modules = {
        "torch": torch, "vllm": types.SimpleNamespace(__version__="image-test", envs=envs),
        "vllm.config": types.SimpleNamespace(
            ParallelConfig=lambda **kwargs: types.SimpleNamespace(
                **kwargs, disable_nccl_for_dp_synchronization=True),
            VllmConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            set_current_vllm_config=lambda config: nullcontext()),
        "vllm.distributed": types.SimpleNamespace(parallel_state=ps),
        "vllm.v1.worker.gpu_worker": types.SimpleNamespace(init_worker_distributed_environment=worker_init),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(CollectiveRunner, "_sleep_cycles_per_us", lambda self: 1.0)
    import os

    nccl_before = {name: value for name, value in os.environ.items() if name.startswith("NCCL_")}
    runner = CollectiveRunner(parallel, {}, engine=engine)
    assert calls == [(not engine.get("disable_custom_all_reduce", False), 0, "env://", 0)]
    runtime = runner.communicators()
    assert runtime["nccl_environment"] == nccl_before
    assert {name: value for name, value in os.environ.items() if name.startswith("NCCL_")} == nccl_before
    assert runtime["vllm_environment"]["VLLM_ALLREDUCE_USE_FLASHINFER"] == {
        "value": True, "environment_override": None, "source": "image_default"}
    assert runtime["vllm_environment"]["VLLM_ALLREDUCE_USE_SYMM_MEM"]["source"] == "environment"
    assert runtime["vllm_environment"]["VLLM_USE_NCCL_SYMM_MEM"]["value"] is False
    assert runtime["enabled_communicators"]["tp"]["fi_ar_comm"] is True
    assert runtime["pynccl_versions"] == {"tp": 23007, "ep": 23007, "dp": 23007}
    assert runtime["nccl_log_destination"] == os.environ.get("NCCL_DEBUG_FILE", "stderr (stage wrapper log)")


@pytest.mark.parametrize("isolated", [False, True])
def test_step_dry_run_reports_native_and_isolated_rails_without_cuda(monkeypatch, tmp_path, capsys, isolated):
    config = {
        "engine": {"tensor_parallel_size": 8, "data_parallel_size": 4,
                   "pipeline_parallel_size": 1, "enable_expert_parallel": True,
                   "dtype": "bfloat16", "async_scheduling": False,
                   "disable_custom_all_reduce": True},
        "model_config": str(Path(__file__).resolve().parents[2] /
                            "data/config/models/stepfun-ai__Step-5-Preview.json"),
        "grid": {"cuda_graph_batch_sizes": [8]},
        "warmup_calls": 1, "repeats": 3, "calls_per_repeat": 2,
        "passes": 3, "shuffle_seed": 7,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = ["benchmark", "--config", str(path), "--list-cases"]
    if isolated:
        args.append("--dp-rail-isolation")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(CollectiveRunner, "__init__", lambda *args, **kwargs: pytest.fail("CUDA initialized"))
    collective_main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "cases 10 passes 3 measurements 30"
    dispatch = next(line for line in lines if "moe_dispatch_all_gatherv_fp8" in line)
    assert "(34304, 137216) 85" in dispatch
    assert dispatch.endswith("dp_rails (0,)" if isolated else "dp_rails (0, 1, 2, 3, 4, 5, 6, 7)")
    host = next(line for line in lines if "dp_batch_sync_all_reduce" in line)
    assert "cpu_all_reduce dp (96, 96) 1" in host and host.endswith("dp_rails (0,)")


def test_pass_order_is_a_seeded_permutation():
    items = list(range(20))
    first = pass_order(items, 7, 0)
    assert first == pass_order(items, 7, 0) and sorted(first) == items
    assert first != pass_order(items, 7, 1) and first != items
    assert pass_order(items, 6, 1) == pass_order(items, 7, 0)


@pytest.mark.parametrize("native_error",[False,True,None])
def test_collector_identity_fault_is_soft_and_native_failure_is_original(tmp_path,monkeypatch,native_error):
    from frontier.profiling.collectives import vllm_collective_benchmark as benchmark
    from frontier.profiling.common import runtime_identity
    definition=json.loads((Path(__file__).parents[1]/"integration/step5_comm/step5_position_model_run.json").read_text())
    definition.update(model_config=str(Path(__file__).parents[2]/"data/config/models/stepfun-ai__Step-5-Preview.json"),
        output_dir=str(tmp_path/"output"),profile_manifest="cpu_fixture",passes=1)
    definition["grid"]=dict(scenarios=definition["grid"]["scenarios"][:1],native_call_names=["moe_combine_reduce_scatterv"])
    if native_error is None: definition.pop("profile_manifest")
    config=tmp_path/"config.json";config.write_text(json.dumps(definition))
    monkeypatch.setenv("FRONTIER_STEP5_PROFILE_MANIFEST","cpu_fixture")
    for k,v in definition["native_environment"].items(): monkeypatch.setenv(k,v)
    monkeypatch.setattr(runtime_identity,"verify_step5_profile_runtime",lambda *a,**kw:(_ for _ in ()).throw(ValueError("bad_source")))
    error=RuntimeError("native_collective_failed");calls=[]
    class Runner:
        rank=0;instance_cpu=None
        groups={"dp":types.SimpleNamespace(all_gatherv=lambda:None,reduce_scatterv_into_output=lambda:None,all_reduce=lambda:None)}
        torch=types.SimpleNamespace(distributed=types.SimpleNamespace(barrier=lambda **kw:None))
        def __init__(self,*a,**kw): pass
        def communicators(self): return {"evidence_kind":"cpu_fixture"}
        def gather(self,value): return [value,["rank8_bad_source"]]
        def measure(self,s,c):
            calls.append(c.name)
            if native_error: raise error
            return None
        def kernel_names(self,*a):
            self._last_kernel_evidence={};return []
    monkeypatch.setattr(benchmark,"CollectiveRunner",Runner)
    monkeypatch.setattr(sys,"argv",["benchmark","--config",str(config)])
    if native_error:
        with pytest.raises(RuntimeError) as caught: benchmark.main()
        assert caught.value is error
    else:
        benchmark.main()
        if native_error is None:
            assert set(json.loads((tmp_path/"output/plan.json").read_text())) == {"config","cases","runtime"}
            assert not (tmp_path/"output/native_identity.json").exists()
        else:
            result=json.loads((tmp_path/"output/native_identity.json").read_text())
            assert result["status"] == "UNVERIFIED" and "bad_source" in result["errors"][0]
            assert "rank8_bad_source" in result["errors"]
    assert calls == ["moe_combine_reduce_scatterv"]


def test_network_rows_keep_eager_single_node_tp_all_reduce():
    rows = [summarize(_record(nbytes=4096)),
            summarize(_record(nbytes=2048, timings=[[1.0, 1.0, 1.0]])),
            summarize(_record(graph=True)),
            summarize(_record(op="all_gather")),
            summarize(_record(group="dp"))]
    small, large = network_all_reduce_rows(rows, gpus_per_node=8)
    assert small["size"] == 2048 and small["time_stats.all_reduce.median"] == 1e-3
    assert large["size"] == 4096 and large["time_stats.all_reduce.median"] == 3e-3
    assert small["num_workers"] == small["devices_per_node"] == 8
    assert network_all_reduce_rows(rows, gpus_per_node=4) == []


def test_layout_checks_the_cache_contract():
    layout = KVRegionLayout.from_config(KV_CACHE)
    assert layout.num_regions == 5 and layout.block_bytes == 3720
    with pytest.raises(ValueError, match="cache contract"):
        KVRegionLayout.from_config({**KV_CACHE, "resident_bytes_per_block": 1})


def test_descriptors_are_region_major():
    layout = KVRegionLayout.from_config(KV_CACHE)
    assert descriptor_ids(5, 10, [2, 3]) == [2, 3, 12, 13, 22, 23, 32, 33, 42, 43]
    descs = block_descriptors([0, 100_000, 200_000, 300_000, 400_000], layout, 3)
    assert len(descs) == 50
    assert descs[12] == (100_000 + 2 * 1152, 1152, 3)
    assert descs[41] == (400_000 + 132, 132, 3)


def test_cases_cover_grid_and_respect_capacity():
    layout = KVRegionLayout.from_config(KV_CACHE)
    cases = read_cases([{"prompt_tokens": [65, 128], "active_pairs": [1, 8],
                         "requests_in_flight": [1, 2]},
                        {"prompt_tokens": [320], "active_pairs": [8],
                         "requests_in_flight": [2]}], layout, 8)
    assert len(cases) == 9 and cases[0].blocks_per_request == 2
    assert request_block_ids(cases[1]) == [[0, 1], [2, 3]]
    assert cases[-1].name == "t320_pairs8_inflight2" and cases[-1].blocks_per_request == 5
    with pytest.raises(ValueError, match="exceed"):
        read_cases([{"prompt_tokens": [640], "active_pairs": [1],
                     "requests_in_flight": [2]}], layout, 8)
    with pytest.raises(ValueError, match="active_pairs"):
        read_cases([{"prompt_tokens": [64], "active_pairs": [9],
                     "requests_in_flight": [1]}], layout, 8)
    with pytest.raises(ValueError, match="repeat a case"):
        read_cases([{"prompt_tokens": [64], "active_pairs": [1], "requests_in_flight": [1]}] * 2,
                   layout, 8)


def test_case_summary_uses_slowest_reader():
    case = ReadCase("c", 128, 2, 2, 1)
    record = {"pass": 2, "case": case.name, "prompt_tokens": 128, "blocks_per_request": 2,
              "active_pairs": 2, "requests_in_flight": 1, "descriptors_per_request": 10,
              "bytes_per_reader": 1000,
              "readers": [{"timings": [{"complete_us": 10.0, "prep_us": 1.0}] * 3,
                           "data_verified": True},
                          {"timings": [{"complete_us": 20.0, "prep_us": 2.0}] * 3,
                           "data_verified": True}],
              "notifications_received": 6, "notifications_expected": 6}
    row = summarize_case(record)
    assert row["complete_median_us"] == 20.0 and row["prep_median_us"] == 2.0
    assert row["per_reader_gigabytes_per_s"] == pytest.approx(0.05)
    assert row["aggregate_gigabytes_per_s"] == pytest.approx(0.1)
    assert row["data_verified"] and row["pass"] == 2


def test_overlap_plan_holds_only_ep_calls():
    model = ModelShape.from_hf_config({
        "n_routed_experts": 8, "hidden_size": 16, "num_hidden_layers": 4,
        "first_k_dense_replace": 1, "num_experts_per_tok": 2, "vocab_size": 128})
    parallel = ServingParallelism.from_engine({
        "tensor_parallel_size": 2, "data_parallel_size": 2, "pipeline_parallel_size": 1,
        "enable_expert_parallel": True, "dtype": "bfloat16", "async_scheduling": True})
    plan = ep_plan(model, parallel, {"eager": [], "cuda_graph_batch_sizes": [2, 4]})
    assert [(s.name, c.op) for s, c in plan] == [
        ("graph_decode_b2", "all_gatherv"), ("graph_decode_b2", "reduce_scatterv"),
        ("graph_decode_b4", "all_gatherv"), ("graph_decode_b4", "reduce_scatterv")]


def test_overlap_read_and_load_sizes():
    layout = KVRegionLayout.from_config(KV_CACHE)
    read = overlap_read({"read": {"prompt_tokens": 128, "requests_in_flight": 2}}, layout, 4)
    assert read == ReadCase("t128_pairs4_inflight2", 128, 2, 4, 2)
    assert expected_notifications(read, {"read_warmup": 2, "repeats": 5}) == 2 * (2 + 3 * 5)
    assert load_runs(1000.0, 300.0, 3.0) == 10
    assert load_runs(10.0, 300.0, 3.0) == 1


class _Event:
    """Completes on its second ``query``."""

    def __init__(self):
        self.queries = 0

    def record(self):
        pass

    def query(self):
        self.queries += 1
        return self.queries >= 2


def test_issue_load_reports_whether_polled_work_ended_first():
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=_Event))
    runs = []
    polls = iter([False] * 3 + [True] * 100)
    assert issue_load(torch, lambda: runs.append(1), 5, 2, lambda: next(polls))
    assert len(runs) == 5
    assert not issue_load(torch, lambda: None, 5, 2, lambda: False)


def test_overlap_summary_uses_slowest_d_rank():
    def rank(scale):
        return {"ep_alone_call_us": [10.0 * scale] * 3, "ep_loaded_call_us": [15.0 * scale] * 3,
                "ep_alone_queue_held": [True] * 3, "ep_loaded_queue_held": [True, True, False],
                "ep_loaded_covered": [True] * 3, "ep_load_runs": 7,
                "read_alone": [{"complete_us": 100.0 * scale}] * 3,
                "read_loaded": [{"complete_us": 150.0 * scale}] * 3,
                "read_loaded_covered": [True, False, True], "data_verified": True}
    record = {"pass": 0, "scenario": "graph_decode_b8", "call": "dispatch", "op": "all_gatherv",
              "cuda_graph": True, "tokens_per_dp_rank": [8, 8], "max_input_bytes": 1,
              "max_output_bytes": 2, "read_case": "r", "bytes_per_reader": 2000,
              "per_rank": [rank(1), rank(2)], "notifications_received": 4,
              "notifications_expected": 4}
    row = summarize_overlap(record)
    assert (row["ep_alone_median_us"], row["ep_loaded_median_us"]) == (20.0, 30.0)
    assert row["ep_loaded_over_alone"] == pytest.approx(1.5)
    assert row["read_loaded_over_alone"] == pytest.approx(1.5)
    assert row["read_alone_per_reader_gigabytes_per_s"] == pytest.approx(0.01)
    assert row["ep_loaded_queue_held_fraction"] == pytest.approx(2 / 3)
    assert row["read_loaded_covered_fraction"] == pytest.approx(2 / 3)
    assert row["ep_load_runs"] == 7 and row["data_verified"]
