"""CPU evidence/exception controls; synthetic profiles are not GPU measurements."""

from contextlib import nullcontext
import json
from types import ModuleType, SimpleNamespace as NS
import sys

import pytest

from frontier.profiling.collectives.vllm_collective_benchmark import CollectiveRunner


@pytest.mark.parametrize("failure", [None, "native", "export"])
def test_rank_root_counts_and_trace_preserve_native_calls(monkeypatch, tmp_path, failure):
    calls, ranges = [], []
    result = object()

    class Library:
        def ncclReduce(self, sendbuff, recvbuff, count, datatype, op, root, comm, stream):
            calls.append((sendbuff, recvbuff, count, datatype, op, root, comm, stream))
            if failure == "native":
                raise RuntimeError("native Reduce failure")
            return result

    original = Library.ncclReduce
    module = ModuleType("vllm.distributed.device_communicators.pynccl_wrapper")
    module.NCCLLibrary = Library
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Profile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def events(self):
            return [NS(name="ncclDevKernel_Reduce_Sum_bf16", device_type="CUDA")]

        def export_chrome_trace(self, path):
            if failure == "export":
                raise OSError("trace export failure")
            with open(path, "w") as handle:
                json.dump({"traceEvents": [], "synthetic_cpu_fixture": True}, handle)

    def record_function(name):
        ranges.append(name)
        return nullcontext()

    runner = object.__new__(CollectiveRunner)
    runner.rank, runner.instance_cpu = 3, object()
    runner.torch = NS(distributed=NS(barrier=lambda **kw: None),
        profiler=NS(profile=lambda **kw: Profile(), record_function=record_function,
                    ProfilerActivity=NS(CPU="CPU", CUDA="CUDA")),
        cuda=NS(synchronize=lambda: None), autograd=NS(DeviceType=NS(CUDA="CUDA")))
    runner.participates = lambda _: True
    observed = []
    runner.gather = lambda local: observed.append(local) or [local]
    library = Library()
    sent, received, comm, stream = object(), object(), object(), object()
    sizes = [768 * 6144] * 8 + [6144] * 8

    def run():
        for root, count in enumerate(sizes):
            assert library.ncclReduce(sent, received, count, 9, 0, root, comm, stream) is result

    runner.prepare = lambda _: (object(), run)
    if failure:
        with pytest.raises((RuntimeError, OSError), match="failure"):
            runner.kernel_evidence(NS(op="reduce_scatterv"), tmp_path)
        assert not observed
    else:
        evidence = runner.kernel_evidence(NS(op="reduce_scatterv"), tmp_path)
        row, = evidence["per_rank"]
        assert row["rank"] == 3
        assert [r["root"] for r in row["reduce_calls"]] == list(range(16))
        assert [r["count"] for r in row["reduce_calls"]] == sizes
        assert sum(r["count"] for r in row["reduce_calls"]) * 2 == 75_595_776
        assert ranges == [f"ncclReduce_root_{root}" for root in range(16)]
        assert evidence["kernels"] == ["ncclDevKernel_Reduce_Sum_bf16"]
        assert json.loads((tmp_path / "rank3.json").read_text())["synthetic_cpu_fixture"]
        assert all(c[0] is sent and c[1] is received and c[-2] is comm and c[-1] is stream
                   for c in calls)
    assert Library.ncclReduce is original
