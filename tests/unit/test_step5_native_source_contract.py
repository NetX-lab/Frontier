"""Check the exported locked native source without importing GPU dependencies.

Run with STEP5_NATIVE_SOURCE_ROOT pointing to the exported vllm package. These
checks exercise native selection/replay methods on CPU and inspect their wiring;
they do not claim that a CUDA graph has executed on an H200.
"""

import ast
from collections import Counter, namedtuple
from collections.abc import Callable
import copy
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.step5_pd import canonical_stage


@pytest.fixture(scope="module")
def native_source():
    root = os.environ.get("STEP5_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Set STEP5_NATIVE_SOURCE_ROOT to the locked exported vllm package")
    directory = Path(root)
    assert directory.is_dir(), f"Native source directory is missing: {directory}"
    return directory


def source_method(directory, relative, name):
    tree = ast.parse((directory / relative).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def compile_methods(methods, namespace):
    tree = ast.Module(body=[ast.ClassDef(name="NativeClass", bases=[], keywords=[],
        body=copy.deepcopy(methods), decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<locked-native-methods>", "exec"), namespace)
    return namespace["NativeClass"]


def test_bindings_cover_selected_mrv2_dispatch_dp_replay_and_observations(native_source):
    required = {
        "config/vllm.py", "v1/worker/gpu_worker.py", "v1/worker/gpu/model_runner.py",
        "v1/worker/gpu/dp_utils.py", "v1/worker/dp_utils.py", "v1/worker/gpu/cudagraph_utils.py",
        "v1/metrics/loggers.py", "compilation/cuda_graph.py",
    }
    assert required <= set(canonical_stage.RUNTIME_SOURCES)
    assert "v1/worker/gpu_model_runner.py" not in canonical_stage.RUNTIME_SOURCES
    assert "v1/cudagraph_dispatcher.py" not in canonical_stage.RUNTIME_SOURCES
    assert all((native_source / relative).is_file() for relative in canonical_stage.RUNTIME_SOURCES)


@pytest.mark.parametrize("override,triton,backend,pp,expected", [
    (None, True, "mp", 1, True),
    (False, True, "mp", 1, False),
    (None, False, "mp", 1, False),
    (None, True, "external_launcher", 2, False),
])
def test_locked_native_runner_selection_and_its_runtime_prerequisites(native_source, override, triton, backend, pp, expected):
    relative = "config/vllm.py"
    methods = [source_method(native_source, relative, name) for name in (
        "use_v2_model_runner", "_get_v2_model_runner_unsupported_features")]
    native = compile_methods(methods, {
        "envs": SimpleNamespace(VLLM_USE_V2_MODEL_RUNNER=override), "HAS_TRITON": triton,
        "logger": SimpleNamespace(warning_once=lambda *args: None),
    })()
    native.attention_config = SimpleNamespace(hisparse_config=None)
    native.speculative_config = None
    native.parallel_config = SimpleNamespace(distributed_executor_backend=backend,
                                             pipeline_parallel_size=pp, use_ubatching=False)
    assert native.use_v2_model_runner is expected
    worker = source_method(native_source, "v1/worker/gpu_worker.py", "_make_model_runner")
    v2_branch = worker.body[1]
    assert ast.unparse(v2_branch.test) == "self.use_v2_model_runner"
    assert any(isinstance(node, ast.ImportFrom) and node.module == "vllm.v1.worker.gpu.model_runner"
               for node in ast.walk(v2_branch))


def test_mrv2_uses_post_dp_descriptor_for_real_batch_stats_and_full_replay(native_source):
    execute = source_method(native_source, "v1/worker/gpu/model_runner.py", "execute_model")
    dispatch = next(node for node in ast.walk(execute) if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "dispatch_cg_and_sync_dp")
    assert ast.unparse(dispatch.targets[0]) == "(batch_desc, dp_sync_state)"
    full = next(node for node in ast.walk(execute) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "batch_desc.cg_mode == CUDAGraphMode.FULL"
                and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "self.cudagraph_manager.run_fullgraph"
                        for child in node.body for call in ast.walk(child)))
    replay = next(node for node in ast.walk(full) if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "self.cudagraph_manager.run_fullgraph")
    assert ast.unparse(replay.args[0]) == "batch_desc" and dispatch.lineno < full.lineno
    stats = next(node for node in ast.walk(execute) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "make_cudagraph_stats")
    assert [ast.unparse(arg) for arg in stats.args] == ["batch_desc", "num_toks"]
    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "not dummy_run"
               and any(child is stats for part in node.body for child in ast.walk(part))
               for node in ast.walk(execute))
    dp = source_method(native_source, "v1/worker/gpu/dp_utils.py", "dispatch_cg_and_sync_dp")
    assert any(isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
               and ast.unparse(node.value.func) == "sync_cudagraph_and_dp_padding" for node in ast.walk(dp))
    sync = source_method(native_source, "v1/worker/gpu/dp_utils.py", "sync_cudagraph_and_dp_padding")
    assert any(isinstance(node, ast.Assign) and any(ast.unparse(target) == "synced_desc" for target in node.targets)
               and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "cudagraph_manager.dispatch"
               for node in ast.walk(sync))


def test_native_mrv2_fullgraph_route_calls_replay_not_only_capture(native_source):
    method = source_method(native_source, "v1/worker/gpu/cudagraph_utils.py", "run_fullgraph")
    calls = []
    mode = SimpleNamespace(FULL=object())
    native = compile_methods([method], {"BatchExecutionDescriptor": object,
        "CUDAGraphMode": mode,
        "get_offloader": lambda: SimpleNamespace(sync_prev_onload=lambda: calls.append("sync"))})()
    descriptor = namedtuple("Descriptor", "cg_mode")(mode.FULL)
    native.graphs = {descriptor: SimpleNamespace(replay=lambda: calls.append("replay"))}
    assert not calls  # A captured/cache entry alone is not evidence of replay.
    native.run_fullgraph(descriptor)
    assert calls == ["sync", "replay"]


def test_mrv2_stats_reach_native_scheduler_and_logging(native_source):
    execute = source_method(native_source, "v1/worker/gpu/model_runner.py", "execute_model")
    sample = source_method(native_source, "v1/worker/gpu/model_runner.py", "sample_tokens")
    for method, constructor in ((execute, "ExecuteModelState"), (sample, "ModelRunnerOutput")):
        assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == constructor
                   and any(kw.arg == "cudagraph_stats" and ast.unparse(kw.value) == "cudagraph_stats"
                           for kw in node.keywords) for node in ast.walk(method))
    scheduler = source_method(native_source, "v1/core/sched/scheduler.py", "make_stats")
    assert scheduler.args.args[3].arg == "cudagraph_stats"  # Includes self.
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == "SchedulerStats"
               and any(kw.arg == "cudagraph_stats" and ast.unparse(kw.value) == "cudagraph_stats"
                       for kw in node.keywords) for node in ast.walk(scheduler))
    tree = ast.parse((native_source / "v1/metrics/loggers.py").read_text())
    logger = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LoggingStatLogger")
    observed = next(node for node in ast.walk(logger) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "self.cudagraph_logging.observe")
    assert ast.unparse(observed.args[0]) == "scheduler_stats.cudagraph_stats"
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == "self.cudagraph_logging.log"
               for node in ast.walk(logger))


def test_actual_native_mrv2_stat_and_table_report_runtime_modes_and_counts(native_source):
    tree = ast.parse((native_source / "compilation/cuda_graph.py").read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("CUDAGraphStat", "CUDAGraphLogging")]
    make_stat = source_method(native_source, "v1/worker/gpu/cudagraph_utils.py", "make_cudagraph_stats")
    emitted = []
    namespace = {"__name__": __name__, "dataclasses": dataclasses, "Counter": Counter,
                 "CUDAGraphMode": str, "BatchExecutionDescriptor": object, "Callable": Callable,
                 "Any": Any, "logger": SimpleNamespace(info=emitted.append)}
    module = ast.Module(body=copy.deepcopy(classes + [make_stat]), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<locked-native-metrics>", "exec"), namespace)
    stat = namespace["make_cudagraph_stats"](SimpleNamespace(num_tokens=4, cg_mode="FULL"), 3)
    assert dataclasses.astuple(stat) == (3, 4, 1, "FULL")
    logging = namespace["CUDAGraphLogging"]("FULL_AND_PIECEWISE", [1, 2, 4])
    logging.observe(stat)
    logging.observe(stat)
    logging.observe(namespace["make_cudagraph_stats"](SimpleNamespace(num_tokens=6144, cg_mode="NONE"), 6144))
    logging.log()
    assert len(emitted) == 1 and "**CUDAGraph Config Settings:**" in emitted[0]
    rows = [[cell.strip() for cell in line.strip("|").split("|")]
            for line in emitted[0].splitlines() if line.startswith("|")]
    assert rows[0] == ["Unpadded Tokens", "Padded Tokens", "Num Paddings", "Runtime Mode", "Count"]
    assert rows[2:] == [["3", "4", "1", "FULL", "2"], ["6144", "6144", "0", "NONE", "1"]]
    logging.log()
    assert len(emitted) == 1  # The native logger resets each emitted interval.
