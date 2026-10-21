"""CPU contracts for native Step attention profiling; no timing claims."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.step5 import resolve_step5_attention_geometry
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.step5.inputs import Step5ProfileInput, build_profile_inputs
from frontier.profiling.step5.rows import OPERATOR_NAMES, Step5ProfileResult, build_row, rows_to_dataframe, summarize_replays
from frontier.profiling.step5.scopes import NativeScopes, trace_signature
from frontier.profiling.step5.job import describe_smoke, IMAGE, CLOUD_MOUNT


@pytest.fixture(scope="module")
def config():
    return ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")


@pytest.mark.parametrize("layer,tokens,prefix", list(product((0, 3), (7, 8, 9, 15, 16, 17, 511, 512, 513), (0, 4096, 4104))))
def test_native_shape_boundaries(config, layer, tokens, prefix):
    cell = Step5ProfileInput(layer, (tokens,), (prefix,), "prefill")
    workload = cell.workload(config, block_size=16, max_model_len=8192)
    geometry = resolve_step5_attention_geometry(config, layer, tensor_parallel_size=8)
    assert workload.valid_tokens == tokens
    assert workload.sequence_lengths == (prefix + tokens,)
    assert geometry.local_q_heads == 8
    assert geometry.local_kv_heads == 1
    assert geometry.kv_replication == 2
    if layer == 3:
        # The native provider-sharded indexer replicates four Q heads per
        # provider, not the naive global index_heads / TP = two heads.
        assert geometry.local_index_q_heads == 4
        assert geometry.local_fused_indexer_width == 1544


def test_smoke_grid_and_shared_cpu_runtime(config):
    cells = build_profile_inputs(config, prefill_tokens=[16, 64], prefixes=[0, 4104],
                                decode_batches=[1], decode_contexts=[512, 4104])
    assert len(cells) == 12
    assert {c.global_layer_id for c in cells} == {0, 3}
    assert {c.graph_mode for c in cells if c.phase == "prefill"} == {"none"}
    assert {c.graph_mode for c in cells if c.phase == "decode"} == {"full"}
    assert STEP5_ATTENTION_FAMILY.execution_enabled is True
    assert config.get_model_architecture_profile().attention_linear_ops.profiling_enabled is True


@pytest.mark.parametrize("kwargs", [
    {"query_lens": (0,)}, {"context_lens": (-1,)}, {"context_lens": ()},
    {"phase": "mixed"}, {"phase": "decode"}, {"graph_mode": "invalid"},
    {"graph_mode": "full"}, {"global_layer_id": True},
])
def test_invalid_cells(kwargs):
    values = dict(global_layer_id=0, query_lens=(16,), context_lens=(0,), phase="prefill")
    values.update(kwargs)
    with pytest.raises(ValueError):
        Step5ProfileInput(**values)


def fake_stats():
    return {name: dict(min=1., max=1., mean=1., median=1., std=0., count=10)
            for name in (*OPERATOR_NAMES, "step5_layer_e2e")}


def row(config):
    return build_row(config, Step5ProfileInput(3, (16,), (4104,), "prefill"),
                     tp=8, block_size=16, max_model_len=8192,
                     runtime_signature="cpu_schema_fixture_not_measurement", device="h800",
                     time_stats=fake_stats(), evidence={"attention_backend": "native_fixture"})


def test_typed_row_has_complete_family_schema(config):
    frame = rows_to_dataframe([row(config)])
    assert set(STEP5_ATTENTION_FAMILY.required_profiling_feature_columns) <= set(frame)
    contracts = json.loads(frame.iloc[0].typed_operator_contracts)
    assert set(contracts) == set(OPERATOR_NAMES)
    assert {c["selected_tensor_parallel_size"] for c in contracts.values()} == {8}
    assert {c["operator_family_id"] for c in contracts.values()} == {"step5_attention"}
    assert frame.iloc[0].local_index_q_heads == 4
    assert frame.iloc[0].has_attention_metadata == 1
    assert STEP5_ATTENTION_FAMILY.execution_enabled is True
    assert frame.iloc[0].profiling_precision == "BF16"
    assert frame.iloc[0].model_arch == config.model_arch


def test_gpu_active_rows_use_separate_kernel_only_measurement(config):
    result = row(config)
    result["measurement_type"] = "KERNEL_ONLY"
    with pytest.raises(ValueError, match="measurement_type mismatch"):
        rows_to_dataframe([result])
    frame = rows_to_dataframe([result], measurement_type="KERNEL_ONLY")
    assert set(frame.measurement_type) == {"KERNEL_ONLY"}


@pytest.mark.parametrize("failure", ["operator_cv", "median_disagreement", "zero_work"])
def test_timing_quality_checks_all_active_targets_without_correction(failure):
    window = {"step5_qkv_projection": [1., 1.], "step5_layer_e2e": [3., 3.]}
    raw = {name: values[:] for name, values in window.items()}
    if failure == "operator_cv":
        window["step5_qkv_projection"] = [.8, 1.2]
    elif failure == "median_disagreement":
        window["step5_qkv_projection"] = [1.1, 1.1]
    else:
        window["step5_qkv_projection"] = [0., 0.]
        with pytest.raises(ValueError, match="positive measured work"):
            summarize_replays(window, raw, ("step5_qkv_projection",), max_time_cv=.05)
        return
    stats, quality = summarize_replays(window, raw, ("step5_qkv_projection",), max_time_cv=.05)
    assert quality["time_cv"] == 0.
    assert not quality["stability_pass"]
    assert stats["step5_qkv_projection"]["mean"] == pytest.approx(sum(window["step5_qkv_projection"]) / 2)


def test_timing_quality_retains_single_replay_stats():
    values = {"step5_qkv_projection": [1., 1.], "step5_layer_e2e": [3., 3.]}
    _, quality = summarize_replays(values, values, ("step5_qkv_projection",), max_time_cv=.05)
    assert quality["stability_pass"]
    assert quality["max_median_relative_error"] == 0.
    assert json.loads(quality["per_iteration_time_stats"])["step5_layer_e2e"]["count"] == 2


def test_independent_layer_reference_is_not_labelled_as_per_iteration_work():
    candidates = {"step5_qkv_projection": [1., 1.], "step5_layer_e2e": [3., 3.]}
    raw = {"step5_qkv_projection": [1., 1., 1., 1.]}
    _, quality = summarize_replays(candidates, raw, ("step5_qkv_projection",),
                                  max_time_cv=.05, layer_reference_times=[3.1, 3.1])
    assert quality["stability_pass"]
    assert "step5_layer_e2e" not in json.loads(quality["per_iteration_time_stats"])
    assert json.loads(quality["layer_reference_time_stats"])["count"] == 2
    assert json.loads(quality["median_relative_error_by_target"])["step5_layer_e2e"] == pytest.approx(.1 / 3.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_timing_rejected(config, bad):
    result = row(config)
    result["time_stats"]["step5_qkv_projection"]["mean"] = bad
    with pytest.raises(ValueError):
        rows_to_dataframe([result])


def test_missing_active_timing_is_not_zero_filled(config):
    result = row(config)
    del result["time_stats"]["step5_sparse_attention"]
    with pytest.raises(ValueError, match="missing required"):
        rows_to_dataframe([result])


def test_trace_includes_device_copies(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "name": "_sparse_decode_split", "dur": 4},
        {"cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 2},
    ]}))
    result = trace_signature(trace)
    assert result["kernel_count"] == 1
    assert result["memory_copy_count"] == 1
    assert result["memory_copy_time_ms"] == .002


def test_smoke_description_uses_locked_image_and_personal_cloud_root(tmp_path):
    job = describe_smoke(tmp_path / "stage", "attention_smoke_cpu_fixture")
    assert job["image"] == IMAGE
    assert IMAGE.endswith("sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
    assert CLOUD_MOUNT in job["mounts"]
    assert job["charged_group"] == "codesign" and job["gpu_tag"] == "H800"
    assert job["cloud_output"].startswith("/mnt/codesign-exp/ycfeng/")
    assert job["expected_rows"] == 12
    assert job["expected_gpu_active_rows"] == 8
    assert job["gpus"] == 1
    assert "--tensor-parallel-size 8" in job["worker_command"]
    assert "OFFLINE_DEPENDENCIES_READY" in job["worker_command"]
    assert "submit_profiling_job.py" in job["submission_command"]


@pytest.mark.parametrize("run_id", ["", "../escape", "/absolute", ".", ".."])
def test_smoke_rejects_cloud_path_escape(tmp_path, run_id):
    with pytest.raises(ValueError):
        describe_smoke(tmp_path, run_id)


def test_native_metadata_preserves_request_page_ids_between_prefix_and_continuation(monkeypatch):
    torch = pytest.importorskip("torch")
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *a, **k: arange(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    backend = ModuleType("vllm.v1.attention.backend")
    backend.CommonAttentionMetadata = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.block_size, wrapper.max_model_len = torch, 16, 8192
    wrapper.native = object()
    wrapper._set_current = lambda _: nullcontext()
    wrapper.builder = SimpleNamespace(build=lambda _, common, **kw: common)
    prefix = Step5ProfileInput(3, (512, 512), (0, 0), "prefill")
    current = Step5ProfileInput(3, (16, 16), (512, 512), "prefill")
    _, _, p, _ = wrapper._metadata(prefix)
    positions, slots, c, _ = wrapper._metadata(current)
    assert p.block_table_tensor[1, 0] == c.block_table_tensor[1, 0] == 513
    assert c.query_start_loc_cpu.tolist() == [0, 16, 32]
    assert c.seq_lens_cpu_upper_bound.tolist() == [528, 528]
    assert positions.tolist() == list(range(512, 528)) * 2
    assert slots.tolist() == list(range(528, 544)) + list(range(8720, 8736))


def test_nested_native_scopes_charge_only_the_outer_fused_owner():
    events = []

    class Event:
        def __init__(self, **kwargs):
            events.append(kwargs)

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.

    fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                           profiler=SimpleNamespace(record_function=lambda _: nullcontext()))
    scopes = NativeScopes(fake)
    module = SimpleNamespace(inner=lambda: 3)
    module.outer = lambda: module.inner()
    with scopes.observe([(module, "outer", "fused"), (module, "inner", "indexer")]):
        assert module.outer() == 3
    assert scopes.times() == {"fused": 2.}
    assert len(events) == 2
    assert not scopes.enabled


def test_gpu_activity_annotations_do_not_insert_timing_events():
    names = []

    def event(**_):
        raise AssertionError("GPU-active trace must not contain instrumentation event nodes")

    @contextmanager
    def record(name):
        names.append(name)
        yield

    scopes = NativeScopes(SimpleNamespace(cuda=SimpleNamespace(Event=event),
        profiler=SimpleNamespace(record_function=record)))
    target = SimpleNamespace(forward=lambda: "native output")
    with scopes.observe([(target, "forward", "projection")], record_events=False):
        assert target.forward() == "native output"
    assert names == ["projection"]
    assert scopes.times() == {}
    assert scopes.record_events


def test_eager_activity_windows_use_gpu_work_not_restore_or_host_gaps(tmp_path, monkeypatch):
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    cpu, gpu, events = [0.], [0.], []
    correlation = [0]
    output = object()

    @contextmanager
    def annotation(name):
        start = cpu[0]
        cpu[0] += 1
        yield
        cpu[0] += 1
        events.append(dict(cat="user_annotation", name=name, ts=start,
                           dur=cpu[0] - start, pid=1, tid=1))

    def activity(category, name, duration):
        correlation[0] += 1
        cpu[0] += 1
        events.append(dict(cat="cuda_runtime", name="cudaLaunchKernel" if category == "kernel"
                           else "cudaMemcpyAsync", ts=cpu[0], dur=1., pid=1, tid=1,
                           args={"correlation": correlation[0]}))
        events.append(dict(cat=category, name=name, ts=1_000_000 + cpu[0], dur=duration * 1000,
                           args={"correlation": correlation[0]}))
        cpu[0] += 2
        gpu[0] += duration
        return output

    class Event:
        def __init__(self, **_):
            pass

        def record(self):
            self.timestamp = gpu[0]

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    class Tracer:
        def __init__(self, **_):
            pass

        def __enter__(self):
            events.clear()
            return self

        def __exit__(self, *_):
            Path(self.trace_path).write_text(json.dumps({"traceEvents": events}))

    tracer_module = ModuleType("frontier.profiling.utils.record_function_tracer")
    tracer_module.RecordFunctionTracer = Tracer
    monkeypatch.setitem(sys.modules, tracer_module.__name__, tracer_module)
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                            profiler=SimpleNamespace(record_function=annotation))
    def native_forward():
        cpu[0] += 1000.
        gpu[0] += 1.
        return activity("kernel", "native_kernel", .01)

    target = SimpleNamespace(forward=native_forward)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.trace_dir, wrapper.scopes = torch, tmp_path, NativeScopes(torch)
    wrapper._bindings = lambda: [(target, "forward", "step5_qkv_projection")]
    wrapper._forward = lambda *_, **__: target.forward()
    wrapper._restore = lambda _: activity("gpu_memcpy", "outside_root_restore", 100.)
    window, raw, evidence, last = wrapper._gpu_activity(None, None, None,
        ("step5_qkv_projection",), "fixture", sample_count=2, min_window_ms=.05)
    assert last is output
    assert window["step5_layer_e2e"] == pytest.approx([.01, .01])
    assert all(duration >= .05 for duration in json.loads(evidence["window_durations_ms"]))
    assert min(json.loads(evidence["window_replays"])) >= 5
    assert min(json.loads(evidence["trace_window_event_durations_ms"])) > 5.
    assert len(raw["step5_qkv_projection"]) >= 10
    assert evidence["memory_copy_count"] == 0
    assert evidence["skipped_memory_copy_count"] == 2
    assert evidence["operator_sum_to_layer_ratio"] == pytest.approx(1.)
    assert {item["step5_layer_e2e"] for item in evidence["per_replay_kernel_counts"].values()} == {1}
    assert evidence["kernel_count"] >= 10


def test_piecewise_scoped_replays_retain_their_own_operator_events(monkeypatch):
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    clock, records, current_duration = [0.], [], [0.]
    durations = iter((1., 2., 3.))
    output = object()

    class Event:
        def __init__(self, **_):
            pass

        def record(self):
            self.timestamp = clock[0]
            records.append(self)

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    def native():
        clock[0] += current_duration[0]
        return output

    target = SimpleNamespace(forward=native)
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None),
        profiler=SimpleNamespace(record_function=lambda _: nullcontext()))

    def capture(_torch, _native, _layer, forward, mode):
        assert mode == "piecewise"
        duration = next(durations)
        current_duration[0] = duration
        begin = len(records)
        forward(runtime_mode="piecewise")
        _, start, end, _ = records[begin:]

        def replay_layer():
            start.record()
            clock[0] += duration
            end.record()
            return output

        return lambda: forward(runtime_mode="piecewise", layer=replay_layer)

    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.capture_native_replay", capture)
    wrapper = VllmStep5AttentionWrapper.__new__(VllmStep5AttentionWrapper)
    wrapper.torch, wrapper.native, wrapper.layer = torch, object(), object()
    wrapper.graph_mode, wrapper.scopes = "piecewise", NativeScopes(torch)
    wrapper._bindings = lambda: [(target, "forward", "projection")]
    wrapper._restore = lambda _: None
    wrapper._validate_output_state = lambda *args: None
    wrapper._forward = lambda *args, runtime_mode, layer=None: (target.forward if layer is None else layer)()
    replays = [wrapper._scoped_replay(None, None, (), output, ()) for _ in range(3)]
    for duration, replay in zip((1., 2., 3.), replays):
        assert replay.run() is output
        assert replay.times() == {"projection": duration, "step5_layer_scoped": duration}


def test_one_token_prefill_is_rejected():
    with pytest.raises(ValueError, match="native one-token"):
        Step5ProfileInput(3, (1,), (512,), "prefill")


def test_dry_run_grid_does_not_import_native_runtime(tmp_path, monkeypatch, capsys):
    from frontier.profiling.step5.main import main

    monkeypatch.setitem(sys.modules, "vllm", None)
    main(["--model-path", str(tmp_path / "no_model_needed"), "--max-model-len", "8192",
          "--output-dir", str(tmp_path / "no_output_created"), "--dry-run",
          "--prefixes", "0", "4104", "--cudagraph-capture-sizes", "1", "2"])
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 12
    assert result["gpu_active_rows"] == 8
    assert not (tmp_path / "no_output_created").exists()


def test_empty_profile_grid_is_rejected(config):
    with pytest.raises(ValueError, match="must contain"):
        # Keep workload admission explicit before the GPU wrapper exists.
        build_profile_inputs(config, prefill_tokens=[], prefixes=[],
                             decode_batches=[], decode_contexts=[])


@pytest.mark.parametrize("failure", ["whole_missing", "zero_mean", "short_count", "mismatched_count"])
def test_whole_layer_and_sample_count_contract(config, failure):
    result = row(config)
    stats = result["time_stats"]
    if failure == "whole_missing":
        del stats["step5_layer_e2e"]
    elif failure == "zero_mean":
        stats["step5_layer_e2e"]["mean"] = 0
    elif failure == "short_count":
        for timing in stats.values():
            timing["count"] = 1
    else:
        stats["step5_sparse_attention"]["count"] = 3
    with pytest.raises(ValueError):
        rows_to_dataframe([result])


@pytest.fixture
def fake_native_wrapper(config, monkeypatch):
    constructed = []

    class Wrapper:
        def __init__(self, **kwargs):
            self.args = kwargs
            constructed.append((kwargs["layer_id"], kwargs["graph_mode"]))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def profile(self, cell, **kwargs):
            assert cell.graph_mode == self.args["graph_mode"]
            event = build_row(config, cell, tp=8, block_size=16, max_model_len=8192,
                runtime_signature="cpu_schema_fixture_not_measurement", device="h800",
                time_stats=fake_stats(), evidence={"attention_backend": "native_fixture",
                "stability_pass": True, "time_cv": 0., "trace_path": "cpu_fixture_no_trace"})
            active = {**event, "measurement_type": "KERNEL_ONLY"} if cell.graph_mode == "none" else None
            return Step5ProfileResult(event_row=event, gpu_active_row=active)

    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.VllmStep5AttentionWrapper", Wrapper)
    return constructed


def small_main_args(tmp_path):
    return ["--model-path", str(tmp_path / "model"), "--max-model-len", "8192",
            "--output-dir", str(tmp_path / "output"), "--prefill-tokens", "16",
            "--prefixes", "0", "--decode-contexts", "512", "--cudagraph-capture-sizes", "1", "2"]


def test_main_separates_native_graph_policies_and_writes_complete_summary(tmp_path, fake_native_wrapper):
    from frontier.profiling.step5.main import main

    main(small_main_args(tmp_path))
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["expected_rows"] == summary["actual_rows"] == summary["stable_rows"] == 4
    assert summary["expected_gpu_active_rows"] == summary["gpu_active_rows"] == 2
    assert summary["complete"] and not summary["admitted"]
    assert set(fake_native_wrapper) == {(0, "none"), (0, "full"), (3, "none"), (3, "full")}


def test_main_writes_isolated_no_metadata_identity_and_independent_role_output(tmp_path, fake_native_wrapper):
    import pandas as pd
    from frontier.profiling.step5.main import main

    path = tmp_path / "idle.json"
    path.write_text(json.dumps([dict(global_layer_id=0, query_lens=[1], context_lens=[0],
                                     phase="decode", has_attention_metadata=False)]))
    for role in ("PREFILL", "DECODE"):
        output = tmp_path / role
        main(["--model-path", str(tmp_path), "--max-model-len", "8192", "--input-file", str(path),
              "--role", role, "--output-dir", str(output)])
        summary = json.loads((output / "summary.json").read_text())
        assert summary["complete"] and summary["actual_rows"] == summary["gpu_active_rows"] == 1
        frame = pd.read_csv(summary["gpu_active_csv"])
        assert frame.has_attention_metadata.tolist() == [0]
        assert frame.graph_mode.tolist() == ["none"]
        assert frame.measurement_role.tolist() == [role]
        assert frame.block_table_columns.tolist() == [0]
        assert frame.native_workload.tolist() == ["isolated_no_metadata"]


def test_failed_csv_write_is_not_a_completed_row(tmp_path, fake_native_wrapper, monkeypatch):
    import pandas as pd
    from frontier.profiling.step5.main import main

    def unavailable(*args, **kwargs):
        raise OSError("CPU test injected output failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", unavailable)
    with pytest.raises(OSError, match="injected output failure"):
        main(small_main_args(tmp_path))
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["actual_rows"] == 0
    assert not summary["complete"]


def test_main_does_not_retry_failed_quality(tmp_path, fake_native_wrapper, monkeypatch):
    from frontier.profiling.step5 import main as producer, vllm_wrapper

    calls = []
    original = vllm_wrapper.VllmStep5AttentionWrapper.profile

    def unstable(self, cell, **kwargs):
        calls.append(cell)
        result = original(self, cell, **kwargs)
        for row in (result.event_row, result.gpu_active_row):
            if row is not None:
                row["stability_pass"] = False
        return result

    monkeypatch.setattr(vllm_wrapper.VllmStep5AttentionWrapper, "profile", unstable)
    producer.main(small_main_args(tmp_path))
    assert len(calls) == len(set(calls)) == 4
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["stable_rows"] == 0 and summary["actual_rows"] == 4


@pytest.mark.parametrize("fail_workspace", [False, True])
def test_native_initialization_keeps_config_until_group_cleanup(config, tmp_path, monkeypatch, fail_workspace):
    """Run the real constructor with CPU-only native boundaries, not a fake wrapper."""
    from frontier.profiling.step5.vllm_wrapper import VllmStep5AttentionWrapper

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    calls = []
    current = []
    groups = set()
    world = []
    native = SimpleNamespace(quant_config=object())

    def require_config(name):
        assert current and current[-1] is native, f"Current vLLM config is not set: {name}"
        calls.append(name)

    @contextmanager
    def set_current(value):
        current.append(value)
        calls.append("enter_config")
        try:
            yield value
        finally:
            assert not groups, "Config must outlive the groups it configured"
            current.pop()
            calls.append("exit_config")

    def configure_quant(*args):
        # Do not fail here: the pre-fix constructor must reach the observed
        # initialize_model_parallel assertion, rather than a different failure.
        calls.append(("configure_quant", bool(current)))

    def init_distributed(**kwargs):
        calls.append(("init_distributed", bool(current)))
        if not world:
            world.append(SimpleNamespace(valid=True))
        groups.add("distributed")

    def initialize_model_parallel(*args):
        require_config("initialize_model_parallel")
        assert "distributed" in groups
        assert world[0].valid, "Invalid process group: vLLM retained the destroyed world group"
        groups.add("model_parallel")

    def destroy_model_parallel():
        require_config("destroy_model_parallel")
        groups.remove("model_parallel")

    def destroy_distributed():
        calls.append(("destroy_distributed", bool(current)))
        assert "model_parallel" not in groups
        world[0].valid = False
        groups.remove("distributed")

    def destroy_native_distributed():
        destroy_distributed()
        world.clear()

    def init_workspace(device):
        require_config("init_workspace")
        assert groups == {"distributed", "model_parallel"}
        if fail_workspace:
            raise RuntimeError("CPU-injected workspace initialization failure")

    paths = (
        "torch", "vllm.config", "vllm.distributed.parallel_state", "vllm.engine.arg_utils",
        "vllm.model_executor.model_loader.utils", "vllm.models.step5.nvidia.model",
        "vllm.model_executor.layers.linear", "vllm.utils.network_utils", "vllm.v1.worker.workspace",
    )
    modules = {}
    for path in paths:
        parts = path.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name not in modules:
                modules[name] = ModuleType(name)
                modules[name].__path__ = []
                monkeypatch.setitem(sys.modules, name, modules[name])
                if depth > 1:
                    setattr(modules[".".join(parts[:depth - 1])], parts[depth - 1], modules[name])
    torch = modules["torch"]
    torch.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda _: None)
    torch.version = SimpleNamespace(hip=None, cuda="cpu_fixture")
    torch.distributed = SimpleNamespace(is_initialized=lambda: bool(groups),
                                        destroy_process_group=destroy_distributed)
    torch.manual_seed = lambda _: None
    torch.get_default_dtype = lambda: "original_cpu_fixture_dtype"
    torch.set_default_dtype = lambda _: None
    torch.bfloat16 = "bfloat16"
    torch.device = lambda _: nullcontext()
    torch.no_grad = nullcontext
    torch.__version__ = modules["vllm"].__version__ = "cpu_fixture"
    modules["vllm.config"].set_current_vllm_config = set_current
    parallel = modules["vllm.distributed.parallel_state"]
    parallel.init_distributed_environment = init_distributed
    parallel.initialize_model_parallel = initialize_model_parallel
    parallel.destroy_model_parallel = destroy_model_parallel
    parallel.destroy_distributed_environment = destroy_native_distributed
    outer = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=object()),
                            with_hf_config=lambda _: native)
    modules["vllm.engine.arg_utils"].EngineArgs = lambda **_: SimpleNamespace(create_engine_config=lambda: outer)
    modules["vllm.model_executor.model_loader.utils"].configure_quant_config = configure_quant
    modules["vllm.utils.network_utils"].get_open_port = lambda: 12345
    modules["vllm.v1.worker.workspace"].init_workspace_manager = init_workspace
    step_model = modules["vllm.models.step5.nvidia.model"]
    step_model.Step5ForCausalLM = object()
    for owner in (step_model, modules["vllm.model_executor.layers.linear"]):
        owner.get_tensor_model_parallel_world_size = lambda: 1
        owner.get_tensor_model_parallel_rank = lambda: 0

    def build_metadata(**kwargs):
        require_config("build_metadata")
        return object()

    layer = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True), attn=SimpleNamespace(
        layer_name="cpu_fixture.attn", attn_backend=SimpleNamespace(get_builder_cls=lambda: build_metadata)))
    layer.eval = lambda: layer
    layer.requires_grad_ = lambda _: layer

    def construct_layer(self):
        require_config("construct_layer")
        assert step_model.get_tensor_model_parallel_world_size() == 8
        return layer

    def bind_cache(self):
        require_config("bind_cache")
        self.cache_spec = object()

    monkeypatch.setattr(VllmStep5AttentionWrapper, "_construct_layer", construct_layer)
    monkeypatch.setattr(VllmStep5AttentionWrapper, "_initialize_weights", lambda _: require_config("initialize_weights"))
    monkeypatch.setattr(VllmStep5AttentionWrapper, "_bind_cache", bind_cache)
    monkeypatch.setattr(VllmStep5AttentionWrapper, "_check_shapes", lambda _: require_config("check_shapes"))
    monkeypatch.setattr("frontier.profiling.step5.vllm_wrapper.importlib.metadata.version", lambda _: "cpu_fixture")
    kwargs = dict(config=config, model_path="cpu_fixture", device="h800", tensor_parallel_size=8,
                  layer_id=0, block_size=16, max_model_len=8192, max_batch_size=1,
                  trace_dir=tmp_path, seed=0, graph_mode="none")
    if fail_workspace:
        with pytest.raises(RuntimeError, match="CPU-injected workspace"):
            VllmStep5AttentionWrapper(**kwargs)
    else:
        # The real grid creates sequential layer/graph-policy wrappers. Native
        # WORLD state must be reset, not just its torch process group destroyed.
        for _ in range(2):
            wrapper = VllmStep5AttentionWrapper(**kwargs)
            require_config("wrapper_alive")
            assert not wrapper.layer.o_proj.reduce_results
            wrapper.close()
    assert not current and not groups and not world
    assert calls[0] == "enter_config"
    assert ("configure_quant", True) in calls and ("init_distributed", True) in calls
    assert calls[-3:] == ["destroy_model_parallel", ("destroy_distributed", True), "exit_config"]
    assert step_model.get_tensor_model_parallel_world_size() == 1


def test_staged_credential_wrapper_sources_environment_before_launcher(tmp_path, monkeypatch):
    from frontier.profiling.step5 import job

    root = tmp_path / "repo"
    for name in ("frontier", "data/config", "tests/integration/step5_profiling"):
        (root / name).mkdir(parents=True)
    for name in ("submit_profiling_job.py", "run_on_worker.sh"):
        (root / "tests/integration/step5_profiling" / name).write_text("# CPU fixture\n")
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    environment = tmp_path / "stepmind_env.sh"
    environment.write_text("export BRAINPP_ACCESS_KEY=cpu_fixture_key\n"
                           "export BRAINPP_SECRET_KEY=cpu_fixture_secret\n")
    monkeypatch.setattr(job, "REPO_ROOT", root)
    monkeypatch.setattr(job, "STEPMIND_ENV", environment)
    stage = tmp_path / "stage"
    manifest = job.stage_smoke(stage, "attention_smoke_20261020_02", dependency_site=dependencies)
    assert manifest["cloud_output"].endswith("/attention_smoke_20261020_02")
    assert {"submit.sh", "submit_with_env.sh"} <= manifest["sha256"].keys()
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == expected
    wrapper = stage / "submit_with_env.sh"
    assert f"source {environment}" in wrapper.read_text()
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    # Only this disposable CPU fixture gets a stub payload. Never execute the
    # generated real platform launcher or source the real credential file.
    (stage / "submit.sh").write_text(
        'test "$BRAINPP_ACCESS_KEY" = cpu_fixture_key\n'
        'test "$BRAINPP_SECRET_KEY" = cpu_fixture_secret\n'
        'test "$STEPMIND_BACKEND" = rjob\n'
        'echo CPU_CREDENTIAL_WRAPPER_PASS\n')
    result = subprocess.run(["bash", str(wrapper)], env={"PATH": os.environ["PATH"]},
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "CPU_CREDENTIAL_WRAPPER_PASS"
