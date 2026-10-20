"""Synthetic Kineto trace contracts; no native runtime or GPU timing claims."""

import json

import pytest

from frontier.profiling.step5.device_activity import parse_device_activity


QKV = "step5_qkv_projection"
PREPARE = "step5_attention_prepare"
ROOT = "step5_iteration/0/0"


def annotation(name, start, duration, *, pid=1, tid=2):
    return {"cat": "user_annotation", "name": name, "ts": start, "dur": duration,
            "pid": pid, "tid": tid}


def launch(correlation, start, *, duration=1, category="cuda_runtime", pid=1, tid=2):
    return {"cat": category, "name": "cudaLaunchKernel", "ts": start, "dur": duration,
            "pid": pid, "tid": tid, "args": {"correlation": correlation}}


def activity(correlation, duration, *, start=1000, name="native_kernel", category="kernel"):
    return {"cat": category, "name": name, "ts": start, "dur": duration,
            "pid": 0, "tid": 7, "args": {"correlation": correlation}}


def trace(tmp_path, events):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def test_async_gpu_timestamp_is_not_used_for_cpu_scope_association(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 50, start=1000)])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"] == {ROOT: {QKV: .05, "step5_layer_e2e": .05}}
    assert result["kernel_names"] == ["native_kernel"]
    assert result["kernel_count"] == 1
    assert result["kernel_time_ms"] == .05


def test_all_graph_launch_activity_and_runtime_driver_aliases_count_once(tmp_path):
    runtime = launch(5, 11, duration=8)
    runtime["name"] = "cudaGraphLaunch"
    driver = launch(5, 12, category="cuda_driver")
    driver["name"] = "cuGraphLaunch"
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            runtime, driver, runtime.copy(),
                            activity(5, 4, name="first"), activity(5, 6, name="second"),
                            activity(5, 3, name="Memcpy DtoD", category="gpu_memcpy")])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"][ROOT] == pytest.approx({QKV: .013, "step5_layer_e2e": .013})
    assert result["kernel_count"] == 2
    assert result["kernel_names"] == ["first", "second"]
    assert result["memory_copy_count"] == 1
    assert result["memory_copy_names"] == ["Memcpy DtoD"]
    assert result["memory_copy_time_ms"] == .003


def test_nested_owners_and_repeated_scopes_do_not_doublecount(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            annotation(PREPARE, 12, 10), launch(1, 15), activity(1, 4),
                            annotation(PREPARE, 40, 10), launch(2, 42), activity(2, 6),
                            annotation(QKV, 60, 10), launch(3, 62), activity(3, 3)])
    result = parse_device_activity(path, [QKV, PREPARE])
    assert result["per_iteration"][ROOT] == pytest.approx({
        QKV: .007, PREPARE: .006, "step5_layer_e2e": .013})
    assert result["kernel_count"] == 3


def test_two_windows_and_replays_use_cpu_launches_not_gpu_timestamps(tmp_path):
    names = ["step5_iteration/0/0", "step5_iteration/0/1", "step5_iteration/1/0"]
    events = []
    for index, name in enumerate(names):
        start = index * 100
        events += [annotation(name, start, 90), annotation(QKV, start + 10, 20),
                   launch(index, start + 15),
                   activity(index, (index + 1) * 10, start=1000 - index * 100)]
    result = parse_device_activity(trace(tmp_path, list(reversed(events))), [QKV])
    assert result["per_iteration"] == {
        name: {QKV: (index + 1) / 100, "step5_layer_e2e": (index + 1) / 100}
        for index, name in enumerate(names)}


@pytest.mark.parametrize("pid,tid", [(9, 2), (1, 9)])
def test_cpu_pid_and_tid_are_required_for_attribution(tmp_path, pid, tid):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15, pid=pid, tid=tid), activity(1, 50),
                            launch(2, 16), activity(2, 10)])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"][ROOT][QKV] == .01
    assert result["skipped_kernel_count"] == 1


def test_simultaneous_iterations_on_distinct_threads_are_separate(tmp_path):
    other = "step5_iteration/1/0"
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10),
                            annotation(other, 0, 100, tid=3), annotation(QKV, 10, 20, tid=3),
                            launch(2, 15, tid=3), activity(2, 20)])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"][ROOT][QKV] == .01
    assert result["per_iteration"][other][QKV] == .02


@pytest.mark.parametrize("other", ["step5_iteration/0/1", ROOT])
def test_correlation_with_conflicting_iteration_or_operator_is_rejected(tmp_path, other):
    events = [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
              launch(1, 15), activity(1, 10)]
    if other != ROOT:
        events += [annotation(other, 100, 100), annotation(QKV, 110, 20), launch(1, 115)]
    else:
        events += [annotation(PREPARE, 40, 20), launch(1, 45)]
    path = trace(tmp_path, events)
    with pytest.raises(ValueError, match="conflicting iteration/operator owners"):
        parse_device_activity(path, [QKV, PREPARE] if other == ROOT else [QKV])


@pytest.mark.parametrize("annotations", [[], [annotation(QKV, 10, 20)]])
def test_missing_active_operator_activity_is_not_zero_filled(tmp_path, annotations):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), *annotations])
    with pytest.raises(ValueError, match="missing GPU activity.*step5_qkv_projection"):
        parse_device_activity(path, [QKV])


def test_known_inactive_operators_are_not_added(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10)])
    times = parse_device_activity(path, [QKV])["per_iteration"][ROOT]
    assert PREPARE not in times


def test_an_active_annotation_without_activity_cannot_hide_behind_other_operators(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10), annotation(PREPARE, 40, 10)])
    with pytest.raises(ValueError, match="missing GPU activity.*step5_attention_prepare"):
        parse_device_activity(path, [QKV, PREPARE])


def test_copy_only_active_operator_is_kept(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(PREPARE, 10, 20),
                            launch(1, 15), activity(1, 7, name="Memcpy DtoD", category="gpu_memcpy")])
    result = parse_device_activity(path, [PREPARE])
    assert result["per_iteration"][ROOT] == {PREPARE: .007, "step5_layer_e2e": .007}
    assert result["kernel_count"] == 0
    assert result["memory_copy_count"] == 1


def test_measured_gpu_activity_without_an_operator_owner_has_a_clear_error(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), launch(1, 15), activity(1, 10)])
    with pytest.raises(ValueError, match="no operator owner"):
        parse_device_activity(path, [QKV])


def test_prime_kernels_and_restore_copies_outside_each_iteration_are_excluded(tmp_path):
    other = "step5_iteration/1/0"
    restore = launch(11, 90)
    restore["name"] = "cudaMemcpyAsync"
    second_restore = launch(12, 290)
    second_restore["name"] = "cudaMemcpyAsync"
    measured_copy = launch(2, 116)
    measured_copy["name"] = "cudaMemcpyAsync"
    path = trace(tmp_path, [
        annotation("profiler_cuda_capture_prime", 0, 50), launch(10, 10),
        activity(10, 1000, start=120, name="prime"),
        restore, activity(11, 2000, start=125, name="restore", category="gpu_memcpy"),
        annotation(ROOT, 100, 100), annotation(QKV, 110, 20),
        launch(1, 115), activity(1, 10, start=1000),
        measured_copy, activity(2, 5, start=1100, name="Memcpy DtoD", category="gpu_memcpy"),
        second_restore, activity(12, 3000, start=325, name="restore", category="gpu_memcpy"),
        annotation(other, 300, 100), annotation(QKV, 310, 20),
        launch(3, 315), activity(3, 20, start=1200)])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"] == {
        ROOT: {QKV: .015, "step5_layer_e2e": .015},
        other: {QKV: .02, "step5_layer_e2e": .02}}
    assert result["kernel_count"] == 2
    assert result["kernel_names"] == ["native_kernel"]
    assert result["memory_copy_count"] == 1
    assert result["memory_copy_names"] == ["Memcpy DtoD"]
    assert result["memory_copy_time_ms"] == .005
    assert result["skipped_kernel_count"] == 1
    assert result["skipped_memory_copy_count"] == 2
    assert result["skipped_uncorrelated_activity_count"] == 0


def test_uncorrelated_gpu_records_are_explicitly_skipped_not_measured(tmp_path):
    uncorrelated = activity(2, 500, start=15)
    uncorrelated["args"] = {}
    orphan = activity(3, 700, start=20, category="gpu_memcpy")
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10), uncorrelated, orphan])
    result = parse_device_activity(path, [QKV])
    assert result["per_iteration"][ROOT] == {QKV: .01, "step5_layer_e2e": .01}
    assert result["kernel_count"] == 1
    assert result["memory_copy_count"] == 0
    assert result["skipped_kernel_count"] == 1
    assert result["skipped_memory_copy_count"] == 1
    assert result["skipped_uncorrelated_activity_count"] == 2


@pytest.mark.parametrize("name", ["cudaLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch", "cudaMemcpyAsync"])
@pytest.mark.parametrize("missing", ["correlation", "gpu_activity"])
def test_measured_launch_correlation_gaps_cannot_hide_behind_valid_operator_activity(tmp_path, name, missing):
    broken = launch(2, 17, category="cuda_driver" if name.startswith("cuLaunch") else "cuda_runtime")
    broken["name"] = name
    if missing == "correlation":
        broken["args"] = {}
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10), broken])
    with pytest.raises(ValueError, match="Measured GPU launch.*correlation.*activity"):
        parse_device_activity(path, [QKV])


def test_nonlaunch_runtime_apis_without_device_activity_inside_roots_are_ignored(tmp_path):
    event_record = launch(2, 17)
    event_record["name"] = "cudaEventRecord"
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10), event_record])
    assert parse_device_activity(path, [QKV])["kernel_count"] == 1


def test_correlation_shared_between_preparation_and_measured_launch_is_ambiguous(tmp_path):
    path = trace(tmp_path, [launch(1, -10), annotation(ROOT, 0, 100),
                            annotation(QKV, 10, 20), launch(1, 15), activity(1, 10)])
    with pytest.raises(ValueError, match="measured and unmeasured launches"):
        parse_device_activity(path, [QKV])


def test_launch_starting_inside_root_cannot_be_skipped_by_crossing_its_end(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10),
                            launch(2, 99, duration=2), activity(2, 20)])
    with pytest.raises(ValueError, match="crosses iteration boundary"):
        parse_device_activity(path, [QKV])


def test_crossing_operator_intervals_are_ambiguous(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 30),
                            annotation(PREPARE, 20, 30), launch(1, 25), activity(1, 10)])
    with pytest.raises(ValueError, match="ambiguous operator owner"):
        parse_device_activity(path, [QKV, PREPARE])


@pytest.mark.parametrize("kind", ["duplicate_root", "overlap_root", "no_root"])
def test_iteration_identity_must_be_unambiguous(tmp_path, kind):
    events = [annotation(ROOT, 0, 100)]
    if kind == "duplicate_root":
        events.append(annotation(ROOT, 100, 100))
    elif kind == "overlap_root":
        events.append(annotation("step5_iteration/0/1", 50, 100))
    else:
        events = []
    with pytest.raises(ValueError, match={"duplicate_root": "Duplicate iteration", "overlap_root": "Overlapping iteration",
                                         "no_root": "No iteration"}[kind]):
        parse_device_activity(trace(tmp_path, events), [QKV])


def test_unrelated_categories_and_uncorrelated_cpu_apis_are_ignored(tmp_path):
    path = trace(tmp_path, [annotation(ROOT, 0, 100), annotation(QKV, 10, 20),
                            launch(1, 15), activity(1, 10),
                            launch(2, 500), {"cat": "cpu_op", "name": "aten::linear"},
                            annotation("unrelated_annotation", 500, 10)])
    assert parse_device_activity(path, [QKV])["kernel_count"] == 1


def test_custom_iteration_prefix_and_launch_on_annotation_boundaries(tmp_path):
    path = trace(tmp_path, [annotation("custom/0", 0, 100), annotation(QKV, 0, 100),
                            launch(1, 0, duration=100), activity(1, 10)])
    assert parse_device_activity(path, [QKV], "custom/")["per_iteration"] == {
        "custom/0": {QKV: .01, "step5_layer_e2e": .01}}
