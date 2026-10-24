"""CPU envelopes and source-backed live range controls; no CUDA initialization."""

import ctypes
import copy
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.test_cupti_graph_resource_payload import probe, capture, envelope, invoke

SOURCE = Path(__file__).parents[2] / "task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/probe_scope_lifetime.py"
spec = importlib.util.spec_from_file_location("scope_subject", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


@pytest.fixture
def case(capture, probe, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("FRONTIER_NATIVE_PROFILE_LIMITS", '{"records":10000,"bytes":10000000}')
    profile = subject.serving_profile()
    assert Path(profile.range.__wrapped__.__code__.co_filename).resolve() == SOURCE.parents[4] / "tests/comparison/calibration/native_serve_profile.py"
    base, state = capture
    sources, clones = (11, 31, 19, 47, 23), (111, 97, 203, 61, 143)
    original_ids = (2**59 + 23, 2**59 + 701, 2**59 + 19, 2**59 + 202, 2**59 + 61)
    full_ids = (2**60 + 999, 2**60 + 17, 2**60 + 8101, 2**60 + 33, 2**60 + 519)
    mapping = dict(zip(sources + clones + (3030,), original_ids + full_ids + (2**58 + 7,)))
    def node_id(handle):
        assert state["in_callback"], "Post-capture/copied-handle query is forbidden"
        if state.get("query_error"):
            raise RuntimeError("actual query failure")
        return mapping[handle]
    def node_type(handle):
        assert state["in_callback"], "NodeGetType outside callback is forbidden"
        return (0, 0)
    ids = SimpleNamespace(library=base.ids.library, node_id=node_id, graph_id=base.ids.graph_id, exec_id=base.ids.exec_id)
    value = subject.ScopeGraphCallbacks(ids, SimpleNamespace(cudaGraphNodeGetType=node_type), profile)
    def dispatch(callback_id, **fields):
        native, storage, resource = envelope(probe, **fields)
        invoke(probe, (value, state), callback_id, ctypes.byref(resource))
        native.node = native.graph = native.originalNode = 999
        resource.resourceDescriptor = None
    report = dict(callback_capture=value.receipt, capture_frame=subject.frame(), windows=[], errors=[],
                  profile_errors=profile.errors, default_constructor=True, capture_complete=True, replay_complete=True)
    value.start()
    ops = [(lambda a, b, out, node=node: dispatch(13, node=node, graph=701), 0, 0, 0) for node in sources]
    subject.capture_operations(ops, profile, report)
    dispatch(13, node=3030, graph=701)
    for source, clone in reversed(tuple(zip(sources, clones))):
        dispatch(20, node=clone, originalNode=source, graph=900, originalGraph=701)
    dispatch(18, graph=701, graphExec=702)
    assert value.close()
    graph = SimpleNamespace(raw_cuda_graph=lambda: (_ for _ in ()).throw(RuntimeError("defaultFalse")), raw_cuda_graph_exec=lambda: 702)
    subject.lifetime(graph, report)
    events = [dict(name=probe.TARGET, ph="X", cat="user_annotation", pid=1, tid=2, ts=10, dur=20),
              dict(name="cudaGraphLaunch", cat="cuda_runtime", pid=1, tid=2, ts=11, dur=1, args=dict(correlation=71))]
    events += [dict(cat="kernel", args={"correlation":71,"graph id":202,"graph node id":key}) for key in reversed(full_ids)]
    return SimpleNamespace(report=report, events=events, value=value, state=state, dispatch=dispatch, ops=ops, profile=profile)


def test_actual_range_callback_and_full_ids_join(case):
    subject.analyze(case.report, case.events)
    assert case.report["verdict"] == "PASS" and case.report["unmatched_created_count"] == 1
    assert case.report["links"][0]["source"]["node_id"] != case.report["links"][0]["clone"]["node_id"]
    assert getattr(case.profile.local, "scope", None) is None
    with pytest.raises(AssertionError, match="forbidden"):
        case.value.ids.node_id(11)


@pytest.mark.parametrize("kind,field,value", [
    ("source", "scope", "stale_same_tid"), ("source", "monotonic_ns", "closed"),
    ("window", "slot", 4), ("clone", "original_node_id", 2**62 + 1),
    ("clone", "original_node", 999), ("source", "queried_type", 2),
    ("source", "node_type", 2), ("source", "graph", 800), ("source", "graph_id", 102),
    ("clone", "node_id", 2**60 + 33), ("exec", "graph_exec", 700),
    ("kernel", "graph node id", 2**60 + 33), ("kernel", "graph node id", 2**63 + 71)])
def test_wrong_scope_window_and_full_id_records(case, kind, field, value):
    rows = case.report["callback_capture"]["events"]
    selected = {name: next(r for r in rows if r["callback_id"] == callback) for name, callback in (("source",13),("clone",20),("exec",18))}
    selected.update(window=case.report["windows"][0], kernel=case.events[-1]["args"])
    selected[kind][field] = case.report["windows"][0]["end"] if value == "closed" else value
    subject.analyze(case.report, case.events)
    expected = "NEGATIVE_LIVE_TLS_ROUTE" if kind == "window" or field in ("scope", "monotonic_ns") else "UNVERIFIED"
    assert case.report["verdict"] == expected


@pytest.mark.parametrize("fault", ["ambiguous", "clone_missing", "clone_duplicate", "created_duplicate", "trace", "query", "descriptor", "stage"])
def test_incomplete_callbacks_or_trace_preserve_evidence(case, fault, monkeypatch):
    collection = case.report["windows"] if fault == "ambiguous" else case.events if fault == "trace" else case.report["callback_capture"]["events"]
    if fault == "query":
        case.state["query_error"] = True
        case.dispatch(13, node=11, graph=701)
    elif fault == "descriptor":
        invoke(subject.probe, (case.value, case.state), 13, ctypes.byref(subject.probe.CuptiResourceData()))
    elif fault == "stage":
        monkeypatch.setattr(subject.probe, "_diagnostic_stage", lambda *args: (_ for _ in ()).throw(OSError("stage sink failure")))
        case.dispatch(13, node=11, graph=701)
    else:
        row = collection[0] if fault in ("ambiguous", "trace") else next(r for r in collection if r["callback_id"] == (13 if fault == "created_duplicate" else 20))
        if fault in ("clone_missing", "trace"):
            collection.remove(row)
        else:
            collection.append(copy.deepcopy(row))
    subject.analyze(case.report, case.events)
    assert case.report["verdict"] == ("NEGATIVE_LIVE_TLS_ROUTE" if fault == "ambiguous" else "UNVERIFIED")


def test_other_thread_has_distinct_identities_and_no_live_tls(case):
    case.report["callback_capture"]["events"][:] = [r for r in case.report["callback_capture"]["events"] if r["callback_id"] != 13]
    case.report["windows"] = []
    def deferred(a, b, out):
        thread = threading.Thread(target=lambda: case.dispatch(13, node=11, graph=701))
        thread.start()
        thread.join()
    subject.capture_operations([(deferred, 0, 0, 0)] + case.ops[1:], case.profile, case.report)
    subject.analyze(case.report, case.events)
    assert case.report["verdict"] == "NEGATIVE_LIVE_TLS_ROUTE"
    assert case.report["thread_observations"][-1] == {"native_id":False,"python_ident":False}


@pytest.mark.parametrize("failure", [None, "cuptiSubscribe", "cuptiEnableCallback", "cuptiUnsubscribe", "enable13"])
def test_owned_lifecycle_never_unsubscribes_foreign_slot(capture, failure):
    base, state = capture
    if failure == "enable13":
        original = base.ids.library.cuptiEnableCallback
        base.ids.library.cuptiEnableCallback = lambda *args: 7 if args[-1] == 13 else original(*args)
    elif failure:
        state["results"][failure] = 7
    value = subject.ScopeGraphCallbacks(base.ids, None, SimpleNamespace(local=threading.local()))
    value.start()
    assert value.owned == (failure != "cuptiSubscribe")
    assert value.close() == (failure != "cuptiUnsubscribe")
    if failure == "cuptiSubscribe":
        assert state["calls"] == [("cuptiSubscribe", None)]
    state["results"]["cuptiUnsubscribe"] = 0
    assert value.close() and value not in subject.probe._OWNED_CALLBACKS


@pytest.mark.parametrize("result", ["runtime", "returned", "other", "exec_error"])
def test_one_raw_source_observation_never_retains_or_queries_return(result):
    calls, report = [], {"errors":[]}
    def raw():
        calls.append("source")
        if result == "runtime":
            raise RuntimeError("expected")
        if result == "other":
            raise ValueError("original failure")
        return object()
    def executable():
        if result == "exec_error":
            raise RuntimeError("secondary failure")
        return 702
    subject.lifetime(SimpleNamespace(raw_cuda_graph=raw, raw_cuda_graph_exec=executable), report)
    assert calls == ["source"] and "raw_cuda_graph" not in report
    assert report.get("raw_source_runtime_error") is (True if result == "runtime" else None if result == "other" else False)
    assert bool(report["errors"]) == (result in ("other", "exec_error"))
