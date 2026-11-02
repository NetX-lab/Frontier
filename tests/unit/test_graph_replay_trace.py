"""Synthetic Kineto traces for graph clone joins and nested eager attribution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from frontier.profiling.attention.dsa_kernel_trace import ScopeKernelTime, ScopeKey, annotation_name
from frontier.profiling.attention.graph_capture import (
    CaptureRecord,
    GraphCapture,
    GraphNode,
    NodeClone,
    ScopeEntry,
)
from frontier.profiling.attention.graph_replay_trace import (
    attribute_replay_trace,
    parse_step_annotation,
    step_annotation_name,
)

LAYER = (4, "layer")
INNER = (4, "attention")
PROJECTION = (4, "projection")
EMPTY = (4, "zero_kernel")
CONTAINERS = frozenset({LAYER[1]})
GRID = (2, 1, 1)
BLOCK = (32, 1, 1)


def _annotation(name, start, duration, tid=1, pid=10):
    return dict(cat="user_annotation", name=name, ts=start, dur=duration, pid=pid, tid=tid)


def _step(index=0, start=0, duration=400, tid=1, pid=10):
    return _annotation(step_annotation_name(index), start, duration, tid, pid)


def _scope(owner, start, duration, step=0, tid=1, pid=10):
    return _annotation(annotation_name(step, *owner), start, duration, tid, pid)


def _launch(correlation, start, graph=True, tid=1, pid=10, category="cuda_runtime"):
    return dict(cat=category, name="cudaGraphLaunch_v10000" if graph else "cudaLaunchKernel",
                ts=start, dur=1, pid=pid, tid=tid, args={"correlation": correlation})


def _activity(node_id, name, start, duration, correlation=1, stream=7, category="kernel", graph_id=99):
    return dict(cat=category, name=name, ts=start, dur=duration, pid=0, tid=stream,
                args={"graph node id": node_id, "graph id": graph_id, "correlation": correlation,
                      "grid": list(GRID), "block": list(BLOCK), "stream": stream})


def _node(node_id, name=None, owner=INNER, dependencies=(), kind="kernel", capture=0, demangled=None):
    return GraphNode(capture, node_id, kind, name, demangled,
                     GRID if kind == "kernel" else None, BLOCK if kind == "kernel" else None,
                     tuple(dependencies), owner)


def _record(nodes, entries=None):
    captures = tuple(GraphCapture(index, "graph", None, 7) for index in sorted({node.capture for node in nodes}))
    if entries is None:
        entries = (ScopeEntry(0, LAYER, None), ScopeEntry(0, INNER, LAYER),
                   ScopeEntry(0, PROJECTION, LAYER), ScopeEntry(0, EMPTY, LAYER))
    return CaptureRecord(captures, tuple(nodes), tuple(entries), (),
                         tuple(NodeClone(node.node_id + 1000, node.node_id) for node in nodes))


@pytest.fixture
def full():
    nodes = [
        _node(10, "_Zfirst", demangled="first()"),
        _node(11, kind="event_record", dependencies=(10,)),
        _node(12, owner=None, kind="memcpy", dependencies=(11,)),
        _node(13, kind="empty", dependencies=(12,)),
        _node(14, kind="event_wait", dependencies=(13,)),
        _node(15, "_Zsecond", owner=PROJECTION, demangled="second()", dependencies=(14,)),
        _node(16, "outside", owner=None),
    ]
    events = [
        _step(), _launch(1, 10),
        _activity(1010, "_Zfirst", 100, 20),
        _activity(1012, "Memcpy HtoD", 130, 30, category="gpu_memcpy", stream=9),
        _activity(1015, "second()", 200, 30, stream=8),
        _activity(1016, "outside", 60, 4),
    ]
    return _record(nodes), events


def test_full_clone_bridge_costs_windows_and_graph_gaps(full):
    record, events = full
    # A generator deliberately reverses trace order; indexing cannot rely on it.
    result = attribute_replay_trace((event for event in reversed(events)), record, CONTAINERS)
    assert result.scope_times == {
        ScopeKey(0, *LAYER): ScopeKernelTime(0.0, 0),
        ScopeKey(0, *INNER): ScopeKernelTime(0.020, 1),
        ScopeKey(0, *PROJECTION): ScopeKernelTime(0.030, 1),
        ScopeKey(0, *EMPTY): ScopeKernelTime(0.0, 0),
    }
    assert result.layer_windows == {(0, 4): (100.0, 230.0)}
    assert result.outside_kernel_counts == {0: 1}
    assert result.unattributed_launch_count == 0
    intervals = {row.original_node_id: row for row in result.intervals}
    assert intervals[10].name == "_Zfirst"
    assert intervals[10].graph_gap_us is None
    assert intervals[15].name == "second()"
    assert intervals[15].graph_gap_us == 40.0  # Latest dependency is the copy end at 160.
    assert intervals[12].activity == "memcpy"
    assert intervals[12].owner is None
    assert intervals[12].graph_gap_us is None
    assert intervals[15].stream == 8
    assert intervals[15].graph_id == 99
    assert intervals[15].graph_node_id == 1015
    assert intervals[15].correlation == 1
    assert intervals[15].kind == "graph"
    assert intervals[15].end_us == 230.0


@pytest.mark.parametrize("field,value", [
    ("name", "wrong"), ("grid", [3, 1, 1]), ("block", [64, 1, 1]), ("grid", None),
])
def test_kernel_identity_mismatch_reports_both_values(full, field, value):
    record, events = full
    events = deepcopy(events)
    kernel = events[2]
    if field == "name":
        kernel[field] = value
    else:
        kernel["args"][field] = value
    with pytest.raises(ValueError, match=f"{field} mismatch.*trace.*captured"):
        attribute_replay_trace(events, record, CONTAINERS)


@pytest.mark.parametrize("node_id,hint", [(10, "is itself a captured node id"), (9999, "is not a captured node id either")])
def test_positive_graph_id_requires_a_clone_id(full, node_id, hint):
    record, events = full
    events[2]["args"]["graph node id"] = node_id
    with pytest.raises(ValueError, match=f"not a clone id.*{hint}"):
        attribute_replay_trace(events, record, CONTAINERS)


@pytest.mark.parametrize("missing", ["kernel", "memcpy"])
def test_incomplete_graph_launch_fails(full, missing):
    record, events = full
    events = [event for event in events if not (event.get("cat") == ("kernel" if missing == "kernel" else "gpu_memcpy")
              and event.get("name") == ("second()" if missing == "kernel" else "Memcpy HtoD"))]
    with pytest.raises(ValueError, match="missing nodes.*incomplete trace"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_duplicate_activity_of_one_launch_is_ambiguous(full):
    record, events = full
    events.append(deepcopy(events[2]))
    with pytest.raises(ValueError, match="repeats node 10.*ambiguous mapping"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_graph_launch_cannot_span_captures(full):
    record, events = full
    nodes = tuple(replace(node, capture=1) if node.node_id == 15 else node for node in record.nodes)
    record = replace(record, nodes=nodes)
    with pytest.raises(ValueError, match="spans multiple captures"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_piecewise_two_graphs_and_repeated_nested_eager_scopes():
    nodes = [_node(10, "before"), _node(20, "after", owner=PROJECTION, capture=1)]
    record = _record(nodes, [ScopeEntry(0, LAYER, None), ScopeEntry(0, INNER, LAYER),
                            ScopeEntry(1, PROJECTION, LAYER), ScopeEntry(1, EMPTY, LAYER)])
    record = replace(record, captures=tuple(replace(row, kind="breakable_segment", breakable_capture=0) for row in record.captures))
    events = [
        _step(), _launch(1, 10), _activity(1010, "before", 100, 20),
        _scope(LAYER, 50, 100), _scope(INNER, 65, 25), _scope(INNER, 110, 15),
        _scope(EMPTY, 130, 10),
        _launch(2, 70, graph=False), _activity(0, "eager_one", 170, 10, correlation=2),
        _launch(3, 115, graph=False, category="cuda_driver"), _activity(None, "eager_two", 210, 15, correlation=3, stream=9),
        _launch(4, 200), _activity(1020, "after", 300, 30, correlation=4, stream=8),
    ]
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.scope_times[ScopeKey(0, *INNER)] == ScopeKernelTime(0.045, 3)
    assert result.scope_times[ScopeKey(0, *PROJECTION)] == ScopeKernelTime(0.030, 1)
    assert result.scope_times[ScopeKey(0, *LAYER)] == ScopeKernelTime(0.0, 0)
    assert result.scope_times[ScopeKey(0, *EMPTY)] == ScopeKernelTime(0.0, 0)
    assert result.layer_windows == {(0, 4): (100.0, 330.0)}
    eager = [row for row in result.intervals if row.kind == "eager"]
    assert [row.owner for row in eager] == [INNER, INNER]
    assert all(row.original_node_id is None and row.graph_gap_us is None for row in eager)


@pytest.mark.parametrize("kind", ["graph", "eager"])
def test_scope_with_children_owns_kernels_outside_them_unless_declared_container(full, kind):
    # The indexer top-k scope owns its fill and top-k kernels and has a scores child.
    record, events = full
    if kind == "graph":
        entries = record.scope_entries + (ScopeEntry(0, EMPTY, INNER),)
        record = replace(record, scope_entries=entries)
    else:
        record = CaptureRecord()
        events = [_step(), _scope(INNER, 10, 100), _scope(PROJECTION, 20, 20),
                  _launch(1, 30, graph=False), _activity(0, "child", 50, 10),
                  _launch(2, 60, graph=False, category="cuda_driver"), _activity(0, "leftover", 120, 10, correlation=2)]
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.scope_times[ScopeKey(0, *INNER)] == ScopeKernelTime(0.020 if kind == "graph" else 0.010, 1)
    with pytest.raises(ValueError, match="A2:.*owned directly by container scope.*attention"):
        attribute_replay_trace(events, record, CONTAINERS | {INNER[1]})


@pytest.mark.parametrize("kind", ["graph", "eager"])
def test_kernel_owned_directly_by_container_fails(full, kind):
    record, events = full
    if kind == "graph":
        record = replace(record, nodes=tuple(replace(node, owner=LAYER) if node.node_id == 10 else node for node in record.nodes))
    else:
        record = CaptureRecord()
        events = [_step(), _scope(LAYER, 10, 100), _scope(INNER, 20, 20),
                  _launch(1, 60, graph=False), _activity(0, "container", 120, 10)]
    with pytest.raises(ValueError, match="A2:.*owned directly by container scope"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_outside_kernels_and_unattributed_launches_count_once(full):
    record, events = full
    events.extend([
        _launch(2, 30, graph=False), _activity(0, "outside_eager", 140, 7, correlation=2),
        _launch(3, -10, graph=False), _activity(0, "unmeasured1", -5, 1, correlation=3),
        _activity(0, "unmeasured2", -3, 1, correlation=3),
        _launch(4, 500), _activity(99999, "unmeasured_graph", 510, 2, correlation=4),
        _launch(5, 600),  # Even a graph launch with no activity is counted outside steps.
        _launch(6, 40, graph=False), _activity(0, "Memcpy", 150, 1000, correlation=6, category="gpu_memcpy"),
    ])
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.outside_kernel_counts == {0: 2}
    assert result.unattributed_launch_count == 3
    assert len(result.intervals) == 6
    assert result.layer_windows == {(0, 4): (100.0, 230.0)}


def test_scope_ranges_without_kernels_are_reported():
    result = attribute_replay_trace([_step(), _scope(EMPTY, 10, 20)], CaptureRecord(), CONTAINERS)
    assert result.scope_times == {ScopeKey(0, *EMPTY): ScopeKernelTime(0.0, 0)}
    assert result.outside_kernel_counts == {0: 0}
    assert result.layer_windows == {}


def test_graph_memset_is_an_interval_and_dependency_not_a_cost():
    record = _record([_node(10, owner=INNER, kind="memset"), _node(11, "kernel", dependencies=(10,))])
    events = [_step(), _launch(1, 10), _activity(1010, "Memset", 100, 40, category="gpu_memset"),
              _activity(1011, "kernel", 145, 10)]
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.scope_times[ScopeKey(0, *INNER)] == ScopeKernelTime(0.010, 1)
    assert result.intervals[0].activity == "memset"
    assert result.intervals[1].graph_gap_us == 5.0


def test_latest_dependency_end_over_several_streams():
    record = _record([_node(10, "one"), _node(11, "two"),
                      _node(12, kind="event_record", dependencies=(10, 11)),
                      _node(13, "three", dependencies=(12,))])
    events = [_step(), _launch(1, 10), _activity(1010, "one", 50, 60),
              _activity(1011, "two", 80, 70, stream=8), _activity(1013, "three", 160, 10, stream=9)]
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.layer_windows[(0, 4)] == (50.0, 170.0)
    assert result.intervals[-1].graph_gap_us == 10.0


@pytest.mark.parametrize("ranges,message", [
    ([_step(duration=100), _step(1, 50, 100)], "Overlapping step"),
    ([_step(), _scope(LAYER, 10, 100), _scope(INNER, 50, 100)], "Partially overlapping scope"),
    ([_step(), _scope(LAYER, 10, 100), _scope(INNER, 10, 100)], "Ambiguous identical scope"),
])
def test_invalid_ranges_raise(ranges, message):
    with pytest.raises(ValueError, match=message):
        attribute_replay_trace(ranges, CaptureRecord(), CONTAINERS)


def test_thread_indices_use_pid_and_tid_and_step_seams():
    events = [
        _step(0, 0, 100), _step(1, 100, 100),
        _scope(INNER, 10, 80, step=0), _scope(PROJECTION, 100, 100, step=1),
        _scope(LAYER, 10, 180, tid=2),  # Another thread cannot own tid=1 launches.
        _launch(1, 50, graph=False), _activity(0, "one", 60, 10),
        _launch(2, 100, graph=False), _activity(0, "two", 110, 10, correlation=2),
        _launch(3, 150, graph=False, pid=11), _activity(0, "other_pid", 160, 10, correlation=3),
    ]
    result = attribute_replay_trace(events, CaptureRecord(), CONTAINERS)
    assert result.scope_times[ScopeKey(0, *INNER)].kernel_count == 1
    assert result.scope_times[ScopeKey(1, *PROJECTION)].kernel_count == 1
    assert result.unattributed_launch_count == 1


def test_wrong_scope_step_is_rejected():
    events = [_step(), _scope(INNER, 10, 100, step=1), _launch(1, 20, graph=False), _activity(0, "one", 60, 10)]
    with pytest.raises(ValueError, match="disagrees with scope annotation"):
        attribute_replay_trace(events, CaptureRecord(), CONTAINERS)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "driver_graph"])
def test_graph_launch_correlation_requires_one_runtime_graph_launch(full, fault):
    record, events = full
    if fault == "missing":
        events.pop(1)
    elif fault == "duplicate":
        events.append(deepcopy(events[1]))
    else:
        events[1]["cat"] = "cuda_driver"
    with pytest.raises(ValueError, match="Expected one cudaGraphLaunch.*incomplete trace or ambiguous mapping"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_eager_launch_correlation_requires_one_launch():
    events = [_step(), _activity(0, "one", 60, 10)]
    with pytest.raises(ValueError, match="Expected one eager launch"):
        attribute_replay_trace(events, CaptureRecord(), CONTAINERS)


def test_missing_activity_correlation_fails(full):
    record, events = full
    del events[2]["args"]["correlation"]
    with pytest.raises(ValueError, match="has no launch correlation"):
        attribute_replay_trace(events, record, CONTAINERS)


def test_clone_chains_resolve_and_unknown_originals_fail(full):
    record, events = full
    record = replace(record, clones=record.clones + (NodeClone(2010, 1010),))
    events[2]["args"]["graph node id"] = 2010
    assert attribute_replay_trace(events, record, CONTAINERS).scope_times[ScopeKey(0, *INNER)].kernel_count == 1
    unknown = replace(record, clones=record.clones + (NodeClone(3000, 9999),))
    with pytest.raises(ValueError, match="unknown original node id"):
        attribute_replay_trace(events, unknown, CONTAINERS)


def test_dependency_traversal_handles_a_deep_event_chain():
    nodes = [_node(1, "root")]
    for index in range(2, 1200):
        nodes.append(_node(index, kind="empty", dependencies=(index - 1,)))
    nodes.append(_node(1200, "leaf", dependencies=(1199,)))
    record = replace(_record(nodes[:1]), nodes=tuple(nodes),
                     clones=tuple(NodeClone(node.node_id + 10000, node.node_id) for node in nodes))
    result = attribute_replay_trace([
        _step(), _launch(1, 10), _activity(10001, "root", 50, 10), _activity(11200, "leaf", 100, 10),
    ], record, CONTAINERS)
    assert result.intervals[-1].graph_gap_us == 40.0


@pytest.mark.parametrize("name", ["frontier_dsa_step", "frontier_dsa_step|x", "frontier_dsa_step|1|2", "frontier_dsa_step_extra|1"])
def test_malformed_step_annotation(name):
    with pytest.raises(ValueError, match="Malformed DSA step annotation"):
        parse_step_annotation(name)


def test_step_annotation_round_trip_and_unrelated_names():
    assert step_annotation_name(12) == "frontier_dsa_step|12"
    assert parse_step_annotation(step_annotation_name(12)) == 12
    assert parse_step_annotation(annotation_name(0, *INNER)) is None
    assert parse_step_annotation("other") is None
    assert parse_step_annotation(None) is None


@pytest.mark.parametrize("category,activity", [("gpu_memcpy", "memcpy"), ("gpu_memset", "memset")])
def test_eager_non_kernel_attribution_does_not_change_kernel_metrics(full, category, activity):
    record, events = full
    baseline = attribute_replay_trace(events, record, CONTAINERS)
    events += [
        _scope(LAYER, 20, 30), _scope(INNER, 25, 15),
        _launch(10, 30, graph=False), _activity(0, "owned copy", 140, 500, correlation=10, category=category),
        _launch(11, 55, graph=False), _activity(0, "outside copy", 150, 600, correlation=11, category=category),
        _launch(12, 500, graph=False), _activity(0, "unmeasured copy", 550, 10, correlation=12, category=category),
    ]
    result = attribute_replay_trace(events, record, CONTAINERS)
    assert result.scope_times == baseline.scope_times
    assert result.layer_windows == baseline.layer_windows
    assert result.outside_kernel_counts == baseline.outside_kernel_counts
    assert result.unattributed_launch_count == baseline.unattributed_launch_count + 1
    owned, outside = [row for row in result.intervals if row.kind == "eager"]
    assert (owned.owner, outside.owner) == (INNER, None)
    assert owned.activity == outside.activity == activity
    assert owned.capture is None and outside.capture is None
    assert (owned.launch_end_us, outside.launch_end_us) == (31.0, 56.0)
    assert all(row.capture == 0 and row.launch_end_us == 11.0 for row in result.intervals if row.kind == "graph")


def test_hold_is_an_unowned_dependency_with_shape_and_no_scope_cost():
    from frontier.profiling.attention.graph_replay_trace import HOLD_ACTIVITY,hold_annotation_name
    events=[_step(),_scope(LAYER,1,30),_annotation(hold_annotation_name(0),2,8),
            _launch(1,3,graph=False),_activity(0,'sleep',50,40),
            _scope(INNER,20,10),_launch(2,22,graph=False),_activity(0,'attention',91,5,correlation=2)]
    result=attribute_replay_trace(events,CaptureRecord(),CONTAINERS)
    hold=result.intervals[0]
    assert hold.activity==HOLD_ACTIVITY and hold.owner is None
    assert (hold.grid,hold.block)==(GRID,BLOCK)
    assert result.outside_kernel_counts=={0:0}
    assert result.layer_windows=={(0,4):(91.0,96.0)}
    assert result.scope_times[ScopeKey(0,*INNER)]==ScopeKernelTime(0.005,1)
    # A hold under a declared container still has no owner and cannot violate A2.
    assert result.scope_times[ScopeKey(0,*LAYER)].kernel_count==0


@pytest.mark.parametrize('count',[0,2])
def test_hold_requires_exactly_one_kernel_launch(count):
    from frontier.profiling.attention.graph_replay_trace import hold_annotation_name
    events=[_step(),_annotation(hold_annotation_name(0),2,8)]
    for index in range(count):events.extend([_launch(index+1,3+index,graph=False),_activity(0,'sleep',50+index*20,10,correlation=index+1)])
    with pytest.raises(ValueError,match='exactly one kernel'):
        attribute_replay_trace(events,CaptureRecord(),())


@pytest.mark.parametrize('index,start,duration',[(1,2,8),(0,-1,8),(0,399,8)])
def test_hold_must_be_inside_the_same_step(index,start,duration):
    from frontier.profiling.attention.graph_replay_trace import hold_annotation_name
    with pytest.raises(ValueError,match='inside its step range'):
        attribute_replay_trace([_step(),_annotation(hold_annotation_name(index),start,duration)],CaptureRecord(),())


@pytest.mark.parametrize('field',['grid','block'])
@pytest.mark.parametrize('hold',[False,True])
def test_eager_and_hold_kernels_require_launch_dimensions(field,hold):
    from frontier.profiling.attention.graph_replay_trace import hold_annotation_name
    activity=_activity(0,'kernel',50,10)
    activity['args'].pop(field)
    events=[_step(),_launch(1,3,graph=False),activity]
    if hold:events.append(_annotation(hold_annotation_name(0),2,8))
    with pytest.raises(ValueError,match=f'requires {field}'):
        attribute_replay_trace(events,CaptureRecord(),())


def test_copies_do_not_keep_launch_dimensions():
    events=[_step(),_launch(1,3,graph=False),_activity(0,'copy',50,10,category='gpu_memcpy')]
    result=attribute_replay_trace(events,CaptureRecord(),())
    assert result.intervals[0].grid is None and result.intervals[0].block is None
