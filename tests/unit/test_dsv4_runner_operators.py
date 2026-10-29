"""CPU acceptance contracts for the B13r native runner producer."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.operators.families import KV_CACHE_FAMILY, DEEPSEEK_V4_RUNNER_FAMILY
from frontier.profiling.attention import dsa_vllm_runner as cases
from frontier.profiling.attention import dsv4_graph_replay_runner as replay
from frontier.profiling.attention.dsv4_step_scopes import STEP_LAYER_ID, STEP_SCOPES, is_step_scope
from frontier.profiling.attention.graph_capture import GraphCaptureRecorder
from frontier.profiling.kernel_gap import dsv4_replay_gaps as gaps
from frontier.profiling.model_runner import dsv4_block_operators as direct
from frontier.profiling.model_runner import dsv4_runner_rows as rows
from test_dsv4_replay_gaps import interval, records

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner"
CONFIG = ROOT / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json"


@pytest.fixture
def grid():
    return direct.load_grid(PACKAGE / "block_operator_grid.json", json.loads(CONFIG.read_text()))


@pytest.fixture
def fake_runner(monkeypatch):
    recorder = GraphCaptureRecorder(None, lambda owner, fn: fn())
    monkeypatch.setattr(replay, "_RECORDER", recorder)
    annotations, live, kernels = [], [], []

    @contextmanager
    def annotate(name):
        live.append(name)
        annotations.append(name)
        try:
            yield
        finally:
            live.pop()

    state = SimpleNamespace(active=SimpleNamespace(step_index=0),
                            case=SimpleNamespace(measurement="KERNEL_ONLY"), hold_cycles=0,
                            torch=SimpleNamespace(profiler=SimpleNamespace(record_function=annotate)))

    def timed(layer, scope, fn, args, kwargs):
        with state.torch.profiler.record_function(f"frontier_dsa|0|{layer}|{scope}"):
            return fn(*args, **kwargs)

    state.timed_call = timed
    monkeypatch.setattr(cases, "_STATE", state)

    def kernel(name):
        kernels.append((name, recorder.current_owner, tuple(live)))
        return name

    runner = SimpleNamespace(
        kv_block_zeroer=SimpleNamespace(zero_block_ids=lambda ids: kernel("zero")),
        block_tables=SimpleNamespace(apply_staged_writes=lambda: kernel("write")),
        model_state=SimpleNamespace(preprocess_state=lambda: kernel("preprocess"),
                                    prepare_attn=lambda: kernel("state_attn"), prepare_inputs=lambda: kernel("state_inputs")),
        model=SimpleNamespace(compute_logits=lambda hidden: kernel("logits"), forward=lambda: kernel("forward")),
        prepare_inputs=lambda: kernel("inputs"), prepare_attn=lambda: kernel("attn"),
        postprocess_sampled=lambda: kernel("postprocess"),
    )

    def sample():
        kernel("row_gather")
        runner.model.compute_logits("hidden")
        return kernel("sampler")

    runner.sample = sample
    replay._wrap_runner_step_scopes(runner)
    return runner, state, recorder, kernels, annotations


@pytest.mark.parametrize("member,attribute,scope", replay._RUNNER_STEP_CALLS)
def test_every_native_bound_callable_has_a_step_owner(fake_runner, member, attribute, scope):
    runner, _, recorder, kernels, _ = fake_runner
    owner = runner if member is None else getattr(runner, member)
    parent = dict(replay.SCOPE_TREE)[scope]
    with recorder.scope(None if parent is None else (STEP_LAYER_ID, parent)):
        getattr(owner, attribute)([1]) if attribute in ("zero_block_ids", "compute_logits") else getattr(owner, attribute)()
    assert kernels[0][1] == (STEP_LAYER_ID, scope)
    assert recorder.current_owner is None
    assert parent is None or scope == "dsv4_compute_logits" and parent == "dsv4_sampling"


def test_sampling_owns_its_kernels_around_the_nested_logits(fake_runner):
    runner, _, recorder, kernels, annotations = fake_runner
    assert runner.sample() == "sampler"
    assert [item[1] for item in kernels] == [(-1, "dsv4_sampling"), (-1, "dsv4_compute_logits"), (-1, "dsv4_sampling")]
    assert annotations == ["frontier_dsa|0|-1|dsv4_sampling", "frontier_dsa|0|-1|dsv4_compute_logits"]
    assert recorder.current_owner is None


def test_logits_outside_sampling_fails(fake_runner):
    runner, _, _, _, _ = fake_runner
    with pytest.raises(ValueError, match="expected parent"):
        runner.model.compute_logits("hidden")


def test_step_scope_inside_another_fails(fake_runner):
    runner, _, recorder, _, _ = fake_runner
    with recorder.scope((-1, "dsv4_sampling")):
        with pytest.raises(ValueError, match="expected parent None"):
            runner.prepare_inputs()
        with pytest.raises(ValueError, match="expected parent None"):
            runner.sample()


def test_step_opens_at_execute_and_closes_after_sample_tokens(fake_runner):
    runner, state, _, _, _ = fake_runner
    boundaries = []
    state.open_step_range = lambda name: boundaries.append(("open", name))
    state.close_step_range = lambda: boundaries.append(("close", None))
    state.capture_metadata = lambda key, fn: None
    runner.execute_model_state = SimpleNamespace(attn_metadata={})
    runner.execute_model = lambda: boundaries.append(("execute", None))
    runner.sample_tokens = lambda: boundaries.append(("tokens", None))
    replay._wrap_step(runner, SimpleNamespace(modules={}))
    replay._wrap_sample_tokens(runner)
    runner.execute_model()
    assert [name for name, _ in boundaries] == ["open", "execute"]
    runner.sample_tokens()
    assert [name for name, _ in boundaries] == ["open", "execute", "tokens", "close"]


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE", "NONE"])
def test_step_intervals_keep_unowned_gap_classes_and_values(mode):
    launch = "eager" if mode == "NONE" else "graph"
    capture = None if mode == "NONE" else "graph" if mode == "FULL" else "breakable_segment"
    source = records(mode, [interval(0, 1), interval(2, 3, activity="memcpy"),
                           interval(4, 5, scope="dsv4_q_projection", launch=launch, capture_kind=capture),
                           interval(6, 7)])
    changed = [dict(row, scope="dsv4_input_preparation", global_layer_id=-1) if row["scope"] is None else row
               for row in source.intervals]
    extra = dict(kind="call", engine_index=0, case_id="case", step_index=0, global_layer_id=-1,
                 scope="dsv4_input_preparation", time_ms=.002, kernel_count=2)
    candidate = replace(source, intervals=tuple(changed), calls=(*source.calls, extra))
    before, after = gaps.activity_gaps([source]), gaps.activity_gaps([candidate])
    assert [(x.segment, x.dependency_end_us, x.gap_us, x.attention_owned) for x in before] == [
        (x.segment, x.dependency_end_us, x.gap_us, x.attention_owned) for x in after]
    assert gaps.reduce_dsv4_gaps([source]).diagnostics == gaps.reduce_dsv4_gaps([candidate]).diagnostics


@pytest.mark.parametrize("scope,layer", [("dsv4_sampling", 0), ("other", -1), ("dsv4_sampling", None)])
def test_malformed_step_owners_fail(scope, layer):
    with pytest.raises(ValueError, match="Step scope requires owner"):
        is_step_scope(dict(scope=scope, global_layer_id=layer))


def test_interval_classification_and_unowned_grid_census():
    items = [interval(0, 1, scope="dsv4_q_projection"),
             dict(interval(2, 3), scope="dsv4_sampling", global_layer_id=-1),
             interval(4, 5, launch="graph"), dict(interval(6, 7), model_forward=True),
             dict(interval(8, 9, activity="memcpy", name="output copy"), grid=[2, 2, 1])]
    step = dict(case_id="case", step_index=0, phase="decode", cudagraph_dispatch={"cg_mode": "FULL"})
    result = rows.ownership_report([step], items)
    assert result["steps"][0]["counts"] == dict(attention_scope=1, step_scope=1, model_forward=2, unowned=1)
    assert result["case_kinds"]["FULL"]["unowned"] == [dict(activity="memcpy", name="output copy", grid=[2, 2, 1], count=1)]


@pytest.mark.parametrize("phase,case,mode,kind", [("decode", "a", "FULL", "FULL"),
    ("decode", "b", "PIECEWISE", "PIECEWISE"), ("first_tail", "first_tail", "PIECEWISE", "first_tail"),
    ("first_tail", "multi_tail", "PIECEWISE", "multi_tail"), ("dummy", "a", "FULL", "dummy"),
    ("prefill", "a", "NONE", "prefill")])
def test_case_kind_census(phase, case, mode, kind):
    assert rows.case_kind(dict(phase=phase, case_id=case, cudagraph_dispatch={"cg_mode": mode})) == kind


def test_grid_equals_expected_rows_and_native_entries(grid):
    expected = json.loads((PACKAGE / "expected_rows.json").read_text())
    key = lambda row: json.dumps(row, sort_keys=True)
    assert sorted(map(key, grid.rows())) == sorted(map(key, expected["path_b_rows"]))
    assert len(grid.rows()) == 394
    assert grid.entries(grid.layout["block_sizes"]) == [24, 96, 96, 1536, 768]


@pytest.mark.parametrize("field,value,error", [("zero_block_counts", [2521], "free blocks"),
    ("new_requests", [129], "request rows"), ("routed_expert_count", 5, "routing top-k")])
def test_invalid_grids_fail_capacity_or_routing(grid, tmp_path, field, value, error):
    payload = dict(grid.payload, **{field: value})
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        direct.load_grid(path, json.loads(CONFIG.read_text()))


@pytest.mark.parametrize("w,r", [(0, 1), (1, 0), (2, 2), (5, 4), (32, 4)])
def test_native_staging_counts_continuing_pairs_and_full_rows(grid, w, r):
    entries = grid.entries(grid.layout["block_sizes"])
    staged = direct.write_staging_plan(w, r, entries)
    observed = []
    tables = SimpleNamespace(append_block_ids=lambda row, ids, overwrite: observed.append((row, ids, overwrite)))
    direct.stage_writes(tables, staged)
    assert observed == list(staged)
    assert sum(bool(values) for _, ids, _ in staged for values in ids) == w + len(entries) * r
    assert sum(len(values) for _, ids, _ in staged for values in ids) == w + sum(entries) * r
    pairs = [(row, group) for row, ids, _ in staged for group, values in enumerate(ids) if values]
    assert len(pairs) == len(set(pairs))
    assert all(overwrite is (index >= w) for index, (_, _, overwrite) in enumerate(staged))
    assert sum(bool(values) for _, ids, _ in direct.initial_writes(staged) for values in ids) == w


def test_registry_targets_leave_execution_disabled():
    operators = (*KV_CACHE_FAMILY.operators, *DEEPSEEK_V4_RUNNER_FAMILY.operators)
    assert {op.name for op in operators} == set(STEP_SCOPES) - {"dsv4_compute_logits"}
    assert all(op.profiling_target and not op.predictor_target and not op.e2e_trace_target for op in operators)
    assert not KV_CACHE_FAMILY.execution_enabled and not DEEPSEEK_V4_RUNNER_FAMILY.execution_enabled


def test_package_check_writes_no_destination():
    spec = importlib.util.spec_from_file_location("b13r_builder", PACKAGE / "prepare_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((PACKAGE / "package_plan.json").read_text())
    before = (PACKAGE / plan["destination"]).exists()
    result = module.check_plan(plan)
    assert result["path_a_cases_per_engine"] == {"0": 686, "1": 296, "2": 7}
    assert result["path_b_total_rows"] == 394 and result["package_built"] is False
    assert result["phases"] == ["G2H_RUNNER_REPLAY", "G2H_BLOCK_OPERATORS"]
    assert plan["destination"] == "package_03"
    jobs = result["jobs"]
    assert [job["name"] for job in jobs] == ["G2H_V4_RUNNER_REPLAY", "G2H_V4_BLOCK_OPERATORS"]
    assert [job["job_tag"] for job in jobs] == ["v4-h200-g2h-runner-20261029-03-replay",
                                              "v4-h200-g2h-runner-20261029-03-block"]
    assert [job["bootstrap_calls"] for job in jobs] == [1, 1]
    assert len({job["output_dir"] for job in jobs}) == 2
    assert all(len(job["phases"]) == 1 for job in jobs)
    commands = [job["phases"][0]["steps"][0]["commands"][0]["argv"] for job in jobs]
    assert [Path(argv[-1]).name for argv in commands] == ["bridge_config.json", "block_config.json"]
    # The bridge driver selects the V2 runner for its replay children; the
    # block entry runs the prefill entry directly, which requires it.
    assert commands[0][0] == "{python}"
    assert commands[1][:3] == ["env", f"{replay.V2_MODEL_RUNNER_ENV}=1", "{python}"]
    assert (PACKAGE / plan["destination"]).exists() == before


def test_failing_logits_restores_ownership(fake_runner):
    runner, state, recorder, _, _ = fake_runner
    original = state.timed_call

    def timed(layer, scope, fn, args, kwargs):
        if scope == "dsv4_compute_logits":
            raise RuntimeError("logits failed")
        return original(layer, scope, fn, args, kwargs)

    state.timed_call = timed
    with pytest.raises(RuntimeError, match="logits failed"):
        runner.sample()
    assert recorder.current_owner is None


def test_forward_membership_uses_correlations_not_names():
    events = [dict(cat="user_annotation", name=rows.MODEL_FORWARD_PREFIX+"0", ts=10, dur=20, pid=1, tid=1),
              dict(cat="cuda_runtime", name="same kernel", ts=12, pid=1, tid=1, args={"correlation":11}),
              dict(cat="cuda_runtime", name="same kernel", ts=32, pid=1, tid=1, args={"correlation":12}),
              dict(cat="cuda_driver", ts=12, pid=1, tid=2, args={"correlation":13}),
              dict(cat="cuda_runtime", ts=12, pid=1, tid=1, args={})]
    assert rows.forward_correlations(events) == {11}
    assert rows.forward_correlations(events[1:]) is None


@pytest.mark.parametrize("field,value", [("segment_count",30), ("chunks_per_page",9),
                                        ("block_sizes",[256,64,64,8,8])])
def test_native_layout_mismatch_fails_before_measurement(grid, field, value):
    layout = dict(grid.layout, **{field:value})
    with pytest.raises(ValueError, match="Native layout"):
        direct.check_layout(grid, layout, json.loads(CONFIG.read_text()))


def test_replay_report_keeps_step_rows_separate_from_attention(tmp_path):
    engine = dict(kind="engine", engine_index=0, runner_layout={"segment_count":30})
    step = dict(kind="step", case_id="case", step_index=0, phase="decode", cudagraph_dispatch={"cg_mode":"FULL"})
    call = dict(kind="call", case_id="case", step_index=0, scope="dsv4_sampling", global_layer_id=-1, time_ms=.003)
    activity = dict(interval(0,3), scope="dsv4_sampling", global_layer_id=-1)
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(item)+"\n" for item in [engine,step,call,activity]))
    rows.write_replay_report(path)
    assert json.loads(path.with_suffix(".ownership.json").read_text())["steps"][0]["counts"]["step_scope"] == 1
    import csv
    row = list(csv.DictReader(path.with_suffix(".runner.csv").open()))[0]
    assert row["schema_version"] == rows.ROW_SCHEMA
    assert row["row_role"] == rows.COST
    assert set(rows.REPLAY_ROW_ROLES) == set(STEP_SCOPES)
    assert row["time_stats.dsv4_sampling.median"] == "0.003"
    assert row["memcpy_stats.dsv4_sampling.median"] == "0.0"


def test_engine_builder_forwards_expert_override_without_other_changes(monkeypatch, grid):
    from frontier.profiling.attention import dsv4_attention_runner as builder
    spec = json.loads((PACKAGE / "package_plan.json").read_text())
    args = [*spec["plan_args"], "--pilot_cases", str(PACKAGE / spec["pilot_cases"]),
            "--engine_policy", "graph_replay", "--cudagraph_mode", spec["runs"][0]["mode"],
            "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"])]
    plan = direct.direct_engine_plan(args, CONFIG, grid)
    observed=[]
    @contextmanager
    def finalizers():
        yield
    monkeypatch.setattr(builder,"patch_dummy_weight_finalizers",finalizers)
    monkeypatch.setattr(cases,"build_engine",lambda changed,*args,**kwargs: observed.append(changed) or "engine")
    result = builder.build_dsv4_engine(plan,0,"dummy",0,.85,routed_expert_count=8)
    assert result=="engine"
    assert observed[0].hf_overrides == dict(plan.hf_overrides,n_routed_experts=8)
    assert "n_routed_experts" not in plan.hf_overrides



def test_signed_step_owner_survives_the_actual_trace_reader():
    from frontier.profiling.attention.dsa_kernel_trace import ScopeKey
    from frontier.profiling.attention.graph_capture import CaptureRecord
    from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace
    from test_graph_replay_trace import _step, _scope, _launch, _activity
    events = [_step(duration=50), _scope((-1, "dsv4_input_preparation"), 1, 10),
              _launch(1, 2, graph=False), _launch(2, 3, graph=False),
              _activity(None, "Memcpy HtoD", 19, 1, correlation=1, category="gpu_memcpy", graph_id=0),
              _activity(None, "arbitrary kernel name", 20, 3, correlation=2, graph_id=0)]
    attributed = attribute_replay_trace(events, CaptureRecord(), frozenset())
    assert len(attributed.intervals) == 2
    assert all(item.owner == (-1, "dsv4_input_preparation") for item in attributed.intervals)
    sample = attributed.scope_times[ScopeKey(0, -1, "dsv4_input_preparation")]
    assert sample.kernel_count == 1
    assert sample.kernel_time_ms == .003


def test_trace_reader_gives_nested_logits_kernels_to_the_logits_scope():
    from frontier.profiling.attention.dsa_kernel_trace import ScopeKey
    from frontier.profiling.attention.graph_capture import CaptureRecord
    from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace
    from test_graph_replay_trace import _step, _scope, _launch, _activity
    sampling, logits = (-1, "dsv4_sampling"), (-1, "dsv4_compute_logits")
    events = [_step(duration=100), _scope(sampling, 1, 30), _scope(logits, 5, 10),
              _launch(1, 2, graph=False), _launch(2, 6, graph=False), _launch(3, 20, graph=False),
              _activity(None, "row gather", 40, 1, correlation=1, graph_id=0),
              _activity(None, "lm head", 41, 5, correlation=2, graph_id=0),
              _activity(None, "sampler", 46, 2, correlation=3, graph_id=0)]
    attributed = attribute_replay_trace(events, CaptureRecord(), replay.CONTAINER_SCOPES)
    assert [item.owner for item in attributed.intervals] == [sampling, logits, sampling]
    assert attributed.scope_times[ScopeKey(0, *sampling)].kernel_count == 2
    assert attributed.scope_times[ScopeKey(0, *logits)].kernel_count == 1


@pytest.fixture
def native_order_runner(monkeypatch):
    """Real recorder/timer with a native V2 execute-then-sample stand-in."""
    events = []

    @contextmanager
    def annotation(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    torch = SimpleNamespace(
        profiler=SimpleNamespace(record_function=annotation),
        cuda=SimpleNamespace(_sleep=lambda cycles: events.append(("hold", cycles))),
    )
    state = cases._WorkerState(0, torch, ())
    state.next_step_index = 7
    state.hold_cycles = 123
    monkeypatch.setattr(cases, "_STATE", state)
    monkeypatch.setattr(replay, "_RECORDER", GraphCaptureRecorder(None, lambda owner, fn: fn()))
    noop = lambda *args, **kwargs: None
    runner = SimpleNamespace(
        kv_block_zeroer=SimpleNamespace(zero_block_ids=noop),
        block_tables=SimpleNamespace(apply_staged_writes=noop),
        model_state=SimpleNamespace(preprocess_state=noop, prepare_attn=noop, prepare_inputs=noop),
        model=SimpleNamespace(compute_logits=noop, forward=noop),
        prepare_inputs=noop, prepare_attn=noop, postprocess_sampled=noop,
        execute_model_state=None,
    )

    def execute_model(scheduler_output, dummy_run=False):
        if not dummy_run:
            runner.kv_block_zeroer.zero_block_ids([1])
            runner.block_tables.apply_staged_writes()
            runner.prepare_inputs()
            runner.prepare_attn()
            runner.model_state.preprocess_state()
        runner.model_state.prepare_attn()
        runner.model_state.prepare_inputs()
        runner.execute_model_state = SimpleNamespace(attn_metadata={})
        return None

    def sample():
        runner.model.compute_logits("hidden")
        return "tokens"

    def sample_tokens():
        assert runner.execute_model_state is not None
        runner.execute_model_state = None
        result = runner.sample()
        runner.postprocess_sampled()
        return result

    runner.execute_model = execute_model
    runner.sample = sample
    runner.sample_tokens = sample_tokens
    replay._wrap_step(runner, SimpleNamespace(modules={}))
    replay._wrap_sample_tokens(runner)
    replay._wrap_runner_step_scopes(runner)
    # Same installation order as install_graph_replay_hooks: classifier outside step.
    cases._wrap_execute_model(runner)
    return runner, state, events


def _native_schedule(state, kind):
    phase = "first_tail" if kind in ("first_tail", "multi_tail") else "prefill" if kind == "prefill" else "decode"
    batch_size = 2 if kind == "multi_tail" else 1
    state.case = cases._CaseRecorder(kind, phase, batch_size, 0, 1, "KERNEL_ONLY",
                                    prompt_len=8, tail_count=batch_size)
    computed = 7 if phase == "first_tail" else 0 if phase == "prefill" else 8
    return SimpleNamespace(
        finished_req_ids=set(),
        num_scheduled_tokens={str(i): 1 for i in range(batch_size)},
        scheduled_new_reqs=[SimpleNamespace(req_id=str(i), prompt_token_ids=list(range(8)),
                                           num_computed_tokens=computed) for i in range(batch_size)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
    )


@pytest.mark.parametrize("kind", ["FULL", "PIECEWISE", "first_tail", "multi_tail", "prefill"])
def test_native_order_records_sampling_after_execute_returns(native_order_runner, kind):
    runner, state, events = native_order_runner
    schedule = _native_schedule(state, kind)
    assert runner.execute_model(schedule) is None
    assert runner.sample_tokens() == "tokens"
    step = "frontier_dsa_step|7"
    hold = "frontier_dsa_hold|7"
    names = ["kv_cache_block_zero", "dsv4_block_table_write"] + ["dsv4_input_preparation"] * 5
    expected = [("enter", step), ("enter", hold), ("hold", 123), ("exit", hold)]
    for scope in names:
        name = f"frontier_dsa|7|-1|{scope}"
        expected += [("enter", name), ("exit", name)]
    sampling, logits = "frontier_dsa|7|-1|dsv4_sampling", "frontier_dsa|7|-1|dsv4_compute_logits"
    expected += [("enter", sampling), ("enter", logits), ("exit", logits), ("exit", sampling),
                 ("enter", sampling), ("exit", sampling), ("exit", step)]
    assert events == expected
    assert state.active is None and state.step_range is None
    assert state.case.recorded == 1 and state.case.steps[0]["step_index"] == 7
    state.close_step_range()
    assert events == expected and state.case.recorded == 1
    # A later call outside the recorded window must create no range or scopes.
    assert runner.execute_model(schedule) is None
    assert runner.sample_tokens() == "tokens"
    assert events == expected and state.case.recorded == 1


def test_native_dummy_closes_once_without_sampling(native_order_runner):
    runner, state, events = native_order_runner
    state.case = cases._CaseRecorder("dummy", "dummy", 1, 0, 1, "KERNEL_ONLY")

    def dummy_batch():
        assert runner.execute_model(None, dummy_run=True) is None
        # Native _dummy_run consumes execute_model_state and gathers hidden rows,
        # then returns; it never calls sample_tokens or sample.
        assert state.active is not None and state.case.recorded == 0
        runner.execute_model_state = None

    cases.run_dummy_step(SimpleNamespace(execute_dummy_batch=dummy_batch))
    assert events.count(("enter", "frontier_dsa_step|7")) == 1
    assert events.count(("exit", "frontier_dsa_step|7")) == 1
    assert not any("sampling" in str(name) or "compute_logits" in str(name) for _, name in events)
    assert state.active is None and state.step_range is None and state.case.recorded == 1
    state.close_step_range()
    assert events.count(("exit", "frontier_dsa_step|7")) == 1
