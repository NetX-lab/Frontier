"""CPU checks against retained native states and independent scheduler examples."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys

import pytest

from frontier.profiling.attention import dsa_vllm_runner as runner
from frontier.profiling.attention.dsa_case_prediction import predict_case
from frontier.profiling.attention.dsa_plan import DsaEngineSpec, DsaMeasurementLimits, build_pilot_cases
from frontier.profiling.attention.v4_attention_cases import _contexts, _place, build_attention_cases
from frontier.profiling.attention.v4_attention_windows import (
    C1, assign_windows, describe_step, matching_profiles, max_writes, required_windows, serving_classes,
)
from v4_attention_coverage import check_steps, native_coverage, plan_from_spec, prediction_steps, read_steps

ROOT = Path(__file__).resolve().parents[2]
H3O = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs"
P1H_RAW = Path("/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/P1H_BRIDGE/bridge/FULL_AND_PIECEWISE/raw")


@pytest.fixture
def limits():
    return DsaMeasurementLimits(7168, 267665, 128, (8192,), 256, 2, 8, 1, 3,
                                "graph_replay", "FULL_AND_PIECEWISE", (1, 2, 4, 8, 16, 24), 200000)


@pytest.fixture
def explicit():
    return {"engines": [{"engine_index": 0, "prefill_query_len": 0},
                        {"engine_index": 1, "prefill_query_len": 6143, "chunk_budget": 24576}],
            "cases": [{"case_id": "heterogeneous", "engine_index": 0, "phase": "decode",
                       "batch_size": 2, "prompt_len": 6144, "prompt_lens": [6144, 6200],
                       "warmup": 0, "measured": 2, "dp_peer": None}]}


def test_fcfs_heterogeneous_contexts_and_one_token_advance(explicit, limits):
    engines, cases = build_pilot_cases(limits, explicit)
    states = predict_case(cases[0], engines[0], limits)
    assert [[r["num_computed_tokens"] for r in s["requests"]] for s in states] == [[6145, 6200], [6146, 6201]]
    assert cases[0].request_prompt_lens == (6144, 6200)
    assert cases[0].max_tokens == 5
    assert cases[0].estimated_steps == 4


@pytest.mark.parametrize("key,value,message", [
    ("prompt_lens", [6144], "batch_size"), ("prompt_lens", "6144", "JSON list"),
    ("prompt_lens", [True, 6144], "positive int"), ("prompt_lens", [0, 6144], "positive int"),
    ("prompt_lens", [7168, 6144], "max_context_len"),
    ("tail_count", 1, "only valid"), ("warmup", -1, "non-negative"),
    ("warmup", True, "non-negative"), ("measured", 0, "positive"),
    ("measured", 1.5, "positive"), ("extra", 0, "unknown keys"),
])
def test_strict_extended_case_keys(explicit, limits, key, value, message):
    explicit["cases"][0][key] = value
    with pytest.raises(ValueError, match=message):
        build_pilot_cases(limits, explicit)


@pytest.mark.parametrize("key,value", [("chunk_budget", True), ("chunk_budget", 0),
                                        ("chunk_budget", 127), ("prefill_query_len", 24577)])
def test_strict_engine_bounds(explicit, limits, key, value):
    explicit["engines"][1][key] = value
    with pytest.raises(ValueError):
        build_pilot_cases(limits, explicit)


def test_capacity_bound(explicit, limits):
    with pytest.raises(ValueError, match="max_kv_tokens_per_lane"):
        build_pilot_cases(replace(limits, max_kv_tokens_per_lane=12000), explicit)


def test_infeasible_window_hits_native_context_limit(explicit, limits):
    explicit["cases"][0].update(prompt_lens=[7166, 7166], measured=8)
    with pytest.raises(ValueError, match="stop advancing"):
        build_pilot_cases(limits, explicit)


def tail_payload(explicit):
    row = explicit["cases"][0]
    row.update(engine_index=1, phase="first_tail", batch_size=3, prompt_len=6144,
               prompt_lens=[6500, 6144, 6144], tail_count=2, warmup=0, measured=1,
               dp_peer={"num_tokens": 8, "cg_mode": "PIECEWISE", "uniform_token_count": None, "max_query_len": 1})
    return explicit


def test_multi_tail_same_chunk_step_and_forced_piecewise_width(explicit, limits):
    payload = tail_payload(explicit)
    engines, cases = build_pilot_cases(limits, payload)
    step, = predict_case(cases[0], engines[1], limits)
    assert [r["num_computed_tokens"] for r in step["requests"]] == [6501, 6143, 6143]
    assert step["cudagraph_dispatch"] == {"cg_mode": "PIECEWISE", "num_tokens": 8,
                                           "num_reqs": None, "num_tokens_across_dp": [8, 8]}
    point = describe_step(step)
    assert point["first_tails"] == [1, 2]
    assert point["writers"] == {"4": [1, 2], "128": [1, 2]}
    with pytest.raises(ValueError, match="chunks exceed chunk budget"):
        payload["engines"][1]["chunk_budget"] = 8192
        build_pilot_cases(limits, payload)


@pytest.mark.parametrize("mutate", ["one_tail", "unfinished_chunk", "wrong_context", "two_tokens", "valid"])
def test_begin_step_matches_the_own_multi_tail_definition(explicit, limits, mutate, monkeypatch):
    engines, cases = build_pilot_cases(limits, tail_payload(explicit))
    case = cases[0]
    step, = predict_case(case, engines[1], limits)
    rows = deepcopy(step["requests"])
    if mutate == "one_tail": rows[2]["num_computed_tokens"] += 1
    if mutate == "unfinished_chunk": rows[2]["num_computed_tokens"] -= 1
    if mutate == "wrong_context": rows[0]["prompt_len"] -= 1
    if mutate == "two_tokens": rows[0]["num_scheduled_tokens"] = 2
    state = runner._WorkerState(1, None, ())
    state.case = runner._CaseRecorder(case.case_id, case.phase, case.batch_size, 0, 1,
                                      "KERNEL_ONLY", {"num_tokens": 8}, case.prompt_len,
                                      case.prompt_lens, case.tail_count)
    monkeypatch.setattr(state, "_scheduled_requests", lambda _: rows)
    active = state.begin_step(None)
    assert (active is not None) == (mutate == "valid")
    if active is not None:
        assert active.record["requests"] == rows
        assert state.matched_dp_peer == {"num_tokens": 8}


@pytest.fixture(scope="module")
def p1h():
    plan = plan_from_spec(H3O / "p1h_bridge/package_plan.json")
    if not P1H_RAW.exists():
        pytest.skip("retained P1h package 03 GPU evidence is not available on this host")
    records = list(read_steps([P1H_RAW]))
    assert len(records) == 175
    return plan, records


@pytest.mark.parametrize("case_index", range(25))
def test_every_package03_case_against_native_records(p1h, case_index):
    plan, records = p1h
    case = plan.cases[case_index]
    observed = [r for _, _, r in records if r["case_id"] == case.case_id]
    expected = predict_case(case, plan.engine(case.engine_index), plan.limits)
    assert len(observed) == len(expected)
    for got, wanted in zip(observed, expected):
        assert got["requests"] == wanted["requests"]
        assert got["cudagraph_dispatch"] == wanted["cudagraph_dispatch"]


def test_package03_plan_json_frozen(p1h):
    plan, _ = p1h
    frozen = json.loads((H3O / "p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json").read_text())
    assert json.loads(json.dumps(plan.to_json())) == frozen
    assert (json.dumps(plan.to_json(), indent=2, sort_keys=True) + "\n").encode() == (
        H3O / "p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json").read_bytes()


def test_checker_reports_native_prediction_and_missing_windows(p1h):
    plan, records = p1h
    report = check_steps(plan, records)
    assert report["prediction_matches"]
    assert not report["grid_complete"]
    assert report["cases_seen"] == 25
    assert report["measured_steps"] == 175
    assert report["outside_steps"]
    assert any(s["case_id"] == "e0-decode-b4-p7150" for s in report["outside_steps"])


def test_checker_locates_changed_request_and_dispatch(p1h):
    plan, records = p1h
    changed = deepcopy(records)
    _, _, step = next(row for row in changed if row[2]["phase"] == "decode")
    step["requests"][0]["num_computed_tokens"] += 1
    step["cudagraph_dispatch"]["num_tokens"] = 32
    report = check_steps(plan, changed)
    assert {m["field"] for m in report["mismatches"]} == {"requests", "cudagraph_dispatch"}
    assert all(m["case_id"] == step["case_id"] and m["line"] > 0 for m in report["mismatches"])


def test_checker_missing_extra_and_unknown_case(p1h):
    plan, records = p1h
    report = check_steps(plan, records[1:])
    assert not report["prediction_matches"]
    assert report["missing_steps"]
    record = deepcopy(records[-1])
    record[2]["case_id"] = "unknown"
    report = check_steps(plan, [*records, record])
    assert report["mismatches"][-1]["reason"].startswith("unknown")
    report = check_steps(plan, [*records, records[-1]])
    assert report["mismatches"][-1]["reason"].endswith("extra measured step")


def test_stream_reader_skips_call_records(tmp_path):
    file = tmp_path / "dsv4_raw_e0.jsonl"
    file.write_text('{"kind":"call"}\n{"kind":"step","case_id":"a"}\n')
    rows = list(read_steps([tmp_path]))
    assert rows == [(str(file), 2, {"kind": "step", "case_id": "a"})]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No dsv4_raw"):
        list(read_steps([empty]))


def test_all_native_classes_and_multi_tails_in_grid():
    result = native_coverage([H3O / "evidence/coverage" / f"{name}.json" for name in ("p2_32", "p2_64", "p0")])
    assert result["classes"] == 145
    assert result["outside"] == 0
    assert [r["inside"] for r in result["cases"]] == [43, 54, 48]


def test_grid_and_compression_table_are_c1_union():
    assert len(serving_classes()) == 39
    assert len(required_windows()) == 1041
    assert max_writes(4) == (4, 4)
    assert max_writes(8) == (7, 4)
    assert max_writes(13) == (6, 1)
    assert max_writes(24) == (7, 4)
    assert len([w for w in required_windows() if w.category == "dummy"]) == 13


def test_generated_targets_complete_amended_grid(limits):
    payload, report = build_attention_cases(limits)
    engines, cases = build_pilot_cases(limits, payload)
    assert len(report["windows"]) == 1041
    assert report["complete"]
    assert report["context_tolerance"] == max(C1["ratios"]) - 1
    assert report["grid_decision"] == "D-G1H-GRID-AMENDMENT"
    assert "context_tolerance_status" not in report
    assert all(w["mix"][0] >= w["tails"] and w["mix"][1] >= w["tails"]
               for w in report["windows"] if w["tails"])
    assert sum(c.tail_count > 1 for c in cases) == 7
    assert engines[2].chunk_budget == max(t for _, t in C1["multi_tail"]) * C1["prefill_query"]
    for case in cases:
        assert case.warmup == 0 and case.measured == 1
        for step in predict_case(case, engines[case.engine_index], limits):
            assert assign_windows(step)


def test_g1h_check_predictions_has_no_outside_step():
    plan = plan_from_spec(H3O / "g1h/package_plan.json")
    records = (("predicted", i, step) for i, step in enumerate(prediction_steps(plan), 1))
    report = check_steps(plan, records, predicted=True)
    assert report["prediction_matches"]
    assert not report["outside_steps"]
    assert report["grid_complete"]
    assert report["window_counts"] == {"predicted": 1041}
    assert report["measured_steps"] == len(plan.cases)


@pytest.mark.parametrize("phase", ["decode", "first_tail"])
def test_runner_submits_explicit_prompts_and_simultaneous_tails(explicit, limits, monkeypatch, tmp_path, phase):
    if phase == "first_tail": tail_payload(explicit)
    engines, cases = build_pilot_cases(limits, explicit)
    case = cases[0]
    requests, submissions, batches = {}, [], []
    recorded = 0
    def add(request_id, prompt, params):
        length = len(prompt["prompt_token_ids"])
        submissions.append(length)
        requests[request_id] = {"length": length, "tokens": params["max_tokens"], "steps": 0}
    def step():
        nonlocal recorded
        batches.append(list(requests))
        outputs = []
        for request_id, request in list(requests.items()):
            request["steps"] += 1
            finished = request["tokens"] == 1 and request["steps"] == 2
            outputs.append(SimpleNamespace(request_id=request_id, outputs=[SimpleNamespace(token_ids=[1])], finished=finished))
            if finished: requests.pop(request_id)
        recorded += 1
        return outputs
    def rpc(fn, args=()):
        nonlocal recorded
        if fn is runner.begin_case:
            recorded = 0
            assert args[-2:] == (case.prompt_lens, case.tail_count)
        if fn is runner.case_progress: return [recorded]
        if fn is runner.finish_case: return [{"recorded": case.measured, "steps": [], "calls": []}]
        return [None]
    engine = SimpleNamespace(add_request=add, step=step, has_unfinished_requests=lambda: bool(requests),
                             abort_request=lambda ids: [requests.pop(i, None) for i in ids])
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    plan = SimpleNamespace(engine=lambda _: engines[case.engine_index])
    runner.run_case(SimpleNamespace(llm_engine=engine, collective_rpc=rpc), plan, case,
                    "KERNEL_ONLY", random.Random(1), 10, tmp_path, True, lambda *args: [])
    assert submissions == list(case.prompt_lens)
    if phase == "first_tail": assert any(len(batch) == case.batch_size for batch in batches)


def test_package_builder_reuses_p1h_without_mutating_it(monkeypatch, tmp_path):
    file = H3O / "g1h/prepare_package.py"
    spec = importlib.util.spec_from_file_location("test_g1h_prepare", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.package_builder()
    assert builder.HERE == file.parent
    assert builder.ROOT == ROOT
    called = []
    monkeypatch.setattr(module, "package_builder", lambda: SimpleNamespace(main=lambda: called.append(True)))
    module.main()
    assert called == [True]
    # The incomplete-grid gate still operates without a stale blocker list.
    monkeypatch.setattr(module, "HERE", tmp_path)
    (tmp_path / "package_plan.json").write_text(json.dumps({"coverage_grid": "grid.json"}))
    (tmp_path / "grid.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(RuntimeError, match="planning is incomplete"):
        module.main()
    assert called == [True]


@pytest.mark.parametrize("key,value", [("prompt_len", 1), ("dp_peer", None), ("prompt_lens", [1])])
def test_explicit_dummy_retains_original_bounds(explicit, limits, key, value):
    row = explicit["cases"][0]
    row.update(phase="dummy", batch_size=1, prompt_len=0, prompt_lens=[0],
               dp_peer={"num_tokens": 1, "cg_mode": "FULL", "uniform_token_count": 1, "max_query_len": 1})
    row[key] = value
    with pytest.raises(ValueError, match="dummy"):
        build_pilot_cases(limits, explicit)


def test_explicit_prefill_uses_submitted_length_for_repeat_windows(explicit, limits):
    limits = replace(limits, chunk_budgets=(1024,))
    explicit["engines"] = explicit["engines"][:1]
    explicit["cases"][0].update(phase="prefill", batch_size=1, prompt_len=1024,
                                 prompt_lens=[2049], warmup=1, measured=1)
    engines, cases = build_pilot_cases(limits, explicit)
    case = cases[0]
    assert runner._case_window(case, engines[0]) == (3, 3)
    states = predict_case(case, engines[0], limits)
    assert [step["requests"][0]["num_computed_tokens"] for step in states] == [0, 1024, 2048]
    assert case.estimated_steps == 6


def test_checker_requires_the_defined_first_tail_context(explicit, limits):
    engines, cases = build_pilot_cases(limits, tail_payload(explicit))
    step, = predict_case(cases[0], engines[1], limits)
    step["cudagraph_dispatch"]["num_tokens"] = 4
    step["requests"][0].update(num_computed_tokens=6144, prompt_len=6144)
    assert assign_windows(step)
    # This tail is still decode-shaped, but its cached context is not C1's 6143.
    step["requests"][1].update(num_computed_tokens=6271, prompt_len=6272)
    assert not assign_windows(step)


def test_checker_requires_uncached_prefill_with_the_full_query():
    step = {"phase": "prefill", "requests": [{"prompt_len": 6144,
            "num_computed_tokens": 0, "num_scheduled_tokens": 6144}],
            "cudagraph_dispatch": {"cg_mode": "NONE", "num_tokens": 6144, "num_reqs": 1}}
    assert assign_windows(step)
    # Same descriptor and compression mix do not qualify a different cached/query state.
    step["requests"][0].update(num_computed_tokens=1, num_scheduled_tokens=6143)
    assert not assign_windows(step)


def test_checker_rejects_a_query_one_chunk_that_is_still_prefilling():
    step = {"phase": "decode", "requests": [{"prompt_len": 6144,
            "num_computed_tokens": 6140, "num_scheduled_tokens": 1}],
            "cudagraph_dispatch": {"cg_mode": "PIECEWISE", "num_tokens": 1, "num_reqs": None}}
    assert not assign_windows(step)


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
def test_context_phases_fit_the_last_fcfs_construction_step(limits, mode):
    window = next(w for w in required_windows() if w.category == mode and w.width == 24
                  and w.own == 24 and w.profile == "low" and w.mix == (7, 0))
    engine = DsaEngineSpec(0, 8192, 128, 7168, 256, 0)
    contexts = _contexts(window, engine.chunk_budget)
    assert sorted(contexts) == [6144] * 15 + [6145, 6146] + [6147] * 7
    assert sum(contexts) % engine.chunk_budget == 24
    row = _place(window, contexts, engine, "budget_boundary")
    engines, cases = build_pilot_cases(limits, {"engines": [{"engine_index": 0, "prefill_query_len": 0}],
                                                "cases": [row]})
    step, = predict_case(cases[0], engines[0], limits)
    assert window in assign_windows(step)


@pytest.mark.parametrize("width,own", serving_classes())
def test_tail_mix_amendment_deduplicates_in_grid_order(width, own):
    k4, k128 = max_writes(own)
    for profile in C1["profiles"]:
        windows = [w for w in required_windows() if (w.category, w.width, w.own, w.profile)
                   == ("first_tail", width, own, profile)]
        expected = list(dict.fromkeys(((1, 1), (k4, 1), (k128, k128))))
        assert [w.mix for w in windows] == expected
    # Multi-tail singleton windows already carry their mandatory writes.
    assert all(w.mix == (w.tails, w.tails) for w in required_windows() if w.tails > 1)


def test_amendment_removed_only_duplicate_first_tail_windows():
    assert len(required_windows()) == len(set(required_windows()))
    original_count = len(serving_classes()) * len(C1["profiles"]) * 3 * 3
    original_count += len(C1["multi_tail"]) + len(C1["widths"]) * 2 + 2
    assert original_count - len(required_windows()) == 33


def continuing_rows(contexts):
    return [{"prompt_len": C1["cached_low"], "num_computed_tokens": c,
             "num_scheduled_tokens": 1} for c in contexts]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("distance,near", [(127, True), (128, False)])
def test_spread_profile_allowance_boundary_in_both_directions(sign, distance, near):
    low, high = C1["cached_low"], C1["cached_high"]
    middle = round((low + high) / 2)
    rows = continuing_rows([low, middle + sign * distance, high])
    assert ("spread" in matching_profiles(rows)) == near


@pytest.mark.parametrize("profile,context,near", [
    ("low", 6271, True), ("low", 6272, False),
    ("high", 7039, True), ("high", 7038, False),
])
def test_low_high_allowance_boundaries(profile, context, near):
    assert (profile in matching_profiles(continuing_rows([context]))) == near


@pytest.mark.parametrize("context", [6143, 7167])
def test_continuing_context_outside_native_range_has_no_profile(context):
    rows = continuing_rows([context])
    rows[0]["prompt_len"] = context
    assert matching_profiles(rows) == ()


def test_context_allowance_has_only_the_shared_table_setting(monkeypatch):
    monkeypatch.setitem(C1, "context_tolerance", 0)
    assert "low" in matching_profiles(continuing_rows([C1["cached_low"]]))
    assert "low" not in matching_profiles(continuing_rows([C1["cached_low"] + 1]))


@pytest.mark.parametrize("width,own,profile,mix,expected_contexts,expected_writers", [
    (8, 6, "low", (5, 1), [6146, 6147, 6147, 6147, 6147, 6143],
     {"4": [1, 2, 3, 4, 5], "128": [5]}),
    (16, 6, "low", (5, 1), [6146, 6147, 6147, 6147, 6147, 6143],
     {"4": [1, 2, 3, 4, 5], "128": [5]}),
    (24, 18, "high", (1, 1), [7166] * 17 + [6143], {"4": [17], "128": [17]}),
    (24, 18, "high", (7, 1), [7163] * 6 + [7166] * 11 + [6143],
     {"4": [0, 1, 2, 3, 4, 5, 17], "128": [17]}),
    (24, 18, "high", (4, 4), [7039] * 3 + [7166] * 14 + [6143],
     {"4": [0, 1, 2, 17], "128": [0, 1, 2, 17]}),
])
def test_each_residual_first_tail_reaches_its_window(
        limits, width, own, profile, mix, expected_contexts, expected_writers):
    window = next(w for w in required_windows() if
                  (w.category, w.width, w.own, w.profile, w.mix)
                  == ("first_tail", width, own, profile, mix))
    engine = DsaEngineSpec(1, limits.chunk_budgets[0], limits.max_num_seqs,
                           limits.max_context_len, limits.block_size, C1["prefill_query"] - 1)
    row = _place(window, _contexts(window, engine.chunk_budget), engine, "residual")
    engines, cases = build_pilot_cases(limits, {
        "engines": [{"engine_index": 0, "prefill_query_len": 0},
                    {"engine_index": 1, "prefill_query_len": engine.prefill_query_len}],
        "cases": [row]})
    step, = predict_case(cases[0], engines[1], limits)
    assert window in assign_windows(step)
    assert [r["num_computed_tokens"] for r in step["requests"]] == expected_contexts
    assert all(r["num_scheduled_tokens"] == 1 for r in step["requests"])
    assert describe_step(step)["writers"] == expected_writers
