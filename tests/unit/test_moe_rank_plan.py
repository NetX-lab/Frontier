"""CPU checks for the G3h SP-local gate and rank-class planning contract."""

import json
import runpy
import sys
from itertools import permutations
from pathlib import Path

import pytest

from frontier.moe_ep_workload import materialize_balanced_rank_classes
from frontier.profiling.moe import main as moe_main
from frontier.profiling.moe.moe_rank_plan import (
    distinct_lane_permutations, parse_dp_lane_local_rows, plan_dp_lane_rows,
)

ROOT = Path(__file__).resolve().parents[2]
G3H = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h"
SPEC = json.loads((G3H / "package_plan.json").read_text())
EXPECTED = json.loads((G3H / "expected_tasks.json").read_text())["tasks"]


def task_key(task):
    return [task.expert_parallel_size, task.num_tokens,
            [task.local_routed_rows, task.local_experts_with_extra_row, task.base_rows_per_expert],
            task.gating_num_tokens, task.ep_rank]


@pytest.mark.parametrize("topology", SPEC["topologies"])
def test_g3h_exact_classes_tasks_and_lowest_rank(topology):
    ep = topology["expert_parallel_size"]
    vectors = [tuple(rows) for rows in topology["dp_lane_local_rows"]]
    tasks = plan_dp_lane_rows(vectors, expert_parallel_size=ep, num_experts=256, router_topk=6)
    selected = [row for row in EXPECTED if row[0] == ep]
    assert [task_key(task) for task in tasks] == selected
    assert len(tasks) == (13 if ep == 16 else 20)
    for rows in vectors:
        gathered = ep // len(rows) * sum(rows)
        classes = materialize_balanced_rank_classes(
            routing_token_count=gathered, router_topk=6, total_expert_num=256,
            moe_expert_parallel_size=ep,
        )
        assert set(classes) == {tuple(row[2]) for row in selected if row[1] == gathered}
        for placement in set(permutations(rows)):
            for rank, rank_class in enumerate(classes):
                local = placement[rank // (ep // len(rows))]
                representatives = [task for task in tasks if task.num_tokens == gathered
                    and (task.local_routed_rows, task.local_experts_with_extra_row,
                         task.base_rows_per_expert) == rank_class and task.gating_num_tokens == local]
                assert len(representatives) == 1
                assert representatives[0].ep_rank <= rank
    # Repeated vectors and their permutations cannot add a measurement.
    repeated = vectors + [placement for rows in vectors for placement in set(permutations(rows))]
    assert plan_dp_lane_rows(repeated, expert_parallel_size=ep, num_experts=256, router_topk=6) == tasks


@pytest.mark.parametrize("rows", [(768, 1, 1, 1), (768, 768, 1, 1), (1,) * 16])
def test_distinct_permutations(rows):
    actual = list(distinct_lane_permutations(rows))
    expected = {rows} if len(rows) == 16 else set(permutations(rows))
    assert actual == sorted(expected)


@pytest.mark.parametrize("value", ["", "1,", "1,x", "0,1", "-1,1"])
def test_lane_row_parser_rejects_invalid_vectors(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_dp_lane_local_rows(value)


@pytest.mark.parametrize("extra", [
    ["--num_tokens_list", "16"], ["--extra_num_tokens", "16"],
    ["--expert_parallel_sizes", "16", "32"], ["--num_tensor_parallel_workers", "2"],
    ["--load_distributions", "uniform"], ["--disable_load_imbalance"],
    ["--routing_runtime_path", "uniform_topk"], ["--expert_parallel_sizes", "3"],
])
def test_lane_mode_rejects_incompatible_arguments(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["moe.main", *SPEC["common_args"],
        "--expert_parallel_sizes", "16", "--dp_lane_local_rows", "1,1", *extra])
    with pytest.raises(SystemExit) as error:
        moe_main.parse_args()
    assert error.value.code == 2


def test_lane_mode_requires_an_explicit_ep_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["moe.main", *SPEC["common_args"],
        "--dp_lane_local_rows", "1,1"])
    with pytest.raises(SystemExit) as error:
        moe_main.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("method", SPEC["methods"])
@pytest.mark.parametrize("topology", SPEC["topologies"])
def test_lane_dry_run_is_the_expected_task_list(monkeypatch, capsys, method, topology):
    builder = runpy.run_path(str(G3H / "prepare_package.py"))
    monkeypatch.setattr(sys, "argv", ["moe.main", *builder["topology_args"](SPEC, topology),
        "--profile_method", method, "--dry_run"])
    moe_main.main()
    output = capsys.readouterr().out
    plan = json.loads(output[output.index("{"):])
    tasks = plan["models"][SPEC["model"]]["rank_tasks"]
    ep = topology["expert_parallel_size"]
    assert [builder["task_key"](task) for task in tasks] == [row for row in EXPECTED if row[0] == ep]
    assert plan["cases_per_parallel_point"] == 1
    assert plan["models"][SPEC["model"]]["profiled_steps"] == len(tasks)


def test_builder_checks_all_four_plans_without_building():
    builder = runpy.run_path(str(G3H / "prepare_package.py"))
    plans = builder["check_plan"](SPEC, ROOT)
    assert set(plans) == {f"{method}_ep{ep}" for method in SPEC["methods"] for ep in (16, 32)}
    assert not (G3H / SPEC["destination"]).exists()


def test_builder_detects_a_wrong_representative_rank(monkeypatch):
    builder = runpy.run_path(str(G3H / "prepare_package.py"))
    import subprocess
    from types import SimpleNamespace
    tasks = [{"expert_parallel_size": row[0], "num_tokens": row[1],
              "local_routed_rows": row[2][0], "local_experts_with_extra_row": row[2][1],
              "base_rows_per_expert": row[2][2], "gating_num_tokens": row[3], "ep_rank": row[4]}
             for row in EXPECTED if row[0] == 16]
    tasks[0]["ep_rank"] = 7
    plan = {"models": {SPEC["model"]: {"rank_tasks": tasks, "profiled_steps": len(tasks)}}}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(plan)))
    with pytest.raises(ValueError, match="rank tasks differ"):
        builder["check_plan"](SPEC, ROOT)
