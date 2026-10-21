"""D21 topology and scheduling-policy checks; CPU fixtures are not native evidence."""

import json
from pathlib import Path

import pytest

from tests.integration.step5_pd import build_stage, canonical_stage, submit_native_pd
from tests.unit.test_step5_native_pd import stage_inputs


@pytest.mark.parametrize("case_id", canonical_stage.CASES)
def test_each_priority_case_uses_native_node_plan_and_final_resource_policy(tmp_path, case_id):
    inputs = stage_inputs(tmp_path)
    definition = canonical_stage.case_definition(json.loads(
        (canonical_stage.REPO / "tests/integration/step5_pd/canonical_run_h200.json").read_text()), case_id)
    definition["probe_helpers"] = inputs["definition"]["probe_helpers"]
    stage = tmp_path / "workspace/stage"
    config = build_stage.build(definition, stage=stage, workspace=stage.parent,
        output_dir=Path("/mnt/codesign-exp/ycfeng/d21_test"), job_tag="fixture", source={"commit": "fixture"},
        **{key: value for key, value in inputs.items() if key != "definition"})
    case = canonical_stage.CASES[case_id]
    rows = json.loads((stage / "preflight.json").read_text())
    assert len(rows) == case["replica"]
    assert [(r["role"], r["lane"]) for r in rows] == [
        (role, lane) for role in ("prefill", "decode") for lane in range(case["data_parallel_size"])]
    arguments = submit_native_pd.resource_arguments(config)
    assert arguments["preemptible"] is case["preemptible"]
    assert arguments["positive_tags"] == [case["gpu_type"]]
    assert config["charged_group"] == case["charged_group"]
    for row in rows:
        command = row["server_command"]
        assert command[command.index("--data-parallel-size") + 1] == str(case["data_parallel_size"])
        assert not any(key.startswith("NCCL_") for key in row["server_env_set"])
