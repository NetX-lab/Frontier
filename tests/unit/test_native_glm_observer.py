"""Check paired GLM orchestration at the existing replay entry point."""
import json
from pathlib import Path

from tests.comparison.calibration.native_glm_observer import run_cases


def test_cases_share_inputs_and_clean_does_not_receive_an_observer(tmp_path):
    cases = [dict(name="c0", mode="clean"),
             dict(name="l0", mode="light", observer_config="l0.json"),
             dict(name="l1", mode="light", observer_config="l1.json"),
             dict(name="c1", mode="clean")]
    config = dict(output_dir=str(tmp_path), replay_cases=cases,
                  startup_timeout_seconds=1200, drain_seconds=3)
    calls = []
    def replay(argv):
        values = dict(zip(argv[::2], argv[1::2]))
        assert values["--engine-config"] == str(tmp_path / "engine.json")
        assert values["--trace-dir"] == str(tmp_path / "trace")
        assert values["--model-dir"] == str(tmp_path / "model")
        case = cases[len(calls)]
        assert values["--output-dir"] == str(tmp_path / case["name"])
        assert ("--observer-config" in values) == (case["mode"] == "light")
        if case["mode"] == "light":
            assert Path(values["--observer-config"]).name == case["observer_config"]
        calls.append(values)
        return 0
    assert run_cases(tmp_path, config, replay) == 0
    assert [c["--mode"] for c in calls] == ["clean", "light", "light", "clean"]
    assert len(json.loads((tmp_path / "case_results.json").read_text())) == 4
    calls.clear()
    assert run_cases(tmp_path, config, lambda argv: calls.append(argv) or 4) == 4
    assert len(calls) == 1
    assert json.loads((tmp_path / "case_results.json").read_text()) == [
        dict(case="c0", mode="clean", returncode=4)]
