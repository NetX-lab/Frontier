from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COLOCATION_ROOT = REPO_ROOT / "examples" / "architecture" / "co-location"
OFFLINE_CASES = {
    "dense_model_basic.sh",
    "moe_model_basic.sh",
    "thinking_mode_basic.sh",
    "moe_spec_dec.sh",
    "moe_prefix_caching.sh",
}
ONLINE_CASES = {
    "dense_model_basic_online.sh",
    "moe_model_basic_online.sh",
    "thinking_mode_basic_online.sh",
    "moe_spec_dec_online.sh",
    "moe_prefix_caching_online.sh",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_colocation_examples_are_split_into_offline_and_online_cases() -> None:
    offline_dir = COLOCATION_ROOT / "offline"
    online_dir = COLOCATION_ROOT / "online"

    assert offline_dir.is_dir()
    assert online_dir.is_dir()

    assert {path.name for path in offline_dir.glob("*.sh")} == OFFLINE_CASES
    assert {path.name for path in online_dir.glob("*.sh")} == ONLINE_CASES

    stale_top_level_cases = {
        path.name
        for path in COLOCATION_ROOT.glob("*.sh")
        if path.name != "run_all.sh"
    }
    assert stale_top_level_cases == set()


def test_every_case_declares_the_expected_simulation_mode() -> None:
    for case_name in OFFLINE_CASES:
        text = _read(COLOCATION_ROOT / "offline" / case_name)
        assert "--simulation_mode offline" in text

    for case_name in ONLINE_CASES:
        text = _read(COLOCATION_ROOT / "online" / case_name)
        assert "--simulation_mode online" in text


def test_colocation_run_all_lists_all_cases_once() -> None:
    run_all = COLOCATION_ROOT / "run_all.sh"
    text = _read(run_all)

    for case_name in sorted(OFFLINE_CASES):
        assert text.count(f"offline/{case_name}") == 1
    for case_name in sorted(ONLINE_CASES):
        assert text.count(f"online/{case_name}") == 1


def test_colocation_run_all_fails_when_case_filter_matches_no_cases() -> None:
    run_all = COLOCATION_ROOT / "run_all.sh"
    env = os.environ.copy()
    env["CASE_FILTER"] = "__no_colocation_case_matches_this_filter__"

    completed = subprocess.run(
        ["bash", str(run_all)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode != 0
    assert "CASE_FILTER matched no co-location cases" in completed.stdout


def test_examples_documentation_points_to_new_layout() -> None:
    docs = [
        REPO_ROOT / "examples" / "README.md",
        REPO_ROOT / "examples" / "architecture" / "README.md",
    ]
    joined = "\n".join(_read(path) for path in docs)

    assert "examples/architecture/co-location/offline/dense_model_basic.sh" in joined
    assert "examples/architecture/co-location/online/dense_model_basic_online.sh" in joined
    assert "examples/architecture/co-location/run_all.sh" in joined
    assert "not profiling fidelity" in joined


def test_case_scripts_do_not_reference_stale_top_level_paths() -> None:
    stale_paths = {
        "examples/architecture/co-location/dense_model_basic.sh",
        "examples/architecture/co-location/moe_model_basic.sh",
        "examples/architecture/co-location/thinking_mode_basic.sh",
        "examples/architecture/co-location/moe_spec_dec.sh",
        "examples/architecture/co-location/moe_prefix_caching.sh",
    }

    for case_path in sorted((COLOCATION_ROOT / "offline").glob("*.sh")) + sorted((COLOCATION_ROOT / "online").glob("*.sh")):
        text = _read(case_path)
        for stale_path in stale_paths:
            assert stale_path not in text, f"{case_path} still references stale path {stale_path}"


def test_every_case_has_one_click_analytical_backend_default() -> None:
    for case_path in sorted((COLOCATION_ROOT / "offline").glob("*.sh")) + sorted((COLOCATION_ROOT / "online").glob("*.sh")):
        text = _read(case_path)
        assert 'CC_BACKEND="${CC_BACKEND:-analytical}"' in text
        assert '--cc_backend_config_type "$CC_BACKEND"' in text
