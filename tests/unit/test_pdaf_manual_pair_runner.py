"""Unit tests for the minimal manual pd-af parity runner."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e.pd_af_parity import runner
from tests.e2e.pd_af_parity.runner import PairRunManifest, load_manifest


def _manifest_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "case_id": "C1",
        "main": {
            "cwd": str(tmp_path / "main"),
            "argv": ["python", "-m", "frontier.main"],
            "output_dir": str(tmp_path / "main-output"),
        },
        "reference": {
            "cwd": str(tmp_path / "reference"),
            "argv": ["python", "-m", "frontier.main"],
            "output_dir": str(tmp_path / "reference-output"),
        },
        "report_path": str(tmp_path / "report.md"),
    }


def test_load_manifest_requires_exact_top_level_keys(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["unexpected"] = True
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level keys"):
        load_manifest(manifest_path)


def test_load_manifest_builds_typed_manifest_and_resolves_case(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, PairRunManifest)
    assert manifest.case_config.case_id == "C1"
    assert manifest.main.argv == ("python", "-m", "frontier.main")
    assert manifest.reference.output_dir == tmp_path / "reference-output"


def test_load_manifest_rejects_empty_argv(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["main"]["argv"] = []  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="argv"):
        load_manifest(manifest_path)


def test_run_manifest_executes_both_commands_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest_payload(tmp_path)
    for branch in ("main", "reference"):
        Path(payload[branch]["cwd"]).mkdir()  # type: ignore[index]
        Path(payload[branch]["output_dir"]).mkdir()  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(argv: list[str], *, cwd: Path, check: bool, text: bool, capture_output: bool):
        calls.append((tuple(argv), cwd))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "generate_report", lambda *args: object())
    monkeypatch.setattr(runner, "report_to_markdown", lambda report: "# PASS\n")

    report_path = runner.run_manifest(manifest)

    assert report_path == Path(payload["report_path"])
    assert report_path.read_text(encoding="utf-8") == "# PASS\n"
    assert [cwd for _, cwd in calls] == [manifest.main.cwd, manifest.reference.cwd]
    assert report_path.with_suffix(".main.stdout.log").read_text(encoding="utf-8") == "ok\n"


def test_run_manifest_fails_fast_on_nonzero_simulator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _manifest_payload(tmp_path)
    Path(payload["main"]["cwd"]).mkdir()  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_manifest(manifest_path)

    def fake_run(argv: list[str], **_: object):
        return subprocess.CompletedProcess(argv, 7, stdout="out", stderr="err")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 7"):
        runner.run_manifest(manifest)
