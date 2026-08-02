"""Minimal manifest-driven Main/Reference parity runner.

The runner intentionally owns orchestration only.  Numerical semantics remain
in :mod:`tests.e2e.pd_af_parity.harness` and simulator configuration remains in
the supplied argv arrays.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tests.e2e.pd_af_parity.cases import get_case_by_id
from tests.e2e.pd_af_parity.harness import (
    ParityCaseConfig,
    generate_report,
    report_to_markdown,
)


_TOP_LEVEL_KEYS = frozenset({"case_id", "main", "reference", "report_path"})
_BRANCH_KEYS = frozenset({"cwd", "argv", "output_dir"})


@dataclass(frozen=True)
class CommandSpec:
    """One immutable simulator invocation."""

    cwd: Path
    argv: tuple[str, ...]
    output_dir: Path


@dataclass(frozen=True)
class PairRunManifest:
    """Validated pair-run inputs and the resolved parity case."""

    case_config: ParityCaseConfig
    main: CommandSpec
    reference: CommandSpec
    report_path: Path


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _load_command(value: object, context: str) -> CommandSpec:
    payload = _require_mapping(value, context)
    keys = frozenset(payload)
    if keys != _BRANCH_KEYS:
        raise ValueError(
            f"{context} keys must be exactly {sorted(_BRANCH_KEYS)}, got {sorted(keys)}"
        )
    raw_argv = payload["argv"]
    if not isinstance(raw_argv, list) or not raw_argv:
        raise ValueError(f"{context}.argv must be a non-empty JSON array")
    if any(not isinstance(item, str) or not item for item in raw_argv):
        raise ValueError(f"{context}.argv must contain non-empty strings")
    return CommandSpec(
        cwd=Path(_require_string(payload["cwd"], f"{context}.cwd")),
        argv=tuple(raw_argv),
        output_dir=Path(
            _require_string(payload["output_dir"], f"{context}.output_dir")
        ),
    )


def load_manifest(path: str | Path) -> PairRunManifest:
    """Load and strictly validate a manual pair-run manifest."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {exc.msg}") from exc
    payload_map = _require_mapping(payload, "manifest")
    keys = frozenset(payload_map)
    if keys != _TOP_LEVEL_KEYS:
        raise ValueError(
            "manifest top-level keys must be exactly "
            f"{sorted(_TOP_LEVEL_KEYS)}, got {sorted(keys)}"
        )
    case_id = _require_string(payload_map["case_id"], "manifest.case_id")
    try:
        case_config = get_case_by_id(case_id)
    except ValueError as exc:
        raise ValueError(f"manifest.case_id is not a supported parity case: {case_id}") from exc
    return PairRunManifest(
        case_config=case_config,
        main=_load_command(payload_map["main"], "manifest.main"),
        reference=_load_command(payload_map["reference"], "manifest.reference"),
        report_path=Path(
            _require_string(payload_map["report_path"], "manifest.report_path")
        ),
    )


def _run_command(spec: CommandSpec, branch: str, report_path: Path) -> None:
    if not spec.cwd.is_dir():
        raise RuntimeError(f"{branch} cwd does not exist or is not a directory: {spec.cwd}")
    result = subprocess.run(
        list(spec.argv),
        cwd=spec.cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    stdout_path = report_path.with_suffix(f".{branch}.stdout.log")
    stderr_path = report_path.with_suffix(f".{branch}.stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{branch} simulator failed with exit code {result.returncode}; "
            f"stdout={stdout_path} stderr={stderr_path}"
        )
    if not spec.output_dir.is_dir():
        raise RuntimeError(
            f"{branch} simulator completed without output directory: {spec.output_dir}"
        )


def run_manifest(manifest: PairRunManifest) -> Path:
    """Run both simulators and write the existing comparator's Markdown report."""
    _run_command(manifest.main, "main", manifest.report_path)
    _run_command(manifest.reference, "reference", manifest.report_path)
    report = generate_report(
        manifest.case_config,
        str(manifest.main.output_dir),
        str(manifest.reference.output_dir),
    )
    manifest.report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.report_path.write_text(report_to_markdown(report), encoding="utf-8")
    return manifest.report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    report_path = run_manifest(manifest)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
