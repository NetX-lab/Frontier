"""Artifact comparison for the refactor fidelity matrix.

The gate is exact equality.  A behavior-preserving refactor has no reason to
change a simulated number, so this module does not implement tolerances.  The
only normalization is the substitution of run-specific absolute paths, which
differ between the two checkouts by construction and are not simulator output.

Observed on ``1f694f7``: ``system_metrics.json``, ``request_metrics.csv``,
``op_precision_metadata.csv`` and ``frontier_stage_batch_ledger.jsonl`` carry no
timestamps, wall-clock durations, hostnames or paths, and ``config.json``
carries exactly one absolute path (the metrics output directory).  The
substitution list below is therefore small on purpose; anything it fails to
cover shows up as a difference rather than being silently accepted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MAX_REPORTED_DIFFERENCES_PER_ARTIFACT = 5


@dataclass(frozen=True)
class PathSubstitution:
    """One literal replaced by a stable token before comparison."""

    literal: str
    token: str


@dataclass(frozen=True)
class ArtifactDifference:
    artifact: str
    kind: str
    detail: str

    def as_record(self) -> dict:
        return {"artifact": self.artifact, "kind": self.kind, "detail": self.detail}


def substitutions_for(repo_root: Path, output_root: Path, label: str) -> list[PathSubstitution]:
    """Build the substitution list for one side of the comparison.

    Longer literals come first so that a nested path is replaced before its
    parent directory.
    """

    pairs = [
        (str(Path(output_root).resolve() / label), "<FIDELITY_LABEL_ROOT>"),
        (str(Path(output_root).resolve()), "<FIDELITY_OUTPUT_ROOT>"),
        (str(Path(repo_root).resolve()), "<REPO_ROOT>"),
    ]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return [PathSubstitution(literal, token) for literal, token in pairs]


def apply_substitutions(text: str, substitutions: Sequence[PathSubstitution]) -> str:
    for substitution in substitutions:
        text = text.replace(substitution.literal, substitution.token)
    return text


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_artifacts(directory: Path) -> list[str]:
    """Return the sorted relative paths of every file under ``directory``."""

    if not directory.is_dir():
        return []
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    )


def _read_text(path: Path, substitutions: Sequence[PathSubstitution]) -> str:
    return apply_substitutions(path.read_text(encoding="utf-8"), substitutions)


def _json_leaf_differences(
    baseline: Any, candidate: Any, path: str = ""
) -> list[str]:
    """Return human-readable descriptions of the first differing leaves."""

    differences: list[str] = []
    if type(baseline) is not type(candidate) and not (
        isinstance(baseline, (int, float)) and isinstance(candidate, (int, float))
    ):
        return [f"{path or '/'}: type {type(baseline).__name__} != {type(candidate).__name__}"]

    if isinstance(baseline, dict):
        baseline_keys = set(baseline)
        candidate_keys = set(candidate)
        for key in sorted(baseline_keys - candidate_keys):
            differences.append(f"{path}/{key}: missing in candidate")
        for key in sorted(candidate_keys - baseline_keys):
            differences.append(f"{path}/{key}: missing in baseline")
        for key in sorted(baseline_keys & candidate_keys):
            if len(differences) >= MAX_REPORTED_DIFFERENCES_PER_ARTIFACT:
                break
            differences.extend(
                _json_leaf_differences(baseline[key], candidate[key], f"{path}/{key}")
            )
        return differences[:MAX_REPORTED_DIFFERENCES_PER_ARTIFACT]

    if isinstance(baseline, list):
        if len(baseline) != len(candidate):
            return [f"{path or '/'}: list length {len(baseline)} != {len(candidate)}"]
        for index, (left, right) in enumerate(zip(baseline, candidate)):
            if len(differences) >= MAX_REPORTED_DIFFERENCES_PER_ARTIFACT:
                break
            differences.extend(_json_leaf_differences(left, right, f"{path}[{index}]"))
        return differences[:MAX_REPORTED_DIFFERENCES_PER_ARTIFACT]

    if baseline != candidate:
        return [f"{path or '/'}: {baseline!r} != {candidate!r}"]
    return []


def _compare_json(
    name: str, baseline: Path, candidate: Path, substitutions_pair: tuple[Sequence[PathSubstitution], Sequence[PathSubstitution]]
) -> list[ArtifactDifference]:
    baseline_subs, candidate_subs = substitutions_pair
    baseline_value = json.loads(_read_text(baseline, baseline_subs))
    candidate_value = json.loads(_read_text(candidate, candidate_subs))
    differences = _json_leaf_differences(baseline_value, candidate_value)
    return [ArtifactDifference(name, "content", detail) for detail in differences]


def _compare_jsonl(
    name: str, baseline: Path, candidate: Path, substitutions_pair: tuple[Sequence[PathSubstitution], Sequence[PathSubstitution]]
) -> list[ArtifactDifference]:
    baseline_subs, candidate_subs = substitutions_pair
    baseline_lines = _read_text(baseline, baseline_subs).splitlines()
    candidate_lines = _read_text(candidate, candidate_subs).splitlines()
    if len(baseline_lines) != len(candidate_lines):
        return [ArtifactDifference(
            name, "content",
            f"record count {len(baseline_lines)} != {len(candidate_lines)}",
        )]

    differences: list[ArtifactDifference] = []
    for index, (left, right) in enumerate(zip(baseline_lines, candidate_lines)):
        if left == right:
            continue
        leaf_differences = _json_leaf_differences(json.loads(left), json.loads(right))
        for detail in leaf_differences:
            differences.append(ArtifactDifference(name, "content", f"record {index}: {detail}"))
        if len(differences) >= MAX_REPORTED_DIFFERENCES_PER_ARTIFACT:
            break
    return differences[:MAX_REPORTED_DIFFERENCES_PER_ARTIFACT]


def _compare_csv(
    name: str, baseline: Path, candidate: Path, substitutions_pair: tuple[Sequence[PathSubstitution], Sequence[PathSubstitution]]
) -> list[ArtifactDifference]:
    baseline_subs, candidate_subs = substitutions_pair
    baseline_rows = list(csv.reader(io.StringIO(_read_text(baseline, baseline_subs))))
    candidate_rows = list(csv.reader(io.StringIO(_read_text(candidate, candidate_subs))))
    if len(baseline_rows) != len(candidate_rows):
        return [ArtifactDifference(
            name, "content",
            f"row count {len(baseline_rows)} != {len(candidate_rows)}",
        )]

    differences: list[ArtifactDifference] = []
    header = baseline_rows[0] if baseline_rows else []
    for row_index, (left_row, right_row) in enumerate(zip(baseline_rows, candidate_rows)):
        if left_row == right_row:
            continue
        if len(left_row) != len(right_row):
            differences.append(ArtifactDifference(
                name, "content",
                f"row {row_index}: column count {len(left_row)} != {len(right_row)}",
            ))
        else:
            for column_index, (left, right) in enumerate(zip(left_row, right_row)):
                if left == right:
                    continue
                column = (
                    header[column_index]
                    if row_index > 0 and column_index < len(header)
                    else f"column {column_index}"
                )
                differences.append(ArtifactDifference(
                    name, "content",
                    f"row {row_index} [{column}]: {left!r} != {right!r}",
                ))
                if len(differences) >= MAX_REPORTED_DIFFERENCES_PER_ARTIFACT:
                    break
        if len(differences) >= MAX_REPORTED_DIFFERENCES_PER_ARTIFACT:
            break
    return differences[:MAX_REPORTED_DIFFERENCES_PER_ARTIFACT]


def _compare_bytes(name: str, baseline: Path, candidate: Path) -> list[ArtifactDifference]:
    if baseline.read_bytes() == candidate.read_bytes():
        return []
    return [ArtifactDifference(
        name, "content",
        f"binary contents differ (sha256 {file_digest(baseline)[:16]} != {file_digest(candidate)[:16]})",
    )]


def compare_artifact_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    baseline_substitutions: Sequence[PathSubstitution],
    candidate_substitutions: Sequence[PathSubstitution],
    ignored_artifacts: Iterable[str] = (),
) -> list[ArtifactDifference]:
    """Compare every artifact under the two directories.

    A file present on one side only is a difference; so is any content
    difference that survives path substitution.
    """

    ignored = set(ignored_artifacts)
    baseline_names = [name for name in list_artifacts(baseline_dir) if name not in ignored]
    candidate_names = [name for name in list_artifacts(candidate_dir) if name not in ignored]

    differences: list[ArtifactDifference] = []
    for name in sorted(set(baseline_names) - set(candidate_names)):
        differences.append(ArtifactDifference(name, "missing_in_candidate", "produced only by the baseline"))
    for name in sorted(set(candidate_names) - set(baseline_names)):
        differences.append(ArtifactDifference(name, "missing_in_baseline", "produced only by the candidate"))

    substitutions_pair = (baseline_substitutions, candidate_substitutions)
    for name in sorted(set(baseline_names) & set(candidate_names)):
        baseline_path = baseline_dir / name
        candidate_path = candidate_dir / name
        suffix = Path(name).suffix.lower()
        try:
            if suffix == ".json":
                differences.extend(_compare_json(name, baseline_path, candidate_path, substitutions_pair))
            elif suffix == ".jsonl":
                differences.extend(_compare_jsonl(name, baseline_path, candidate_path, substitutions_pair))
            elif suffix == ".csv":
                differences.extend(_compare_csv(name, baseline_path, candidate_path, substitutions_pair))
            else:
                differences.extend(_compare_bytes(name, baseline_path, candidate_path))
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as error:
            differences.append(ArtifactDifference(
                name, "unreadable", f"{type(error).__name__}: {error}",
            ))
    return differences
