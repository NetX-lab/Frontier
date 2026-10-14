#!/usr/bin/env python3
"""Structural checks for the frontier-calibration skill: catalog, routes, contract, paths, wording."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ATOMIC_NAMES = {
    "groundtruth-run-check",
    "groundtruth-run",
    "simulator-run",
    "semantic-alignment-check",
    "semantic-alignment-repair",
    "e2e-metrics-gap",
    "dispatch-align-trace",
    "workflow-gap-analysis",
    "workflow-repair",
    "op-gap-analysis",
    "op-supplement",
    "op-repair",
    "residual-lane-classification",
    "residual-repair",
    "cpu-overhead-analysis",
    "cpu-overhead-supplement",
    "cpu-overhead-repair",
}
COMPOSITE_NAMES = {"parity-run", "error-analysis", "error-correction"}
EXPECTED_LANES = {
    "pp": "pp_overhead",
    "collective": "collective_communication",
    "kv_or_m2n_transfer": "kv_or_m2n_transfer",
}
REQUIRED_REGISTRY_FIELDS = {
    "name",
    "kind",
    "owner",
    "inputs",
    "outputs",
    "allows_code_change",
    "stop_status",
    "validation",
}
REQUIRED_REFERENCES = {
    "contract.md",
    "evidence-tiers.md",
    "migration.md",
    "tool-boundaries.md",
}
EXPECTED_GROUNDTRUTH_BRANCH = "feature/frontier-comparison-instrumentation"
# The checked-in skill stays host-neutral; host bindings live in the untracked
# site.local.md (skills/README.md).
HOST_SPECIFIC_PATTERNS = (
    r"/data/",
    r"/mnt/",
    r"/home/",
    r"(?i)stepmind",
    r"(?i)charged[ _]group",
    r"step_main",
    r"codesign",
    r"i-fengyicheng",
    r"ycfeng",
)
REPOSITORY_PATH = re.compile(r"(?:`|\")((?:tests|frontier|docs|skills|data|examples)/[^`\"\s]*)")


def fail(message: str) -> None:
    raise AssertionError(message)


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    # Phrase checks below assert that a document states something, not how it
    # is wrapped. Markdown reflows across lines, so compare against a single
    # whitespace-normalized line.
    return " ".join(text.split())


def entry_doc(root: Path, name: str) -> Path:
    return root / "entries" / f"{name}.md"


def validate_entry_doc(path: Path, expected_name: str) -> None:
    # Entries are documents bundled inside the one skill, not skills of their
    # own: Claude Code only discovers a SKILL.md at the top of a skill
    # directory, so a nested one would never load. The trigger sentence that
    # used to live in frontmatter is now the "When to use" line.
    body = read_text(path)
    if not body.startswith("# "):
        fail(f"entry document must open with its title heading: {path}")
    trigger = re.search(r"^\*\*When to use:\*\*\s*(.+)$", body, re.MULTILINE)
    if not trigger:
        fail(f"entry document lacks a 'When to use' line: {path}")
    if not trigger.group(1).strip().startswith("Use when"):
        fail(f"trigger sentence must start with 'Use when': {path}")
    if path.stem != expected_name:
        fail(f"entry document name mismatch: {path}")


def run(root: Path) -> None:
    registry_path = root / "references" / "registry.json"
    if not registry_path.is_file():
        fail(f"missing fixed registry: {registry_path}")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid registry JSON: {exc}")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        fail("registry.entries must be a list")
    if registry.get("schema_version") != "frontier-calibration-v2":
        fail("registry schema_version is not frontier-calibration-v2")
    if "candidate dependency inventory" not in str(registry.get("calls_semantics", "")):
        fail("registry must label composite calls as a candidate dependency inventory")
    if registry.get("routing_authority") != "references/call-graph.md":
        fail("registry must point conditional routing to references/call-graph.md")
    lanes = registry.get("lane_registry")
    if not isinstance(lanes, list):
        fail("registry.lane_registry must be a list")
    lane_by_name = {}
    for lane in lanes:
        if not isinstance(lane, dict):
            fail(f"lane registry entry must be an object: {lane!r}")
        required_lane_fields = {
            "lane",
            "classification",
            "required_evidence",
            "frontier_owner_field",
            "receipt_path",
        }
        if not required_lane_fields <= set(lane):
            fail(f"lane registry entry lacks required fields: {lane!r}")
        name = lane["lane"]
        if name in lane_by_name:
            fail(f"duplicate residual lane: {name}")
        lane_by_name[name] = lane
        if not lane["required_evidence"] or not lane["receipt_path"]:
            fail(f"lane registry entry has empty evidence or receipt path: {name}")
    if {name: lane["classification"] for name, lane in lane_by_name.items()} != EXPECTED_LANES:
        fail("residual lane registry does not match the fixed lane/classification map")
    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    if len(names) != len(set(names)):
        fail("registry contains duplicate names")
    expected = ATOMIC_NAMES | COMPOSITE_NAMES
    if set(names) != expected:
        fail(f"registry names differ; missing={sorted(expected - set(names))}, extra={sorted(set(names) - expected)}")
    if sum(entry.get("kind") == "atomic" for entry in entries) != len(ATOMIC_NAMES):
        fail("registry atomic count does not match the fixed v2 count")
    if sum(entry.get("kind") == "composite" for entry in entries) != len(COMPOSITE_NAMES):
        fail("registry composite count does not match the fixed v2 count")
    owners = set()
    entry_docs = {path.stem for path in (root / "entries").glob("*.md")}
    if entry_docs != expected:
        fail(
            "entry documents differ from registry; "
            f"missing={sorted(expected - entry_docs)}, extra={sorted(entry_docs - expected)}"
        )
    stray_skill_files = [
        path for path in root.rglob("SKILL.md") if path.parent != root
    ]
    if stray_skill_files:
        fail(
            "a nested SKILL.md never loads in Claude Code; move it under "
            f"entries/: {stray_skill_files[0]}"
        )
    for entry in entries:
        if not isinstance(entry, dict) or not REQUIRED_REGISTRY_FIELDS <= set(entry):
            fail(f"registry entry lacks required fields: {entry!r}")
        owner = entry["owner"]
        if not isinstance(owner, str) or not owner.strip():
            fail(f"owner must be a non-empty string: {entry!r}")
        if owner in owners:
            fail(f"owner collision: {owner}")
        owners.add(owner)
        if owner != entry["name"]:
            fail(f"entry owner must be the unique entry owner: {entry['name']}")
        if entry["kind"] not in {"atomic", "composite"}:
            fail(f"invalid kind for {entry['name']}")
        if not isinstance(entry["allows_code_change"], bool):
            fail(f"allows_code_change must be boolean for {entry['name']}")
        for field in ("inputs", "outputs", "stop_status"):
            if not entry[field]:
                fail(f"empty {field} for {entry['name']}")
        if entry["kind"] == "composite":
            calls = entry.get("calls")
            if not isinstance(calls, list) or not calls:
                fail(f"composite must declare calls: {entry['name']}")
            unknown_calls = set(calls) - set(names)
            if unknown_calls:
                fail(f"composite calls unknown entries: {entry['name']}: {sorted(unknown_calls)}")
            if len(calls) != len(set(calls)):
                fail(f"composite has duplicate calls: {entry['name']}")
        elif "calls" in entry:
            fail(f"atomic entry cannot declare calls: {entry['name']}")

    if len(owners) != len(entries):
        fail("registry owners are not unique")
    if any(name in names for name in {
        "frontier-run",
        "vllm-run",
        "semantic-config-audit",
        "e2e-metrics-analysis",
        "residual-gap-analysis",
        "pp-overhead-repair",
        "collective-overhead-repair",
        "transfer-overhead-repair",
        "op-trace-normalize",
    }):
        fail("compatibility aliases or folded op normalizer leaked into the v2 registry")
    residual = next(entry for entry in entries if entry["name"] == "residual-repair")
    if "lane registry" not in flat(read_text(entry_doc(root, "residual-repair"))).lower():
        fail("residual-repair does not reference the lane registry")
    if residual["allows_code_change"] is not True:
        fail("residual-repair must retain direct code-change permission")

    reference_dir = root / "references"
    for name in REQUIRED_REFERENCES:
        read_text(reference_dir / name)

    for name in expected:
        validate_entry_doc(entry_doc(root, name), name)

    all_docs = flat("\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.md")
        if "tests" not in path.parts
    ))
    for phrase in ("case_id", "run_generation", "formal_request_ids", "selected_checkout", "status"):
        if phrase not in all_docs:
            fail(f"missing normative provenance field: {phrase}")
    if "exec capture: UNKNOWN" not in all_docs:
        fail("exec capture capability must be explicitly UNKNOWN")
    if "code change" not in all_docs.lower() or "human review" not in all_docs.lower():
        fail("plain code-change and human-review language is incomplete")
    for path in (entry_doc(root, name) for name in expected):
        if re.search(r"\bmutation\b", path.read_text(encoding="utf-8"), re.IGNORECASE):
            fail(f"non-plain mutation terminology in {path}")

    repair_names = {
        "semantic-alignment-repair",
        "workflow-repair",
        "op-repair",
        "residual-repair",
        "cpu-overhead-repair",
    }
    for name in repair_names:
        text = flat(read_text(entry_doc(root, name))).lower()
        for phrase in ("human review", "code-change marker", "regression set", "decision id", "repair-approval.md"):
            if phrase not in text:
                fail(f"repair entry {name} lacks explicit approval phrase: {phrase}")
        if "fresh" not in text:
            fail(f"repair entry {name} lacks fresh post-rerun evidence wording")

    migration = flat((reference_dir / "migration.md").read_text(encoding="utf-8"))
    for name in expected:
        if name not in migration:
            fail(f"migration map lacks {name}")
    if "default path switch" not in migration.lower() or "rollback" not in migration.lower():
        fail("migration map lacks default-path-switch/rollback operations")

    call_graph = flat(read_text(reference_dir / "call-graph.md"))
    for composite in COMPOSITE_NAMES:
        if composite not in call_graph:
            fail(f"call graph lacks composite: {composite}")
        entry = next(item for item in entries if item["name"] == composite)
        for callee in entry["calls"]:
            if callee not in call_graph:
                fail(f"call graph lacks declared edge: {composite} -> {callee}")
    if call_graph.index("groundtruth-run-check") > call_graph.index("groundtruth-run"):
        fail("call graph places groundtruth-run before groundtruth-run-check")
    if "op-trace-normalize" not in migration or "internally" not in call_graph:
        fail("folded operator normalization is not documented as an internal phase")

    # These checks protect the executable contract details that a flat route
    # table cannot express on its own.
    op_gap = call_graph.index("op-gap-analysis")
    op_supplement = call_graph.index("op-supplement")
    if op_gap > op_supplement:
        fail("call graph must produce op evidence before requesting supplementation")
    if "fresh op-gap-analysis" not in call_graph:
        fail("call graph must return to a fresh operator analysis after supplementation")

    contract = flat(read_text(reference_dir / "contract.md"))
    if "correction_state=pending|complete|not_applicable" not in contract:
        fail("contract must define the reviewed no-change correction state")
    if "CPU lane" not in contract or "not_applicable" not in contract:
        fail("contract must explain how CPU analysis consumes a no-change state")
    if EXPECTED_GROUNDTRUTH_BRANCH not in contract:
        fail("contract must pin the requested ground-truth branch")
    for field in (
        "groundtruth_checkout_path",
        "groundtruth_remote_url",
        "groundtruth_remote_tip",
        "groundtruth_commit",
        "groundtruth_ref",
        "groundtruth_tree_dirty",
        "groundtruth_diff_artifact",
        "groundtruth_dirty_patch_sha256",
        "groundtruth_overlay_patch_applied",
    ):
        if field not in contract:
            fail(f"contract lacks ground-truth provenance field: {field}")
    if "decision ID" not in contract or "verbatim" not in contract:
        fail("contract must record each decision with an ID and the user's verbatim answer")
    if "requesting_user" not in contract or "reviewer_identity" not in contract:
        fail("contract must bind human review to the requesting user or named delegate")

    e2e = flat(read_text(entry_doc(root, "e2e-metrics-gap")))
    for phrase in (
        "arithmetic mean",
        "metric_statistic=mean",
        "request_num_decode_tokens > 1",
        "excluded_count",
        "last_formal_completion_s - first_formal_arrival_s",
        "positive finite",
        "formal request count",
        "sum(prefill_tokens + decode_tokens)",
        "abs(frontier - groundtruth) / abs(groundtruth)",
    ):
        if phrase not in e2e:
            fail(f"E2E contract lacks normative definition: {phrase}")

    registry_text = flat(read_text(reference_dir / "registry.md"))
    if "candidate dependency inventory" not in registry_text:
        fail("registry must describe composite calls as a candidate inventory")
    if "call-graph.md" not in registry_text or "human review gate" not in registry_text:
        fail("registry must defer conditional routing and approval to the call graph")
    residual_entry = next(entry for entry in entries if entry["name"] == "residual-repair")
    if "case manifest" not in residual_entry["inputs"]:
        fail("residual-repair registry input must include the case manifest")

    tool_boundaries = flat(read_text(reference_dir / "tool-boundaries.md"))
    if "hash mismatch" not in tool_boundaries.lower():
        fail("tool boundary contract must fail on helper hash mismatch")

    runtime_files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"} and "tests" not in path.parts
    ]
    if runtime_files:
        fail(f"v2 contains an unapproved runtime helper: {runtime_files[0]}")

    skill_doc = read_text(root / "SKILL.md")
    description = re.search(r"^description:\s*(.+)$", skill_doc, re.MULTILINE)
    if not description or not description.group(1).startswith("Use when"):
        fail("SKILL.md description must start with 'Use when'")
    for path in sorted(reference_dir.iterdir()):
        if f"references/{path.name}" not in skill_doc:
            fail(f"SKILL.md does not point to references/{path.name}")
    if "site.local.md" not in skill_doc:
        fail("SKILL.md must point to the host bindings in site.local.md")

    tracked = [
        path for path in root.rglob("*")
        if path.is_file()
        and "tests" not in path.relative_to(root).parts
        and "__pycache__" not in path.parts
        and path.name != "site.local.md"
    ]
    readme = root.parent / "README.md"
    if readme.is_file():
        tracked.append(readme)
    for path in tracked:
        text = path.read_text(encoding="utf-8")
        for pattern in HOST_SPECIFIC_PATTERNS:
            match = re.search(pattern, text)
            if match:
                fail(f"host-specific text {match.group(0)!r} in {path}; move it to site.local.md")
        if path.name != "migration.md" and re.search(r"auto_recycle|auto receipt", text):
            fail(f"retired automatic approval wording in {path}")

    repo = root.parent.parent
    if not (repo / "frontier").is_dir():
        fail(f"skill must live at skills/<name>/ in a Frontier checkout: {root}")
    for path in tracked:
        for match in REPOSITORY_PATH.finditer(path.read_text(encoding="utf-8")):
            cited = match.group(1).split("::")[0]
            cited = re.sub(r":\d+$", "", cited)
            if any(mark in cited for mark in "<>*{}$"):
                continue
            if not (repo / cited).exists():
                fail(f"{path.relative_to(repo)} cites a repository path that does not exist: {cited}")

    print("v2_pressure=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
