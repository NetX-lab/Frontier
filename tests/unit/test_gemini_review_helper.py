from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / ".github" / "scripts" / "gemini_review_helper.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "gemini-review.yml"
CODEX_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "codex-review.yml"
STYLEGUIDE_PATH = REPO_ROOT / ".gemini" / "styleguide.md"


def _load_helper():
    spec = importlib.util.spec_from_file_location("gemini_review_helper", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_yaml_and_embedded_settings_are_valid() -> None:
    assert WORKFLOW_PATH.is_file()
    assert STYLEGUIDE_PATH.is_file()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))

    # PyYAML parses unquoted `on:` as boolean True under YAML 1.1 rules
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    pr_trigger = triggers["pull_request"]
    assert "qwen3-4b-pd-af-simulator" in pr_trigger["branches"]
    assert set(pr_trigger["types"]) >= {"opened", "synchronize"}

    review_job = workflow["jobs"]["review"]
    publish_job = workflow["jobs"]["publish"]
    assert review_job["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert publish_job["permissions"] == {
        "pull-requests": "write",
    }
    assert review_job["timeout-minutes"] <= 20
    assert publish_job["timeout-minutes"] <= 10
    assert "artifact_name" in review_job["outputs"]
    assert "pr_number" in review_job["outputs"]
    upload_steps = [
        step
        for step in review_job["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(upload_steps) == 1
    upload_with = upload_steps[0]["with"]
    upload_path_parts = [p for p in str(upload_with["path"]).split("/") if p]
    if not upload_with.get("include-hidden-files", False):
        assert all(not part.startswith(".") for part in upload_path_parts)

    publish_downloads = [
        step
        for step in publish_job["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert len(publish_downloads) == 1
    assert publish_downloads[0]["with"]["name"] == "${{ needs.review.outputs.artifact_name }}"

    gemini_steps = [
        step
        for step in review_job["steps"]
        if str(step.get("uses", "")).startswith("google-github-actions/run-gemini-cli@")
    ]
    assert len(gemini_steps) == 1
    settings = json.loads(gemini_steps[0]["with"]["settings"])
    assert settings["model"]["maxSessionTurns"] == -1
    assert 0 < settings["model"]["compressionThreshold"] <= 0.3
    assert settings["model"]["summarizeToolOutput"]["run_shell_command"]["tokenBudget"] <= 4000
    overrides = settings["modelConfigs"]["customOverrides"]
    assert len(overrides) >= 1
    assert all(
        "MEDIUM" in o["modelConfig"]["generateContentConfig"]["thinkingConfig"]["thinkingLevel"]
        for o in overrides
    )
    assert settings["telemetry"]["enabled"] is False
    assert settings["tools"]["useRipgrep"] is True
    assert 8000 <= settings["tools"]["truncateToolOutputThreshold"] <= 30000
    core_tools = settings["tools"]["core"]
    assert "read_file" in core_tools
    assert "grep_search" in core_tools
    assert "search_file_content" not in core_tools
    assert "write_file" not in core_tools
    assert "replace" not in core_tools
    # Wildcard shell must never be enabled; only explicit read-only command prefixes are allowed.
    assert "run_shell_command" not in core_tools
    shell_rules = [t for t in core_tools if t.startswith("run_shell_command(")]
    allowed_readonly_shell = {
        "run_shell_command(rg)",
        "run_shell_command(git grep)",
        "run_shell_command(git show)",
        "run_shell_command(sed -n)",
        "run_shell_command(nl)",
        "run_shell_command(head)",
        "run_shell_command(tail)",
        "run_shell_command(wc)",
    }
    assert set(shell_rules) == allowed_readonly_shell


def test_codex_workflow_yaml_and_guardrails_are_valid() -> None:
    assert CODEX_WORKFLOW_PATH.is_file()
    raw_text = CODEX_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(raw_text)

    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    pr_trigger = triggers["pull_request"]
    assert "qwen3-4b-pd-af-simulator" in pr_trigger["branches"]
    assert set(pr_trigger["types"]) >= {
        "opened",
        "reopened",
        "synchronize",
        "ready_for_review",
    }
    assert "edited" not in pr_trigger["types"]

    review_job = workflow["jobs"]["review"]
    publish_job = workflow["jobs"]["publish"]
    assert review_job["permissions"] == {"contents": "read"}
    assert publish_job["permissions"] == {"pull-requests": "write"}
    assert review_job["timeout-minutes"] <= 20
    assert publish_job["timeout-minutes"] <= 10
    assert "artifact_name" in review_job["outputs"]
    download_steps = [
        step
        for step in publish_job["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert len(download_steps) == 1
    assert download_steps[0]["with"]["name"] == "${{ needs.review.outputs.artifact_name }}"

    codex_steps = [
        step
        for step in review_job["steps"]
        if str(step.get("uses", "")).startswith("openai/codex-action@")
    ]
    assert len(codex_steps) == 1
    codex_with = codex_steps[0]["with"]
    assert codex_with["sandbox"] == "read-only"
    assert codex_with["safety-strategy"] == "drop-sudo"
    assert "project_doc_max_bytes" in codex_with["codex-args"]
    assert codex_with["prompt-file"]
    assert codex_with["output-file"]
    assert ":(exclude)data/profiling" in raw_text
    assert "MAX_LEN = 65000" in raw_text


def test_build_diff_artifacts_filters_noisy_paths_and_annotates_lines() -> None:
    helper = _load_helper()
    sample_diff = "\n".join(
        [
            "diff --git a/data/profiling/compute/h200/Qwen3-4B/linear_op.csv b/data/profiling/compute/h200/Qwen3-4B/linear_op.csv",
            "--- a/data/profiling/compute/h200/Qwen3-4B/linear_op.csv",
            "+++ b/data/profiling/compute/h200/Qwen3-4B/linear_op.csv",
            "@@ -1,1 +1,2 @@",
            " header",
            "+1,2,3",
            "diff --git a/frontier/simulator.py b/frontier/simulator.py",
            "--- a/frontier/simulator.py",
            "+++ b/frontier/simulator.py",
            "@@ -10,3 +10,4 @@ def run():",
            " context_before()",
            "-old_call()",
            "+new_call()",
            "+extra_call()",
            " context_after()",
        ]
    )

    artifacts = helper.build_diff_artifacts(sample_diff)
    assert artifacts["reviewable_files"] == ["frontier/simulator.py"]
    assert artifacts["skipped_files"] == [
        "data/profiling/compute/h200/Qwen3-4B/linear_op.csv"
    ]
    assert artifacts["valid_diff_lines"] == {
        "frontier/simulator.py": [10, 11, 12, 13]
    }
    assert "L11 +new_call()" in artifacts["annotated_diff"]
    assert "L12 +extra_call()" in artifacts["annotated_diff"]
    assert "     -old_call()" in artifacts["annotated_diff"]


def test_build_review_payload_enforces_severity_and_line_validation() -> None:
    helper = _load_helper()
    raw_output = """Here is the review:
```json
{
  "verdict": "REQUEST_CHANGES",
  "summary": "发现 1 个高优先级问题和 1 个中优先级问题。",
  "comments": [
    {
      "path": "frontier/simulator.py",
      "line": 11,
      "severity": "LOW",
      "category": "风格",
      "title": "变量命名偏好",
      "body": "可忽略的细节建议"
    },
    {
      "path": "frontier/simulator.py",
      "line": 12,
      "severity": "HIGH",
      "category": "仿真保真度",
      "title": "时间单位误用为毫秒",
      "body": "此处应使用秒而非毫秒，会导致 TTFT 放大 1000 倍。",
      "suggestion": "extra_call(duration_s=dur_ms / 1000.0)"
    },
    {
      "path": "frontier/simulator.py",
      "line": 999,
      "severity": "MEDIUM",
      "category": "测试覆盖",
      "title": "缺少边界单测",
      "body": "建议补充空 batch 单测。"
    }
  ]
}
```"""

    payload = helper.build_review_payload(
        raw_response=raw_output,
        pr_meta={"head_sha": "abcdef1234567890", "skipped_files": ["a.csv"]},
        valid_diff_lines={"frontier/simulator.py": [10, 11, 12, 13]},
        max_comments=15,
    )

    assert payload["event"] == "COMMENT"
    assert payload["commit_id"] == "abcdef1234567890"
    # LOW filtered out; line 12 -> inline; line 999 -> body-only fallback
    assert len(payload["comments"]) == 1
    inline = payload["comments"][0]
    assert inline["path"] == "frontier/simulator.py"
    assert inline["line"] == 12
    assert inline["side"] == "RIGHT"
    assert "🟠 [高]" in inline["body"]
    assert "```suggestion" in inline["body"]
    assert "变量命名偏好" not in payload["body"]
    assert "变量命名偏好" not in inline["body"]
    assert "缺少边界单测" in payload["body"]
    assert "高 `1` | 🟡 中 `1`" in payload["body"]


def test_build_review_payload_handles_non_json_fallback() -> None:
    helper = _load_helper()
    payload = helper.build_review_payload(
        raw_response="## 纯文本审查结论\n\n整体逻辑清晰，未发现明显缺陷。",
        pr_meta={"head_sha": "1234567890abcdef"},
        valid_diff_lines={},
    )
    assert payload["comments"] == []
    assert "纯文本审查结论" in payload["body"]
    assert payload["commit_id"] == "1234567890abcdef"


def test_build_review_payload_preserves_dotfile_paths_for_inline_comments() -> None:
    helper = _load_helper()
    raw_output = json.dumps(
        {
            "verdict": "COMMENT",
            "summary": "发现 1 个工作流超时风险。",
            "comments": [
                {
                    "path": ".github/workflows/codex-review.yml",
                    "line": 20,
                    "severity": "MEDIUM",
                    "category": "可维护性",
                    "title": "缺少超时配置",
                    "body": "建议配置 timeout-minutes。",
                },
                {
                    "path": "./.github/workflows/codex-review.yml",
                    "line": 44,
                    "severity": "MEDIUM",
                    "category": "性能",
                    "title": "未排除大文件",
                    "body": "建议排除 profiling CSV。",
                },
            ],
        },
        ensure_ascii=False,
    )

    payload = helper.build_review_payload(
        raw_response=raw_output,
        pr_meta={"head_sha": "bae65cd7bdd718fe"},
        valid_diff_lines={".github/workflows/codex-review.yml": [20, 44]},
    )

    assert len(payload["comments"]) == 2
    assert [c["path"] for c in payload["comments"]] == [
        ".github/workflows/codex-review.yml",
        ".github/workflows/codex-review.yml",
    ]
    assert [c["line"] for c in payload["comments"]] == [20, 44]
    assert "补充审查意见" not in payload["body"]


def test_normalize_truncate_and_cli_stats(tmp_path: Path) -> None:
    helper = _load_helper()
    assert (
        helper.normalize_repo_path("./.github/workflows/codex-review.yml")
        == ".github/workflows/codex-review.yml"
    )
    assert helper.normalize_repo_path("/frontier/main.py") == "frontier/main.py"

    oversized = "x" * 70_000
    truncated = helper.truncate_review_body(oversized, max_chars=65_000)
    assert len(truncated) <= 65_000
    assert "已截断" in truncated

    cli_json = tmp_path / "stdout.log"
    cli_json.write_text(
        json.dumps(
            {
                "response": "{}",
                "stats": {
                    "models": {
                        "gemini-4-neon": {
                            "api": {"totalRequests": 3},
                            "tokens": {
                                "prompt": 12000,
                                "cached": 4000,
                                "candidates": 800,
                                "thoughts": 1500,
                                "total": 14300,
                            },
                        }
                    },
                    "tools": {
                        "totalCalls": 2,
                        "byName": {"read_file": {"count": 2}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    summary = helper.summarize_cli_stats(cli_json)
    assert summary is not None
    assert "gemini-4-neon" in summary
    assert "thoughts=1500" in summary
    assert "read_file=2" in summary


def test_build_diff_artifacts_expands_surrounding_context_only_for_source_files(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    src_file = tmp_path / "frontier" / "core.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text("\n".join(f"src_line_{i}" for i in range(1, 41)) + "\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "unit" / "test_core.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("\n".join(f"test_line_{i}" for i in range(1, 41)) + "\n", encoding="utf-8")

    sample_diff = "\n".join(
        [
            "diff --git a/frontier/core.py b/frontier/core.py",
            "--- a/frontier/core.py",
            "+++ b/frontier/core.py",
            "@@ -18,3 +18,4 @@ def f():",
            " src_line_18",
            "-old_19",
            "+src_line_19",
            "+src_line_20",
            " src_line_21",
            "diff --git a/tests/unit/test_core.py b/tests/unit/test_core.py",
            "--- a/tests/unit/test_core.py",
            "+++ b/tests/unit/test_core.py",
            "@@ -18,2 +18,3 @@ def test_f():",
            " test_line_18",
            "+test_line_19",
            " test_line_20",
        ]
    )

    artifacts = helper.build_diff_artifacts(
        sample_diff,
        repo_root=tmp_path,
        extra_context_lines=5,
    )
    # Valid inline comment lines must strictly match original -U3 diff lines (L<num> only).
    assert artifacts["valid_diff_lines"]["frontier/core.py"] == [18, 19, 20, 21]
    assert artifacts["valid_diff_lines"]["tests/unit/test_core.py"] == [18, 19, 20]

    # Non-test source file gets C<num> surrounding context lines; test file does not.
    diff_out = artifacts["annotated_diff"]
    assert "C13   src_line_13" in diff_out
    assert "C17   src_line_17" in diff_out
    assert "C22   src_line_22" in diff_out
    assert "C26   src_line_26" in diff_out
    assert "C13   test_line_13" not in diff_out
    assert "C21   test_line_21" not in diff_out
