from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / ".github" / "scripts" / "gemini_review_helper.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "gemini-review.yml"
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
    assert review_job["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
        "issues": "write",
    }

    gemini_steps = [
        step
        for step in review_job["steps"]
        if str(step.get("uses", "")).startswith("google-github-actions/run-gemini-cli@")
    ]
    assert len(gemini_steps) == 1
    settings = json.loads(gemini_steps[0]["with"]["settings"])
    assert settings["telemetry"]["enabled"] is False
    assert settings["tools"]["autoAccept"] is True
    assert "read_file" in settings["tools"]["core"]
    assert "write_file" not in settings["tools"]["core"]
    assert "run_shell_command" not in settings["tools"]["core"]


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
