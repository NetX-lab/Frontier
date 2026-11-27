#!/usr/bin/env python3
"""Helper for Gemini CLI GitHub Actions PR code review workflow.

Subcommands:
  prepare  Fetch PR metadata and diff via `gh`, filter noisy paths, annotate
           new-file line numbers, and write context files for Gemini CLI.
  post     Parse Gemini's JSON review output, enforce CRITICAL/HIGH/MEDIUM
           filtering, validate inline comment lines against diff hunks, and
           submit the PR review via `gh api` with automatic 422 fallback.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

IGNORED_GLOBS: tuple[str, ...] = (
    "data/profiling/*",
    "data/profiling/**",
    "figs/*",
    "figs/**",
    "cache/*",
    "cache/**",
    "outputs/*",
    "outputs/**",
    "frontier/cc_backend/backends/collective-sim/*",
    "frontier/cc_backend/backends/collective-sim/**",
    "*.csv",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.pkl",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.lock",
)

SEVERITY_ALIASES: dict[str, str] = {
    "CRITICAL": "CRITICAL",
    "SEVERE": "CRITICAL",
    "BLOCKER": "CRITICAL",
    "P0": "CRITICAL",
    "严重": "CRITICAL",
    "HIGH": "HIGH",
    "ERROR": "HIGH",
    "P1": "HIGH",
    "高": "HIGH",
    "MEDIUM": "MEDIUM",
    "MODERATE": "MEDIUM",
    "WARN": "MEDIUM",
    "WARNING": "MEDIUM",
    "P2": "MEDIUM",
    "中": "MEDIUM",
}

SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
}

SEVERITY_BADGES: dict[str, str] = {
    "CRITICAL": "🔴 [严重]",
    "HIGH": "🟠 [高]",
    "MEDIUM": "🟡 [中]",
}

VERDICT_BADGES: dict[str, str] = {
    "APPROVE": "✅ 未发现中高优先级阻塞问题，建议合并",
    "COMMENT": "⚠️ 发现若干中高优先级关注点，建议确认后合并",
    "REQUEST_CHANGES": "🛑 发现高风险/严重问题，建议修复后再合并",
}

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
MAX_REVIEW_BODY_CHARS = 65_000


def normalize_repo_path(file_path: str) -> str:
    """Normalize a repo-relative path without stripping leading dots on dotfiles."""
    normalized = file_path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def truncate_review_body(body: str, max_chars: int = MAX_REVIEW_BODY_CHARS) -> str:
    """Ensure review body stays under GitHub's 65,536-character API limit."""
    if len(body) <= max_chars:
        return body
    notice = "\n\n...(审查内容超出 GitHub 单条评论长度上限，已截断)\n"
    return body[: max(0, max_chars - len(notice))].rstrip() + notice


def should_ignore_path(file_path: str) -> bool:
    """Return True if the path matches an ignored data/artifact/submodule rule."""
    normalized = normalize_repo_path(file_path)
    for pattern in IGNORED_GLOBS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return True
    return False


def split_unified_diff(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified diff string into `(new_path, raw_file_diff)` tuples."""
    sections: list[tuple[str, list[str]]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    for line in diff_text.splitlines():
        match = DIFF_GIT_RE.match(line)
        if match:
            if current_path is not None:
                sections.append((current_path, current_lines))
            current_path = match.group(2)
            current_lines = [line]
        elif current_path is not None:
            current_lines.append(line)

    if current_path is not None:
        sections.append((current_path, current_lines))

    return [(path, "\n".join(lines)) for path, lines in sections]


def annotate_file_diff(
    raw_diff: str, max_file_bytes: int
) -> tuple[str, list[int], bool]:
    """Annotate diff hunks with right-side line numbers and collect valid lines.

    GitHub's PR Review API allows `side: "RIGHT"` inline comments on added (`+`)
    and context (` `) lines inside diff hunks.
    """
    valid_lines: list[int] = []
    annotated_lines: list[str] = []
    new_line_no: int | None = None
    in_hunk = False

    for line in raw_diff.splitlines():
        hunk_match = HUNK_HEADER_RE.match(line)
        if hunk_match:
            new_line_no = int(hunk_match.group(3))
            in_hunk = True
            annotated_lines.append(line)
            continue

        if not in_hunk or new_line_no is None:
            annotated_lines.append(line)
            continue

        if line.startswith("+") and not line.startswith("+++"):
            valid_lines.append(new_line_no)
            annotated_lines.append(f"L{new_line_no} {line}")
            new_line_no += 1
        elif line.startswith("-") and not line.startswith("---"):
            annotated_lines.append(f"     {line}")
        elif line.startswith("\\"):
            annotated_lines.append(line)
        else:
            # Context line inside hunk (starts with ' ' or empty line)
            valid_lines.append(new_line_no)
            annotated_lines.append(f"L{new_line_no} {line}")
            new_line_no += 1

    annotated_text = "\n".join(annotated_lines)
    truncated = False
    encoded = annotated_text.encode("utf-8")
    if len(encoded) > max_file_bytes:
        truncated = True
        clipped = encoded[:max_file_bytes].decode("utf-8", errors="ignore")
        annotated_text = (
            clipped.rstrip()
            + "\n... [该文件 diff 过长已截断，请按需使用 read_file 查看完整文件]"
        )

    return annotated_text, sorted(set(valid_lines)), truncated


def build_diff_artifacts(
    diff_text: str,
    max_diff_bytes: int = 120_000,
    max_file_bytes: int = 30_000,
) -> dict[str, Any]:
    """Filter, annotate, and budget a unified diff for Gemini review."""
    valid_diff_lines: dict[str, list[int]] = {}
    kept_blocks: list[str] = []
    reviewable_files: list[str] = []
    skipped_files: list[str] = []
    truncated_files: list[str] = []
    omitted_budget_files: list[str] = []

    used_bytes = 0
    for file_path, raw_diff in split_unified_diff(diff_text):
        if should_ignore_path(file_path):
            skipped_files.append(file_path)
            continue

        annotated, valid_lines, was_truncated = annotate_file_diff(
            raw_diff, max_file_bytes=max_file_bytes
        )
        reviewable_files.append(file_path)
        valid_diff_lines[file_path] = valid_lines
        if was_truncated:
            truncated_files.append(file_path)

        block = f"=== FILE: {file_path} ===\n{annotated}\n"
        block_bytes = len(block.encode("utf-8"))
        if used_bytes + block_bytes <= max_diff_bytes or not kept_blocks:
            kept_blocks.append(block)
            used_bytes += block_bytes
        else:
            omitted_budget_files.append(file_path)

    return {
        "annotated_diff": "\n".join(kept_blocks),
        "valid_diff_lines": valid_diff_lines,
        "reviewable_files": reviewable_files,
        "skipped_files": skipped_files,
        "truncated_files": truncated_files,
        "omitted_budget_files": omitted_budget_files,
    }


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=check,
    )


def _fetch_pr_diff(repo: str, pr_number: str) -> str:
    """Fetch PR unified diff using `gh pr diff`, falling back to per-file patches."""
    proc = _run_cmd(
        ["gh", "pr", "diff", str(pr_number), "--repo", repo],
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout

    # Fallback for huge diffs (>300 files or >20k lines where GitHub returns 406)
    files_proc = _run_cmd(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/pulls/{pr_number}/files",
        ],
        check=True,
    )
    files_data = json.loads(files_proc.stdout)
    chunks: list[str] = []
    for item in files_data:
        filename = item.get("filename")
        patch = item.get("patch")
        if filename and patch:
            chunks.append(
                f"diff --git a/{filename} b/{filename}\n"
                f"--- a/{filename}\n"
                f"+++ b/{filename}\n"
                f"{patch}"
            )
    return "\n".join(chunks)


def _set_github_output(values: dict[str, str]) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def cmd_prepare(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_proc = _run_cmd(
        [
            "gh",
            "pr",
            "view",
            str(args.pr_number),
            "--repo",
            args.repo,
            "--json",
            "number,title,body,author,baseRefName,headRefName,headRefOid,"
            "additions,deletions,changedFiles,url,files",
        ],
        check=True,
    )
    pr_meta = json.loads(meta_proc.stdout)
    diff_text = _fetch_pr_diff(args.repo, str(args.pr_number))
    artifacts = build_diff_artifacts(
        diff_text,
        max_diff_bytes=args.max_diff_bytes,
        max_file_bytes=args.max_file_bytes,
    )

    head_sha = str(pr_meta.get("headRefOid", ""))
    summary_meta = {
        "number": pr_meta.get("number", int(args.pr_number)),
        "title": pr_meta.get("title", ""),
        "base_ref": pr_meta.get("baseRefName", ""),
        "head_ref": pr_meta.get("headRefName", ""),
        "head_sha": head_sha,
        "url": pr_meta.get("url", ""),
        "author": (pr_meta.get("author") or {}).get("login", ""),
        "additions": pr_meta.get("additions", 0),
        "deletions": pr_meta.get("deletions", 0),
        "changed_files": pr_meta.get("changedFiles", 0),
        "reviewable_files": artifacts["reviewable_files"],
        "skipped_files": artifacts["skipped_files"],
        "truncated_files": artifacts["truncated_files"],
        "omitted_budget_files": artifacts["omitted_budget_files"],
    }

    (output_dir / "pr_meta.json").write_text(
        json.dumps(summary_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "valid_diff_lines.json").write_text(
        json.dumps(artifacts["valid_diff_lines"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pr_diff.txt").write_text(
        artifacts["annotated_diff"],
        encoding="utf-8",
    )

    pr_body = (pr_meta.get("body") or "").strip() or "（无 PR 描述）"
    context_lines = [
        f"# PR #{summary_meta['number']}: {summary_meta['title']}",
        "",
        f"- **作者**: `{summary_meta['author']}`",
        f"- **合并方向**: `{summary_meta['head_ref']}` -> `{summary_meta['base_ref']}`",
        f"- **Head Commit**: `{head_sha[:12]}`",
        f"- **变更规模**: +{summary_meta['additions']} / -{summary_meta['deletions']}（共 {summary_meta['changed_files']} 个文件）",
        "",
        "## PR 描述",
        "",
        pr_body,
        "",
        f"## 待审查文件（{len(artifacts['reviewable_files'])} 个）",
        "",
    ]
    for path in artifacts["reviewable_files"]:
        tags: list[str] = []
        if path in artifacts["truncated_files"]:
            tags.append("diff 已截断，可用 read_file 查看全文")
        if path in artifacts["omitted_budget_files"]:
            tags.append("超出总 diff 预算已省略，可用 read_file 查看")
        suffix = f"（{'; '.join(tags)}）" if tags else ""
        context_lines.append(f"- `{path}`{suffix}")

    if artifacts["skipped_files"]:
        context_lines.extend(
            [
                "",
                f"## 已自动跳过的文件（{len(artifacts['skipped_files'])} 个，属 profiling/数据/图表/子模块路径）",
                "",
            ]
        )
        for path in artifacts["skipped_files"][:50]:
            context_lines.append(f"- `{path}`")
        if len(artifacts["skipped_files"]) > 50:
            context_lines.append(
                f"- ... 另有 {len(artifacts['skipped_files']) - 50} 个文件已跳过"
            )

    (output_dir / "pr_context.md").write_text(
        "\n".join(context_lines) + "\n",
        encoding="utf-8",
    )

    _set_github_output(
        {
            "reviewable_files": str(len(artifacts["reviewable_files"])),
            "head_sha": head_sha,
        }
    )
    print(
        f"Prepared PR #{summary_meta['number']}: "
        f"{len(artifacts['reviewable_files'])} reviewable files, "
        f"{len(artifacts['skipped_files'])} skipped files."
    )
    return 0


def extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON object from Gemini output, tolerating fences or prose."""
    text = raw_text.strip()
    if not text:
        return None

    # 1. Direct JSON object
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Fenced ```json ... ``` or ``` ... ``` blocks (prefer the last valid dict)
    fence_matches = re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    for block in reversed(fence_matches):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3. Balanced top-level {...} scan from last '{' backwards
    decoder = json.JSONDecoder()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict) and (
                "summary" in parsed or "comments" in parsed or "verdict" in parsed
            ):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def normalize_severity(raw_value: Any) -> str | None:
    """Map severity string to CRITICAL/HIGH/MEDIUM, or None to drop."""
    if not isinstance(raw_value, str):
        return None
    key = raw_value.strip().upper()
    if key in SEVERITY_ALIASES:
        return SEVERITY_ALIASES[key]
    return SEVERITY_ALIASES.get(raw_value.strip())


def _format_comment_body(item: dict[str, Any]) -> str:
    badge = SEVERITY_BADGES[item["severity"]]
    category = item.get("category", "").strip()
    title = item.get("title", "").strip()
    body = item.get("body", "").strip()
    suggestion = item.get("suggestion")

    header_parts = [f"**{badge}**"]
    if category:
        header_parts.append(f"`{category}`")
    if title:
        header_parts.append(f"**{title}**")
    parts = [" ".join(header_parts)]
    if body:
        parts.append(body)
    if isinstance(suggestion, str) and suggestion.strip():
        clean_suggestion = suggestion.rstrip("\n")
        parts.append(f"```suggestion\n{clean_suggestion}\n```")
    return "\n\n".join(parts)


def build_review_payload(
    raw_response: str,
    pr_meta: dict[str, Any],
    valid_diff_lines: dict[str, list[int]],
    max_comments: int = 15,
) -> dict[str, Any]:
    """Build a validated GitHub PR review API payload from Gemini output."""
    parsed = extract_json_payload(raw_response)
    head_sha = str(pr_meta.get("head_sha", ""))
    short_sha = head_sha[:10] if head_sha else "unknown"

    if parsed is None:
        fallback_body = truncate_review_body(
            "<!-- gemini-code-review-bot -->\n"
            f"## 🤖 Gemini Code Review（commit `{short_sha}`）\n\n"
            "> 模型未返回标准 JSON 结构，以下为原始审查结果：\n\n"
            f"{raw_response.strip() or '（Gemini 未返回任何审查文本）'}\n"
        )
        payload: dict[str, Any] = {
            "event": "COMMENT",
            "body": fallback_body,
            "comments": [],
        }
        if head_sha:
            payload["commit_id"] = head_sha
        return payload

    raw_comments = parsed.get("comments")
    if not isinstance(raw_comments, list):
        raw_comments = []

    filtered: list[dict[str, Any]] = []
    for entry in raw_comments:
        if not isinstance(entry, dict):
            continue
        severity = normalize_severity(entry.get("severity"))
        if severity is None:
            continue
        body_text = str(entry.get("body") or "").strip()
        title_text = str(entry.get("title") or "").strip()
        if not body_text and not title_text:
            continue

        raw_path = normalize_repo_path(str(entry.get("path") or ""))
        raw_line = entry.get("line")
        line_no: int | None = None
        if isinstance(raw_line, int) and raw_line > 0:
            line_no = raw_line
        elif isinstance(raw_line, str) and raw_line.strip().isdigit():
            line_no = int(raw_line.strip())

        raw_start = entry.get("start_line")
        start_line: int | None = None
        if isinstance(raw_start, int) and raw_start > 0:
            start_line = raw_start
        elif isinstance(raw_start, str) and raw_start.strip().isdigit():
            start_line = int(raw_start.strip())

        filtered.append(
            {
                "path": raw_path,
                "line": line_no,
                "start_line": start_line,
                "severity": severity,
                "category": str(entry.get("category") or "").strip(),
                "title": title_text,
                "body": body_text,
                "suggestion": entry.get("suggestion"),
            }
        )

    filtered.sort(key=lambda item: SEVERITY_ORDER[item["severity"]])
    omitted_count = max(0, len(filtered) - max_comments)
    kept = filtered[:max_comments]

    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
    for item in kept:
        counts[item["severity"]] += 1

    inline_comments: list[dict[str, Any]] = []
    body_only_comments: list[dict[str, Any]] = []

    valid_sets = {p: set(lines) for p, lines in valid_diff_lines.items()}
    for item in kept:
        path = item["path"]
        line_no = item["line"]
        if path and line_no is not None and line_no in valid_sets.get(path, set()):
            comment_obj: dict[str, Any] = {
                "path": path,
                "line": line_no,
                "side": "RIGHT",
                "body": _format_comment_body(item),
            }
            start_line = item["start_line"]
            if (
                start_line is not None
                and start_line < line_no
                and start_line in valid_sets[path]
            ):
                comment_obj["start_line"] = start_line
                comment_obj["start_side"] = "RIGHT"
            inline_comments.append(comment_obj)
        else:
            body_only_comments.append(item)

    raw_verdict = str(parsed.get("verdict") or "").strip().upper()
    if raw_verdict not in VERDICT_BADGES:
        if counts["CRITICAL"] > 0 or counts["HIGH"] > 0:
            raw_verdict = "REQUEST_CHANGES"
        elif counts["MEDIUM"] > 0:
            raw_verdict = "COMMENT"
        else:
            raw_verdict = "APPROVE"
    verdict_line = VERDICT_BADGES[raw_verdict]

    summary_text = str(parsed.get("summary") or "").strip()
    if not summary_text:
        summary_text = (
            "已完成本次变更审查，未发现中高优先级问题。"
            if not kept
            else "已完成本次变更审查，请查看下方及代码行内审查意见。"
        )

    body_lines = [
        "<!-- gemini-code-review-bot -->",
        f"## 🤖 Gemini Code Review（commit `{short_sha}`）",
        "",
        f"**结论**：{verdict_line}",
        f"**问题统计（仅中高优先级）**：🔴 严重 `{counts['CRITICAL']}` | 🟠 高 `{counts['HIGH']}` | 🟡 中 `{counts['MEDIUM']}`",
        "",
        "### 变更与总体评价",
        "",
        summary_text,
    ]

    if body_only_comments:
        body_lines.extend(
            [
                "",
                "### 补充审查意见（跨文件或非 Diff 行）",
                "",
            ]
        )
        for item in body_only_comments:
            loc = f"`{item['path']}`" if item["path"] else "全局"
            if item["path"] and item["line"] is not None:
                loc = f"`{item['path']}:{item['line']}`"
            body_lines.append(f"- {loc}")
            body_lines.append("")
            indented = "\n".join(
                f"  {line}" if line else ""
                for line in _format_comment_body(item).splitlines()
            )
            body_lines.append(indented)
            body_lines.append("")

    notes: list[str] = []
    if omitted_count > 0:
        notes.append(f"已按严重级别上限截断 {omitted_count} 条额外评论（上限 {max_comments} 条）。")
    skipped_files = pr_meta.get("skipped_files") or []
    if skipped_files:
        notes.append(
            f"已按规范跳过 {len(skipped_files)} 个 profiling/数据/图表/子模块文件。"
        )
    omitted_budget = pr_meta.get("omitted_budget_files") or []
    if omitted_budget:
        notes.append(
            f"另有 {len(omitted_budget)} 个文件因超出单次 Diff 长度预算未展开（详见本次 run 日志）。"
        )
    if notes:
        body_lines.extend(["", "---", "<sub>" + " ".join(notes) + "</sub>"])

    payload = {
        "event": "COMMENT",
        "body": truncate_review_body("\n".join(body_lines) + "\n"),
        "comments": inline_comments,
    }
    if head_sha:
        payload["commit_id"] = head_sha
    return payload


def summarize_cli_stats(cli_json_path: Path) -> str | None:
    """Format token and tool usage from Gemini CLI JSON output if available."""
    if not cli_json_path.is_file():
        return None
    try:
        data = json.loads(cli_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    stats = data.get("stats")
    if not isinstance(stats, dict):
        return None

    lines = ["### Gemini CLI Usage Stats", ""]
    models = stats.get("models")
    if isinstance(models, dict) and models:
        for model_name, model_stats in models.items():
            if not isinstance(model_stats, dict):
                continue
            tokens = model_stats.get("tokens")
            api = model_stats.get("api")
            reqs = api.get("totalRequests") if isinstance(api, dict) else None
            if isinstance(tokens, dict):
                prompt_tok = tokens.get("prompt", tokens.get("input", 0))
                cached_tok = tokens.get("cached", 0)
                cand_tok = tokens.get("candidates", tokens.get("output", 0))
                thought_tok = tokens.get("thoughts", 0)
                total_tok = tokens.get("total", 0)
                req_part = f", requests={reqs}" if reqs is not None else ""
                lines.append(
                    f"- `{model_name}`: input={prompt_tok} (cached={cached_tok}), "
                    f"output={cand_tok}, thoughts={thought_tok}, total={total_tok}{req_part}"
                )

    tools = stats.get("tools")
    if isinstance(tools, dict):
        total_calls = tools.get("totalCalls")
        by_name = tools.get("byName")
        if total_calls is not None:
            detail = ""
            if isinstance(by_name, dict) and by_name:
                counts = [
                    f"{name}={info.get('count', 0)}"
                    for name, info in sorted(by_name.items())
                    if isinstance(info, dict)
                ]
                if counts:
                    detail = f" ({', '.join(counts)})"
            lines.append(f"- Tool calls: {total_calls}{detail}")

    return "\n".join(lines) + "\n" if len(lines) > 2 else None


def _submit_review(repo: str, pr_number: str, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp_path = tmp.name
    try:
        return _run_cmd(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{repo}/pulls/{pr_number}/reviews",
                "--input",
                tmp_path,
            ],
            check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def cmd_post(args: argparse.Namespace) -> int:
    context_dir = Path(args.context_dir)
    pr_meta = json.loads((context_dir / "pr_meta.json").read_text(encoding="utf-8"))
    valid_diff_lines = json.loads(
        (context_dir / "valid_diff_lines.json").read_text(encoding="utf-8")
    )
    raw_response = Path(args.response_file).read_text(encoding="utf-8")

    if args.cli_json_file:
        stats_md = summarize_cli_stats(Path(args.cli_json_file))
        if stats_md:
            print(stats_md.strip())
            step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if step_summary:
                with open(step_summary, "a", encoding="utf-8") as fh:
                    fh.write("\n" + stats_md)

    payload = build_review_payload(
        raw_response=raw_response,
        pr_meta=pr_meta,
        valid_diff_lines=valid_diff_lines,
        max_comments=args.max_comments,
    )

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    proc = _submit_review(args.repo, str(args.pr_number), payload)
    if proc.returncode == 0:
        print(
            f"Submitted Gemini review on {args.repo}#{args.pr_number} "
            f"with {len(payload['comments'])} inline comment(s)."
        )
        return 0

    # Fallback: if GitHub rejected an inline comment line (e.g. HTTP 422), fold
    # all inline comments into the review body and retry once so feedback lands.
    if payload["comments"]:
        print(
            f"::warning::Inline PR review submission failed ({proc.stderr.strip()}); "
            "retrying with findings embedded in review body.",
            file=sys.stderr,
        )
        extra_lines = ["", "### 行内审查意见（回退汇总）", ""]
        for c in payload["comments"]:
            extra_lines.append(f"- `{c['path']}:{c['line']}`")
            extra_lines.append("")
            for line in str(c["body"]).splitlines():
                extra_lines.append(f"  {line}" if line else "")
            extra_lines.append("")
        fallback_payload = dict(payload)
        fallback_payload["body"] = truncate_review_body(
            payload["body"] + "\n".join(extra_lines) + "\n"
        )
        fallback_payload["comments"] = []
        retry = _submit_review(args.repo, str(args.pr_number), fallback_payload)
        if retry.returncode == 0:
            print(
                f"Submitted fallback Gemini review on {args.repo}#{args.pr_number}."
            )
            return 0
        print(retry.stderr, file=sys.stderr)
        return retry.returncode

    print(proc.stderr, file=sys.stderr)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemini Code Review workflow helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Prepare PR context and diff")
    prep.add_argument("--pr-number", required=True)
    prep.add_argument("--repo", required=True)
    prep.add_argument("--output-dir", default=".gemini-review-ctx")
    prep.add_argument("--max-diff-bytes", type=int, default=120_000)
    prep.add_argument("--max-file-bytes", type=int, default=30_000)
    prep.set_defaults(func=cmd_prepare)

    post = subparsers.add_parser("post", help="Validate and post Gemini PR review")
    post.add_argument("--pr-number", required=True)
    post.add_argument("--repo", required=True)
    post.add_argument("--context-dir", default=".gemini-review-ctx")
    post.add_argument("--response-file", required=True)
    post.add_argument("--cli-json-file", default="")
    post.add_argument("--max-comments", type=int, default=15)
    post.add_argument("--dry-run", action="store_true")
    post.set_defaults(func=cmd_post)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
