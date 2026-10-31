"""Record syntax, module sizes, diff whitespace and B29 write-scope checks."""
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path("/data/ycfeng/Frontier/.worktrees/b29-ingress-delays")
B29 = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b29"
SOURCES = ["frontier/config/config.py", "frontier/config/kv_cache_transfer_config.py",
           "frontier/simulator.py", "frontier/events/cluster_batch_end_event.py",
           "docs/cli/README.md", "tests/unit/test_serving_path_delays.py",
           "tests/unit/test_dsv4_a10_tools.py", "tests/unit/test_pdd_parallel_initializer.py",
           "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py"]

def git(*args):
    result = subprocess.run(["timeout", "15", "git", *args], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return result.stdout

def main():
    lines = {}
    for path in [ROOT / item for item in SOURCES if item.endswith(".py")] + list(B29.glob("*.py")):
        source = path.read_text()
        ast.parse(source, filename=str(path), feature_version=(3, 10))
        count = len(source.splitlines())
        assert count <= 2000, (path, count)
        lines[str(path.relative_to(ROOT))] = count
    tracked = git("diff", "--name-only").splitlines()
    assert set(tracked) <= set(SOURCES), tracked
    assert git("diff", "--check") == ""
    status = git("status", "--short", "--untracked-files=normal")
    assert "?? tests/unit/test_serving_path_delays.py" in status
    result = dict(status="PASS", python_line_counts=lines, tracked_changed_paths=tracked,
                  new_source_file="tests/unit/test_serving_path_delays.py",
                  diff_check="PASS", diff_stat=git("diff", "--stat"), git_status=status,
                  base_commit=git("rev-parse", "HEAD").strip())
    (B29 / "static_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
