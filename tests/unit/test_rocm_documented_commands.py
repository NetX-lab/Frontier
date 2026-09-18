"""Check executable arguments in the published ROCm recipes without hardware."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys



ROOT = Path(__file__).resolve().parents[2]


def commands(path):
    for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.S):
        for line in block.replace("\\\n", " ").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                yield shlex.split(line)


def test_mi355x_attention_recipe_selects_native_backend():
    selected = [cmd for cmd in commands(ROOT / "docs/profiling/ROCM_MI355X.md")
                if "examples/profiling/profile_attention_chunked_prefill.sh" in cmd]
    assert selected
    for command in selected:
        result = subprocess.run(
            [*command, "--dry-run"], cwd=ROOT,
            env={**os.environ, "PYTHON_BIN": sys.executable, "ATTENTION_BACKEND": "NO_OP"},
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        invocation = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
        arguments = shlex.split(invocation.removeprefix("Command:"))
        assert arguments[arguments.index("--attention_backend") + 1] == "VLLM_ROCM"
