from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_dense_thinking_mode_example_script_exists_and_is_shell_valid() -> None:
    relative_path = "examples/architecture/co-location/thinking_mode_basic.sh"
    script_path = REPO_ROOT / relative_path
    assert script_path.exists(), f"Missing example script: {relative_path}"

    script_text = script_path.read_text(encoding="utf-8")
    assert "--enable_thinking_mode" in script_text
    assert "--thinking_depth" in script_text
    assert "--thinking_round_prefill_tokens" in script_text
    assert "--thinking_round_decode_tokens" in script_text
    assert 'CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"' in script_text
    assert '--cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"' in script_text

    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

def test_examples_readme_documents_current_release_thinking_mode_entrypoint() -> None:
    readme_text = _read_text("examples/README.md")

    assert "examples/architecture/co-location/thinking_mode_basic.sh" in readme_text
    assert "examples/architecture/pd-disaggregation/thinking_mode_basic.sh" not in readme_text
    assert "pre-release-v0.1` supports only the `co-location` architecture" in readme_text
