"""Run frozen GLM observer cases through the existing native HTTP replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_cases(root, config, replay):
    """Run explicit clean/light cases sequentially in this worker allocation."""
    output = Path(config["output_dir"])
    results = []
    for case in config["replay_cases"]:
        arguments = [
            "--engine-config", str(root / "engine.json"),
            "--trace-dir", str(root / "trace"),
            "--output-dir", str(output / case["name"]),
            "--mode", case["mode"], "--model-dir", str(root / "model"),
            "--startup-timeout-s", str(config["startup_timeout_seconds"]),
            "--drain-s", str(config["drain_seconds"]),
        ]
        if case["mode"] == "light":
            arguments.extend(["--observer-config", str(root / case["observer_config"])])
        result = replay(arguments)
        results.append(dict(case=case["name"], mode=case["mode"], returncode=result))
        (output / "case_results.json").write_text(json.dumps(results, indent=2) + "\n")
        if result:
            return result
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config = json.loads(parser.parse_args().config.read_text())
    root = Path(config["script"]).parent
    output = Path(config["output_dir"])
    assert output.resolve().is_relative_to("/mnt/codesign-exp/ycfeng")
    output.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="glm_queue_"))
    for variable, child in (("HF_HOME", "hf"), ("VLLM_CACHE_ROOT", "vllm"),
                            ("TRITON_CACHE_DIR", "triton"),
                            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                            ("FLASHINFER_WORKSPACE_BASE", "flashinfer")):
        os.environ[variable] = str(scratch / child)
    import torch
    import vllm
    source_root = Path(vllm.__file__).parent.parent
    actual = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
              for name in config["expected_source_hashes"]}
    identity = {"source_hashes": actual, "python": sys.version,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
                "nvidia_smi_L": subprocess.check_output(["nvidia-smi", "-L"], text=True)}
    (output / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    assert actual == config["expected_source_hashes"]
    assert vllm.__version__ == config["expected_vllm_version"]
    sys.path.insert(0, str(root / "tests/comparison/dp_placement_pp"))
    import vllm_replay
    return run_cases(root, config, vllm_replay.main)



if __name__ == "__main__":
    raise SystemExit(main())
