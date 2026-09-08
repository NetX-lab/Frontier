"""Run the frozen H200 simulation on a CPU host with fresh caches."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    settings = json.loads(args.config.read_text())
    for key, value in settings.items():
        if key.endswith(("input_file", "trace_file")):
            if not Path(value).is_file():
                raise FileNotFoundError(f"Input unavailable: {key}={value}")
    if settings.get("cc_backend_config_type") != "collective_sim":
        raise ValueError("This case requires collective_sim.")
    if settings.get("collective_sim_cc_backend_config_intra_server_model") != "nvlink_analytic":
        raise ValueError("This case requires the existing nvlink_analytic path.")
    backend = root / "frontier/cc_backend/backends/collective-sim"
    if not os.access(backend / "sim/datacenter/htsim_ndp", os.X_OK):
        raise FileNotFoundError("Build collective_sim before running this worker.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="issue26-cpu-frontier-", dir="/data/ycfeng/tmp"))
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1",
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
               WANDB_DISABLED="true", VIDUR_DISABLE_WANDB="1")
    settings.update({
        "metrics_config_run_id": out.name,
        "metrics_config_output_dir": str(out / "metrics"),
        "metrics_config_cache_dir": str(scratch / "predictor-cache"),
        "collective_sim_cc_backend_config_cache_dir": str(scratch / "collective-cache"),
        "collective_sim_cc_backend_config_runner_out_dir": str(scratch / "htsim"),
    })
    command = [sys.executable, "-m", "frontier.main"]
    for key, value in settings.items():
        if isinstance(value, bool):
            command.append(("--" if value else "--no-") + key)
        else:
            command.extend(["--" + key, str(value)])
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (out / "scratch.txt").write_text(str(scratch) + "\n")
    audit = [sys.executable, str(root / "tests/e2e/issue26_simulator_runtime_check.py"),
             str(out / "runtime.json")]
    with (out / "runtime.log").open("w") as stream:
        subprocess.run(audit, cwd=root, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    if args.prepare_only:
        print(f"CPU_FRONTIER_PREPARED: {out}")
        return
    with (out / "frontier.log").open("w") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    (out / "exit_code.txt").write_text(str(result.returncode) + "\n")
    result.check_returncode()
    print(f"FRESH_FRONTIER_EXECUTION_COMPLETE: {out}")


if __name__ == "__main__":
    main()
