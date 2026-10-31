"""Run the B28 CPU regression selection from isolated scratch storage."""

import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess

PYTHON = "/data/ycfeng/envs/frontier-py310/bin/python"
SCRATCH = Path("/data/ycfeng/tmp/codex-implement/d684ea95-b7ff-4a6d-8fec-63b038641a41/b28")
RECORDS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("label")
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--continue-collection", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    targets = args.targets or json.loads((RECORDS / "regression_targets.json").read_text())
    cwd = SCRATCH / args.label
    cwd.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "tmp").mkdir(exist_ok=True)
    data = cwd / "data"
    if not data.exists():
        data.symlink_to(root / "data", target_is_directory=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", WANDB_DISABLED="true",
               VIDUR_DISABLE_WANDB="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VLLM_BS_SOURCE_ROOT="/data/ycfeng/tmp/opencode/vllm-v0.28.0",
               TMPDIR=str(SCRATCH / "tmp"), FRONTIER_TMP_ROOT=str(SCRATCH / "tmp"),
               PYTHONPATH=":".join([str(root), str(root / "tests/unit"),
                                    str(root / "tests/comparison/calibration"), str(root / "tests/analysis"),
                                    "/usr/local/lib/python3.10/dist-packages", "/usr/lib/python3/dist-packages"]))
    assert "GIT_DIR" not in env and "GIT_WORK_TREE" not in env
    cmd = ["timeout", "3600", PYTHON, "-m", "pytest", *[str(root / p) for p in targets],
           "-q", "-p", "no:cacheprovider", f"--basetemp={SCRATCH}/tmp/{args.label}",
           f"--junitxml={SCRATCH}/{args.label}.xml"]
    if args.continue_collection:
        cmd.append("--continue-on-collection-errors")
    recipe = dict(cwd=str(cwd), environment={k: env[k] for k in (
        "PYTHONDONTWRITEBYTECODE", "WANDB_DISABLED", "VIDUR_DISABLE_WANDB", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VLLM_BS_SOURCE_ROOT", "TMPDIR", "FRONTIER_TMP_ROOT",
        "PYTHONPATH")}, command=shlex.join(cmd), memory_kib=31457280)
    recipe_path = RECORDS / f"{args.label}_command.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    resource.setrlimit(resource.RLIMIT_AS, (31457280 * 1024, 31457280 * 1024))
    with (SCRATCH / f"{args.label}.log").open("w") as log:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    recipe["returncode"] = result.returncode
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps(dict(label=args.label, returncode=result.returncode,
                          log=str(SCRATCH / f"{args.label}.log"))))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
