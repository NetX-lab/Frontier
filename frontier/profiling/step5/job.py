"""Stage an immutable smoke package; submission remains a separate parent action."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from frontier.profiling.common.step5_job import (
    CLOUD_MOUNT, CLOUD_ROOT, IMAGE, STEPMIND_ENV, describe_job, stage_package,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def describe_smoke(stage, run_id, *, gpus=1):
    stage = Path(stage).resolve()
    if gpus != 1:
        raise ValueError("The smoke package profiles one isolated TP8 rank on one GPU")
    output = CLOUD_ROOT / run_id
    q = shlex.quote
    worker = "\n".join([
        "set -euo pipefail", "export TMPDIR=/data/ycfeng/tmp/step5_attention_worker",
        "export VLLM_USE_BREAKABLE_CUDAGRAPH=1",
        'mkdir -p "$TMPDIR/model"', f"test ! -e {q(str(output))}",
        'cp data/config/models/stepfun-ai__Step-5-Preview.json "$TMPDIR/model/config.json"',
        'export PYTHONPATH="$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"',
        "python3 -c 'import pandas, dateutil, pytz, tzdata, yaml, tqdm, six; "
        "print(\"OFFLINE_DEPENDENCIES_READY\", pandas.__version__, pandas.__file__)'",
        "nvidia-smi -L",
        "bash tests/integration/step5_profiling/run_on_worker.sh "
        "-m frontier.profiling.step5.main --model-path \"$TMPDIR/model\" "
        "--tensor-parallel-size 8 --max-model-len 8192 "
        "--prefill-tokens 16 64 --prefixes 0 4104 --prefill-batches 1 --decode-batches 1 "
        "--decode-contexts 512 4104 --cudagraph-capture-sizes 1 2 "
        "--min-timing-window-ms 0.5 "
        f"--output-dir {q(str(output))}",
    ])
    return {**describe_job(stage, run_id, worker_command=worker, gpus=gpus,
                          cloud_root=CLOUD_ROOT, credential_env=STEPMIND_ENV),
            "expected_rows": 12,
            "expected_gpu_active_rows": 8, "expected_csv_rows": 20,
            "native_max_num_seqs": 1, "native_cudagraph_capture_sizes": [1, 2],
            "native_phase_graph_modes": {"prefill_16_64": "none", "decode_1": "full"}}


def stage_smoke(stage, run_id, *, dependency_site, gpus=1):
    stage = Path(stage).resolve()
    manifest = describe_smoke(stage, run_id, gpus=gpus)
    return stage_package(stage, manifest, repo_root=REPO_ROOT, dependency_site=dependency_site)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dependency-site", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    manifest = stage_smoke(args.stage, args.run_id, dependency_site=args.dependency_site, gpus=args.gpus)
    print(json.dumps({**{k: v for k, v in manifest.items() if k != "sha256"},
                      "staged_files": len(manifest["sha256"])}, indent=2))


if __name__ == "__main__":
    main()
