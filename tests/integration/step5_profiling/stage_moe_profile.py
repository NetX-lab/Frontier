"""Stage a MoE-only H200 profile package that measures a chosen EP rank.

The role packages of h200_profile_plan.json measure EP rank 0, which balanced routing always
fills. At a small token count the highest routed expert sits on a partly filled rank, and
Frontier prices that rank's lane too. ``--profiled-ep-rank last_routed`` measures that rank.
The rows extend the role tables through merge_role_profiles.py.

Usage: stage_moe_profile.py --stage <dir> --run-id <id> --dependency-site <offline_site>
           --expert-parallel-sizes 16 32 --num-tokens 2 4 8 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from frontier.profiling.common.step5_job import CLOUD_ROOT, describe_job, stage_package

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN = REPO_ROOT / "frontier/profiling/step5/h200_profile_plan.json"


def describe(stage, run_id, *, expert_parallel_sizes, num_tokens, profiled_ep_rank):
    plan = json.loads(PLAN.read_text())
    output = CLOUD_ROOT / run_id
    q = shlex.quote
    worker = [
        "set -euo pipefail", f"export TMPDIR={q(f'/data/ycfeng/tmp/step5_p0_profiles/{run_id}')}",
        'mkdir -p "$TMPDIR"', f"test ! -e {q(str(output))}",
        f"mkdir -p {q(str(output.parent))}", f"mkdir {q(str(output))}",
        'export PYTHONPATH="$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"', "nvidia-smi -L",
    ]
    for method in ("cuda_event", "record_function"):
        worker.append(shlex.join([
            "bash", "tests/integration/step5_profiling/run_on_worker.sh",
            "-m", "frontier.profiling.moe.main", "--disable_ray", "--num_gpus", "1",
            "--models", plan["model"], "--num_tensor_parallel_workers", "1",
            "--expert_parallel_sizes", *map(str, expert_parallel_sizes),
            "--num_tokens_list", *map(str, num_tokens), "--max_tokens", str(max(num_tokens)),
            "--precision", "BF16", "--profile_method", method, "--device", plan["device"],
            "--load_distributions", "balanced", "--num_samples_per_distribution", "1",
            "--routing_runtime_path", "standard_fused_topk", "--profiled_ep_rank", profiled_ep_rank,
            "--output_dir", str(output / "moe"), "--yes",
        ]) + f" 2>&1 | tee {q(str(output / f'moe_{method}_worker.log'))}")
    manifest = describe_job(stage, run_id, worker_command="\n".join(worker), gpus=1,
                            gpu_tag=plan["gpu_tag"], charged_group=plan["charged_group"],
                            preemptible=plan["preemptible"], timeout_seconds=plan["timeout_seconds"])
    return {**manifest, "purpose": "MOE_EP_RANK_PROFILE_EXTENDS_ROLE_TABLES",
            "expert_parallel_sizes": list(expert_parallel_sizes), "num_tokens": list(num_tokens),
            "profiled_ep_rank": profiled_ep_rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dependency-site", type=Path)
    parser.add_argument("--expert-parallel-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--num-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--profiled-ep-rank", choices=("first", "last_routed"), default="last_routed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = describe(args.stage, args.run_id, expert_parallel_sizes=args.expert_parallel_sizes,
                        num_tokens=args.num_tokens, profiled_ep_rank=args.profiled_ep_rank)
    if not args.dry_run:
        if args.dependency_site is None:
            parser.error("staging requires --dependency-site")
        manifest = stage_package(args.stage, manifest, repo_root=REPO_ROOT,
                                 dependency_site=args.dependency_site)
    print(json.dumps({**{k: v for k, v in manifest.items() if k != "sha256"},
                      "staged_files": len(manifest.get("sha256", {}))}, indent=2))


if __name__ == "__main__":
    main()
