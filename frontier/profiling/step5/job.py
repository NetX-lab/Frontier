"""Stage an immutable smoke package; submission remains a separate parent action."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

from frontier.profiling.common.step5_job import (
    CLOUD_MOUNT, CLOUD_ROOT, IMAGE, STEPMIND_ENV, describe_job, stage_package,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKLOAD = Path("data/processed_traces/canonical_fixed_6144_1024_req128_qps1")
WORKLOAD_MANIFEST_SHA256 = "5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3"
WORKLOAD_SUMS_SHA256 = "594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9"
PROBE_INPUTS = Path("frontier/profiling/step5/fixed_probe_inputs.json")
PROBE_RUN_ID = "fixed_h200_small_probe_20261021_01"


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


def describe_probe(stage, run_id=PROBE_RUN_ID, *, gpus=1, preemptible=False):
    """Describe an isolated operator probe, not qualified canonical serving."""
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.step5_job import METHODS, planned_profile, profile_arguments
    from frontier.profiling.step5.inputs import read_profile_inputs

    if gpus != 1:
        raise ValueError("The small probe measures one isolated TP-local GPU")
    # Reuse path validation before reading any model or workload inputs.
    manifest = describe_job(stage, run_id, worker_command="", gpus=gpus,
                            gpu_tag="H200", charged_group="step_main", preemptible=preemptible)
    hashes = {"SHA256SUMS": hashlib.sha256(
        (REPO_ROOT / WORKLOAD / "SHA256SUMS").read_bytes()).hexdigest()}
    if hashes["SHA256SUMS"] != WORKLOAD_SUMS_SHA256:
        raise ValueError("The probe requires the immutable D18 common workload checksums")
    for line in (REPO_ROOT / WORKLOAD / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        actual = hashlib.sha256((REPO_ROOT / WORKLOAD / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Common workload checksum differs: {name}")
        hashes[name] = actual
    if hashes["manifest.json"] != WORKLOAD_MANIFEST_SHA256:
        raise ValueError("The probe requires the immutable D18 common workload")
    config = ModelConfig.from_model_name("stepfun-ai/Step-5-Preview")
    cells = read_profile_inputs(REPO_ROOT / PROBE_INPUTS, cudagraph_capture_sizes=(1, 2))
    for cell in cells:
        cell.workload(config, block_size=16, max_model_len=7168)
    output = CLOUD_ROOT / run_id
    linear_output = output / "linear"
    tokens = tuple(sorted({cell.num_tokens for cell in cells}))
    linear_plan = planned_profile(linear_output, tokens=tokens, device="h200")
    attention_args = [
        "--model-path", str(Path("/data/ycfeng/tmp/step5_fixed_probe") / run_id / "model"),
        "--device", "h200",
        "--tensor-parallel-size", "8", "--block-size", "16", "--max-model-len", "7168",
        "--input-file", str(PROBE_INPUTS), "--disable-chunked-prefill",
        "--cudagraph-capture-sizes", "1", "2", "--warmup-iterations", "3",
        "--profile-iterations", "10", "--max-time-cv", "0.05",
        "--min-timing-window-ms", "0.5", "--seed", "0",
        "--output-dir", str(output / "attention"),
    ]
    q = shlex.quote
    worker = [
        "set -euo pipefail",
        f"export TMPDIR={q('/data/ycfeng/tmp/step5_fixed_probe/' + run_id)}",
        'mkdir -p "$TMPDIR/model"', f"test ! -e {q(str(output))}",
        f"mkdir -p {q(str(output.parent))}", f"mkdir {q(str(output))}",
        'cp data/config/models/stepfun-ai__Step-5-Preview.json "$TMPDIR/model/config.json"',
        'export PYTHONPATH="$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"',
        "export VLLM_USE_BREAKABLE_CUDAGRAPH=1",
        f"(cd {q(str(WORKLOAD))} && sha256sum -c SHA256SUMS)",
    ]
    command = shlex.join([
        "bash", "tests/integration/step5_profiling/run_on_worker.sh",
        "-m", "frontier.profiling.step5.main", *attention_args,
    ])
    worker.append(command + f" 2>&1 | tee {q(str(output / 'attention_worker.log'))}")
    for method in METHODS:
        worker.append(shlex.join([
            "bash", "tests/integration/step5_profiling/run_on_worker.sh",
            "-m", "frontier.profiling.linear_op.main",
            *profile_arguments(linear_output, method, tokens=tokens, device="h200"),
        ]) + f" 2>&1 | tee {q(str(output / ('linear_' + method + '_worker.log')))}")
    manifest = describe_job(stage, run_id, worker_command="\n".join(worker), gpus=gpus,
                            gpu_tag="H200", charged_group="step_main", preemptible=preemptible)
    return {
        **manifest, "purpose": "SMALL_OPERATOR_PROBE_ONLY_NOT_CANONICAL_SERVING_QUALIFICATION",
        "verification_scripts": ["verify_fixed_probe.py", "judge_attention_smoke.py"],
        "workload": {"path": str(WORKLOAD), "upstream_commit": "68f891b",
                     "local_commit": "958d81a", "sha256": hashes},
        "attention_arguments": attention_args,
        "native_probe_policy": {
            "max_num_seqs": 2, "max_model_len": 7168, "max_num_batched_tokens": 14336,
            "enable_chunked_prefill": False, "enable_prefix_caching": False,
            "cudagraph_capture_sizes": [1, 2], "cudagraph_mode": "FULL_AND_PIECEWISE",
            "breakable_cudagraph": True, "load_format": "dummy", "mtp": False,
            "scope": "one native attention layer, TP8-local shapes, real process groups TP1",
        },
        "batch_width_basis": "Explicit isolated max_num_seqs=2 and exact native capture buckets 1,2; not QPS or canonical DP4 occupancy",
        "context_convention": "context_lens excludes current query; last decode context 7167 gives sequence length 7168",
        "sparse_context_limit": "DSA probes use region-aligned 6144,6656,7160; partial-region contexts remain unsupported for repeated timing. Only SWA probes sequence length 7168",
        "d_tail_status": "16-token query after 6128 prefix is an observed-range transfer hypothesis, not canonical PD qualification",
        "expected_attention_event_rows": len(cells),
        "expected_attention_gpu_active_rows": sum(c.graph_mode == "none" for c in cells),
        "linear_plan": linear_plan,
        "expected_linear_rows": 2 * linear_plan["expected_rows_per_method"],
        "admitted": False,
        "blocks": ["Canonical TP8/DP4/EP32 P32/D32 runtime and capacity remain unqualified",
                   "Linear per-operator precision correction is CPU-verified only until fresh native execution",
                   "Attention event priming requires fresh native timing; prior 5% failures remain failures"],
    }


def stage_probe(stage, run_id=PROBE_RUN_ID, *, dependency_site, gpus=1, preemptible=False):
    manifest = describe_probe(stage, run_id, gpus=gpus, preemptible=preemptible)
    return stage_package(stage, manifest, repo_root=REPO_ROOT, dependency_site=dependency_site,
                         additional_paths=(WORKLOAD,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-id", default=PROBE_RUN_ID)
    parser.add_argument("--dependency-site", type=Path)
    parser.add_argument("--fixed-workload-probe", action="store_true")
    parser.add_argument("--preemptible", action="store_true",
                        help="Use a preemptible H200 fixed-workload probe under the D21 reservation policy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    if args.preemptible and not args.fixed_workload_probe:
        parser.error("--preemptible requires --fixed-workload-probe")
    resources = {"gpus": args.gpus}
    if args.fixed_workload_probe:
        resources["preemptible"] = args.preemptible
    describe = describe_probe if args.fixed_workload_probe else describe_smoke
    stage = stage_probe if args.fixed_workload_probe else stage_smoke
    if args.dry_run:
        manifest = describe(args.stage, args.run_id, **resources)
    else:
        if args.dependency_site is None:
            parser.error("staging requires --dependency-site")
        manifest = stage(args.stage, args.run_id, dependency_site=args.dependency_site, **resources)
    print(json.dumps({**{k: v for k, v in manifest.items() if k != "sha256"},
                       "staged_files": len(manifest.get("sha256", {}))}, indent=2))


if __name__ == "__main__":
    main()
