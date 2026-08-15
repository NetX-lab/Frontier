#!/usr/bin/env python3
"""Real-data, non-dummy E2E matrix for the MoE EP rank-staggering contract.

The harness deliberately invokes the checked-in architecture example wrappers.  It
does not change simulator semantics, invent profile rows, or convert a failed run
to a pass.  Generation is deterministic so the same manifest can be replayed on a
read-only baseline worktree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARCHITECTURES = ("co-location", "pd-disaggregation", "pd-af-disaggregation")
MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "dense": {
        "model_name": "llama2_7b_dense_example",
        "total_experts": 1,
        "router_topk": 1,
    },
    "moe": {
        "model_name": "Phi-tiny-MoE-instruct",
        "total_experts": 16,
        "router_topk": 2,
    },
    "mixed": {
        "model_name": "step-moe-noquant-small",
        "total_experts": 24,
        "router_topk": 3,
    },
}
ROUTING_DISTRIBUTIONS = ("balanced", "random", "skewed", "zipf")
WORKLOADS: Mapping[str, tuple[int, int, int]] = {
    "prefill-heavy": (8, 1, 1),
    "decode-heavy": (1, 4, 1),
    "mixed": (4, 4, 2),
    "zero-routed": (1, 2, 1),
}
VARIANTS = (
    ("balanced", "prefill-heavy"),
    ("random", "prefill-heavy"),
    ("skewed", "prefill-heavy"),
    ("zipf", "prefill-heavy"),
    ("balanced", "decode-heavy"),
    ("random", "decode-heavy"),
    ("skewed", "decode-heavy"),
    ("zipf", "decode-heavy"),
    ("balanced", "mixed"),
    ("random", "mixed"),
    ("skewed", "zero-routed"),
    ("zipf", "zero-routed"),
)


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    baseline_case_id: str
    architecture: str
    model_kind: str
    model_name: str
    device: str
    routing_distribution: str
    seed: int
    workload_kind: str
    prefill_tokens: int
    decode_tokens: int
    num_requests: int
    ep_size: int
    moe_tensor_parallel_size: int
    total_experts: int
    router_topk: int
    replica_count: int
    prefill_replicas: int
    decode_replicas: int
    decode_attn_replicas: int
    decode_ffn_replicas: int
    attn_tensor_parallel_size: int
    prefill_attn_tensor_parallel_size: int
    decode_attn_tensor_parallel_size: int
    prefill_moe_tensor_parallel_size: int
    decode_moe_tensor_parallel_size: int
    prefill_moe_expert_parallel_size: int
    decode_moe_expert_parallel_size: int
    total_cards: int
    num_layers: int
    moe_layer_ids: tuple[int, ...]

    @property
    def is_moe(self) -> bool:
        return self.model_kind != "dense"

    @property
    def expects_zero_routed_lane(self) -> bool:
        return self.is_moe and self.workload_kind == "zero-routed" and self.ep_size > 1


def _model_layer_shape(model_name: str) -> tuple[int, tuple[int, ...]]:
    # Imported lazily so manifest generation remains usable from a minimal shell
    # while still failing explicitly when the simulator environment is absent.
    from frontier.config.model_config import BaseModelConfig

    config = BaseModelConfig.create_from_name(model_name)
    return int(config.num_layers), tuple(int(x) for x in config.get_moe_layer_ids())


def _case_cards(architecture: str, replica_count: int, ep_size: int) -> tuple[int, dict[str, int]]:
    if architecture == "co-location":
        attn_tp = ep_size
        return replica_count * attn_tp, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": replica_count,
            "decode_replicas": replica_count,
            "decode_attn_replicas": replica_count,
            "decode_ffn_replicas": replica_count,
        }
    if architecture == "pd-disaggregation":
        attn_tp = ep_size
        return 2 * replica_count * attn_tp, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": replica_count,
            "decode_replicas": replica_count,
            "decode_attn_replicas": replica_count,
            "decode_ffn_replicas": replica_count,
        }
    if architecture == "pd-af-disaggregation":
        # Decode-attention is an independent role domain.  Keep it at TP=1 in
        # this matrix so the FFN EP capacity is the dimension under test.
        role_replicas = replica_count if ep_size == 1 else min(replica_count, 2)
        prefill_attn_tp = ep_size
        decode_attn_tp = 1
        cards = role_replicas * (prefill_attn_tp + decode_attn_tp + ep_size)
        return cards, {
            "attn_tp": prefill_attn_tp,
            "prefill_attn_tp": prefill_attn_tp,
            "decode_attn_tp": decode_attn_tp,
            "prefill_replicas": role_replicas,
            "decode_replicas": role_replicas,
            "decode_attn_replicas": role_replicas,
            "decode_ffn_replicas": role_replicas,
        }
    raise ValueError(f"unsupported architecture: {architecture}")


def build_matrix(repo_root: Path) -> list[MatrixCase]:
    """Build the deterministic 108-case matrix and validate its topology."""

    cases: list[MatrixCase] = []
    for architecture in ARCHITECTURES:
        for model_kind, spec in MODEL_SPECS.items():
            num_layers, moe_layer_ids = _model_layer_shape(str(spec["model_name"]))
            for variant_index, (distribution, workload_kind) in enumerate(VARIANTS):
                is_moe = model_kind != "dense"
                ep_size = (1, 2, 4)[variant_index % 3] if is_moe else 1
                replica_count = (1, 2, 4)[variant_index % 3]
                total_cards, topology = _case_cards(architecture, replica_count, ep_size)
                if total_cards > 32:
                    raise AssertionError(
                        f"matrix topology exceeds 32 cards: {architecture} {model_kind} "
                        f"variant={variant_index} cards={total_cards}"
                    )
                prefill_tokens, decode_tokens, num_requests = WORKLOADS[workload_kind]
                case_id = f"{architecture.replace('-', '_')}_{model_kind}_v{variant_index:02d}"
                baseline_id = f"{architecture.replace('-', '_')}_{model_kind}_v00"
                cases.append(
                    MatrixCase(
                        case_id=case_id,
                        baseline_case_id=baseline_id,
                        architecture=architecture,
                        model_kind=model_kind,
                        model_name=str(spec["model_name"]),
                        device="h800",
                        routing_distribution=distribution if is_moe else "balanced",
                        seed=42 + variant_index,
                        workload_kind=workload_kind,
                        prefill_tokens=prefill_tokens,
                        decode_tokens=decode_tokens,
                        num_requests=num_requests,
                        ep_size=ep_size,
                        moe_tensor_parallel_size=1,
                        total_experts=int(spec["total_experts"]),
                        router_topk=int(spec["router_topk"]),
                        replica_count=replica_count,
                        prefill_replicas=topology["prefill_replicas"],
                        decode_replicas=topology["decode_replicas"],
                        decode_attn_replicas=topology["decode_attn_replicas"],
                        decode_ffn_replicas=topology["decode_ffn_replicas"],
                        attn_tensor_parallel_size=1 if not is_moe else topology["attn_tp"],
                        prefill_attn_tensor_parallel_size=topology["prefill_attn_tp"],
                        decode_attn_tensor_parallel_size=topology["decode_attn_tp"],
                        prefill_moe_tensor_parallel_size=1,
                        decode_moe_tensor_parallel_size=1,
                        prefill_moe_expert_parallel_size=ep_size,
                        decode_moe_expert_parallel_size=ep_size,
                        total_cards=total_cards if is_moe else _case_cards(architecture, replica_count, 1)[0],
                        num_layers=num_layers,
                        moe_layer_ids=moe_layer_ids,
                    )
                )
    if len(cases) != 108:
        raise AssertionError(f"expected 108 matrix cases, got {len(cases)}")
    return cases


def _script_for_case(case: MatrixCase, repo_root: Path) -> Path:
    root = repo_root / "examples" / "architecture"
    if case.model_kind == "dense":
        names = {
            "co-location": "co-location/offline/dense_model_basic.sh",
            "pd-disaggregation": "pdd/offline/dense_model_basic.sh",
            "pd-af-disaggregation": "pd-af-disagg/offline/dense_model_basic.sh",
        }
    else:
        names = {
            "co-location": "co-location/offline/moe_model_basic.sh",
            "pd-disaggregation": "pdd/offline/moe_model_basic.sh",
            "pd-af-disaggregation": "pd-af-disagg/offline/moe_model_ep.sh",
        }
    path = root / names[case.architecture]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_shell_command(
    case: MatrixCase, repo_root: Path, output_root: Path
) -> tuple[str, dict[str, str]]:
    """Return an exact wrapper command and its environment for one case."""

    script = _script_for_case(case, repo_root)
    metrics_root = output_root / case.case_id / "metrics"
    env = {key: value for key, value in os.environ.items()}
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "PYTHON_BIN": sys.executable,
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "MODEL_NAME": case.model_name,
            "ENABLE_DUMMY_MODE": "false",
            "DECODE_CUDA_GRAPH_MODE": "none",
            "ENABLE_CHUNKED_PREFILL": "false",
            "NUM_REQUESTS": str(case.num_requests),
            "PREFILL_TOKENS": str(case.prefill_tokens),
            "DECODE_TOKENS": str(case.decode_tokens),
            "QPS": "1.0",
            "RUN_ID": case.case_id,
            "METRICS_OUTPUT_DIR": str(metrics_root),
            "MOE_ROUTING_DISTRIBUTION_TYPE": case.routing_distribution,
            "MOE_ROUTING_SEED": str(case.seed),
            "TOTAL_EXPERTS": str(case.total_experts),
            "ROUTER_TOPK": str(case.router_topk),
            "MAX_TOKENS_IN_BATCH": "64",
            "LONG_PREFILL_TOKEN_THRESHOLD": "0",
        }
    )
    if case.architecture == "co-location":
        env.update(
            {
                "NUM_REPLICAS": str(case.replica_count),
                "ATTN_TP": str(case.attn_tensor_parallel_size),
                "MOE_TP": str(case.moe_tensor_parallel_size),
                "MOE_EP": str(case.ep_size),
                "PP": "1",
                "DEVICE": case.device,
            }
        )
    elif case.architecture == "pd-disaggregation":
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_REPLICAS": str(case.decode_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_DEVICE": case.device,
                "PREFILL_PP": "1",
                "DECODE_PP": "1",
            }
        )
    else:
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_ATTN_REPLICAS": str(case.decode_attn_replicas),
                "DECODE_FFN_REPLICAS": str(case.decode_ffn_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_FFN_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_FFN_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_ATTN_DEVICE": case.device,
                "DECODE_FFN_DEVICE": case.device,
                "PREFILL_PP": "1",
                "DECODE_ATTN_PP": "1",
                "DECODE_FFN_PP": "1",
            }
        )
    command = shlex.join(["bash", str(script)])
    return command, env


def validate_profile_inputs(case: MatrixCase, root: Path) -> list[Path]:
    """Require real profiling rows; never substitute dummy or synthetic data."""

    profile_root = root / "data" / "profiling" / "compute"
    if (root / case.device).is_dir() and not (profile_root).is_dir():
        profile_root = root
    model_dir = profile_root / case.device / case.model_name
    required = [model_dir / "attention.csv", model_dir / "linear_op.csv"]
    if case.is_moe:
        required.append(model_dir / "moe.csv")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required non-dummy profiling files:\n" + "\n".join(str(p) for p in missing)
        )
    return required


def _find_metrics_dir(output_root: Path, case: MatrixCase) -> Path:
    root = output_root / case.case_id
    candidates = sorted(root.rglob("system_metrics.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one system_metrics.json for {case.case_id}, found {len(candidates)}"
        )
    return candidates[0].parent


def _finite_metric_values(value: Any) -> Iterable[float]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _finite_metric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _finite_metric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"non-finite or negative metric value: {value!r}")
        yield number


def check_case_log(
    case: MatrixCase,
    log_path: Path,
    metrics_dir: Path,
    *,
    strict_layers: bool = False,
) -> dict[str, Any]:
    """Check workflow evidence and numeric metrics for one completed run."""

    errors: list[str] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in text:
        errors.append("Traceback")
    if "Simulation completed successfully." not in text:
        errors.append("missing success marker")
    if "Dummy Mode: false" not in text:
        errors.append("dummy mode was not explicitly disabled")
    if re.search(r"(?i)(synthetic latency|scaling factor|visibility multiplier)", text):
        errors.append("forbidden synthetic/scaling wording in log")

    layer_ids = sorted({int(match) for match in re.findall(r"layer_id=(\d+)", text)})
    if not layer_ids:
        errors.append("no layer_id trace")
    if strict_layers:
        expected = list(range(case.num_layers))
        if layer_ids != expected:
            errors.append(f"layer ids are not contiguous expected={expected} actual={layer_ids}")

    moe_trace_count = text.count("[MOE]")
    ep_participant_records = text.count("per_expert_tokens extracted:")
    if case.is_moe:
        if "moe_grouped_gemm" not in text or "moe_shuffling" not in text:
            errors.append("missing MoE grouped-gemm/shuffling trace")
        if case.architecture == "pd-af-disaggregation" and ep_participant_records == 0:
            errors.append("missing DECODE_FFN EP participant maps")
        if case.expects_zero_routed_lane and "0}" not in text and ": 0" not in text:
            errors.append("zero-routed case has no zero-token participant evidence")

    metric_path = metrics_dir / "system_metrics.json"
    numeric_metric_count = 0
    metrics: dict[str, Any] = {}
    if not metric_path.is_file():
        errors.append(f"missing metrics file: {metric_path}")
    else:
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            numeric_metric_count = sum(1 for _ in _finite_metric_values(metrics))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid metrics: {exc}")

    def _stat_value(name: str) -> float | None:
        value = metrics.get(name, {})
        if isinstance(value, Mapping):
            candidate = value.get("mean")
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": "; ".join(errors),
        "layer_ids": layer_ids,
        "moe_trace_count": moe_trace_count,
        "ep_participant_records": ep_participant_records,
        "numeric_metric_count": numeric_metric_count,
        "ttft_mean_ms": _stat_value("ttft_statistics"),
        "e2e_mean_ms": _stat_value("request_e2e_time_statistics"),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_manifest(path: Path, cases: Sequence[MatrixCase]) -> None:
    _write_jsonl(path, [asdict(case) for case in cases])


def _load_manifest(path: Path) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["moe_layer_ids"] = tuple(record["moe_layer_ids"])
        cases.append(MatrixCase(**record))
    return cases


def run_cases(
    cases: Sequence[MatrixCase],
    repo_root: Path,
    output_root: Path,
    results_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    timeout_seconds: int = 600,
    continue_on_failure: bool = False,
) -> list[dict[str, Any]]:
    selected = list(cases[start : (start + limit) if limit is not None else None])
    results: list[dict[str, Any]] = []
    for case in selected:
        validate_profile_inputs(case, repo_root)
        command, env = build_shell_command(case, repo_root, output_root)
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / f"{case.case_id}.log"
        metadata_path = case_dir / "case_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "command": command,
                    "environment": {
                        key: env[key]
                        for key in (
                            "MODEL_NAME",
                            "ENABLE_DUMMY_MODE",
                            "DECODE_CUDA_GRAPH_MODE",
                            "MOE_ROUTING_DISTRIBUTION_TYPE",
                            "MOE_ROUTING_SEED",
                            "TOTAL_EXPERTS",
                            "ROUTER_TOPK",
                        )
                        if key in env
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write(f"MATRIX_COMMAND: {command}\n")
            stream.flush()
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=repo_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
                exit_code = int(completed.returncode)
            except subprocess.TimeoutExpired:
                exit_code = 124
                stream.write(f"MATRIX_TIMEOUT after {timeout_seconds}s\n")
        elapsed = time.monotonic() - started
        check: dict[str, Any]
        try:
            metrics_dir = _find_metrics_dir(output_root, case)
            check = check_case_log(case, log_path, metrics_dir, strict_layers=True)
            metrics_path = str(metrics_dir)
        except (FileNotFoundError, OSError) as exc:
            check = {"status": "FAIL", "errors": str(exc)}
            metrics_path = ""
        status = "PASS" if exit_code == 0 and check.get("status") == "PASS" else "FAIL"
        result = {
            "case_id": case.case_id,
            "architecture": case.architecture,
            "model_kind": case.model_kind,
            "total_cards": case.total_cards,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 3),
            "log_path": str(log_path),
            "metrics_path": metrics_path,
            "status": status,
            "check": check,
        }
        results.append(result)
        _write_jsonl(results_path, results)
        if status != "PASS" and not continue_on_failure:
            raise RuntimeError(f"matrix case failed: {json.dumps(result, sort_keys=True)}")
    return results


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "run"), default="generate")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("task_memory/task_2026-08-12_moe_ep_rank_stragger_analysis"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/ycfeng/tmp/frontier_non_dummy_matrix"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    task_dir = args.task_dir if args.task_dir.is_absolute() else repo_root / args.task_dir
    manifest_path = task_dir / "moe_ep_non_dummy_matrix_manifest.jsonl"
    results_path = task_dir / "moe_ep_non_dummy_matrix_results.jsonl"
    cases = build_matrix(repo_root)
    write_manifest(manifest_path, cases)
    print(f"manifest={manifest_path} cases={len(cases)}")
    if args.mode == "generate":
        return 0
    results = run_cases(
        cases,
        repo_root,
        args.output_root,
        results_path,
        start=args.start,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        continue_on_failure=args.continue_on_failure,
    )
    passed = sum(result["status"] == "PASS" for result in results)
    failed = len(results) - passed
    print(f"results={results_path} passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
