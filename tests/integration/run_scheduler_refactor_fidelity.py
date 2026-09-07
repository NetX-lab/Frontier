"""Compare release example scenarios between two Frontier worktrees.

This is a dummy-predictor refactor parity check, not trained-model validation.
Each subprocess writes its own config, logs, request metrics and stage ledger.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def scenarios():
    cases = []

    def add(arch, mode, recipe, variant, settings=None, flags=None):
        suffix = "_online" if mode == "online" else ""
        cases.append({
            "id": f"{arch}_{mode}_{recipe}_{variant}",
            "script": f"examples/architecture/{arch}/{mode}/{recipe}{suffix}.sh",
            "settings": settings or {},
            "flags": flags or [],
            "expected_requests": 2 if "prefix_caching" in recipe else int(
                (settings or {}).get("NUM_REQUESTS", "1")
            ),
        })

    variants = [
        ("short", {}),
        ("chunked", {"NUM_REQUESTS": "3", "PREFILL_TOKENS": "80",
                     "DECODE_TOKENS": "6", "QPS": "10"}),
        ("unchunked", {"NUM_REQUESTS": "4", "PREFILL_TOKENS": "32",
                       "DECODE_TOKENS": "3", "QPS": "100",
                       "ENABLE_CHUNKED_PREFILL": "false",
                       "LONG_PREFILL_TOKEN_THRESHOLD": "0"}),
    ]
    for arch in ("co-location", "pdd", "pd-af-disagg"):
        for mode in ("offline", "online"):
            for recipe in ("dense_model_basic", "moe_model_basic"):
                for name, settings in variants:
                    add(arch, mode, recipe, name, settings)
    for arch in ("co-location", "pdd"):
        for mode in ("offline", "online"):
            for recipe in ("moe_spec_dec", "moe_prefix_caching", "thinking_mode_basic"):
                add(arch, mode, recipe, "feature")
            if arch == "co-location":
                flags = ["--replica_config_attn_tensor_parallel_size", "4",
                         "--replica_config_attn_dp", "2",
                         "--replica_config_moe_tensor_parallel_size", "1",
                         "--replica_config_moe_expert_parallel_size", "8"]
            else:
                flags = ["--replica_config_attn_dp", "2"]
                for role in ("prefill", "decode"):
                    prefix = f"--cluster_config_{role}_replica_config_"
                    flags.extend([prefix + "attn_tensor_parallel_size", "4",
                                  prefix + "moe_tensor_parallel_size", "1",
                                  prefix + "moe_expert_parallel_size", "8"])
            add(arch, mode, "moe_model_basic", "dp2_ep8", flags=flags)
    for mode in ("offline", "online"):
        for recipe in ("dense_cuda_graph", "moe_cuda_graph", "moe_model_ep"):
            add("pd-af-disagg", mode, recipe, "feature")
    return cases


def compare(left, right, path="root"):
    """Compare all fields, retaining IDs, row order and intermediate timestamps."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), f"{path}: fields differ"
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]")
    elif left != right:
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError):
            raise AssertionError(f"{path}: {left!r} != {right!r}") from None
        assert (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=1e-12, abs_tol=1e-9
        ), f"{path}: {left!r} != {right!r}"


def load_artifact(path):
    if path.suffix == ".csv":
        with path.open() as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()]
    return json.loads(path.read_text())


def run_case(case, roots, output):
    result = {**case, "runs": {}}
    artifacts = {}
    try:
        for label, root in roots.items():
            run_dir = output / case["id"] / label
            run_dir.mkdir(parents=True)
            settings = {
                "PYTHON_BIN": sys.executable, "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(root), "WANDB_DISABLED": "true", "VIDUR_DISABLE_WANDB": "1",
                "TMPDIR": str(output), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "ENABLE_DUMMY_MODE": "true", "DUMMY_EXEC_TIME_MS": "1",
                "NUM_REQUESTS": "1", "PREFILL_TOKENS": "16", "DECODE_TOKENS": "4",
                "QPS": "1", "DECODE_CUDA_GRAPH_MODE": "none",
                "ENABLE_CHUNKED_PREFILL": "true", "LONG_PREFILL_TOKEN_THRESHOLD": "64",
                "METRICS_OUTPUT_DIR": str(run_dir / "metrics"), "RUN_ID": "review",
                **case["settings"],
            }
            command = ["bash", str(root / case["script"]), "--",
                       "--metrics_config_cache_dir", str(run_dir / "cache"), *case["flags"]]
            (run_dir / "invocation.json").write_text(json.dumps({
                "cwd": str(root), "command": command, "environment_overrides": settings,
            }, indent=2) + "\n")
            with (run_dir / "run.log").open("w") as log:
                proc = subprocess.run(command, cwd=root, env={**os.environ, **settings},
                                      stdout=log, stderr=subprocess.STDOUT, timeout=180)
            result["runs"][label] = {"exit_code": proc.returncode, "path": str(run_dir)}
            if proc.returncode:
                continue
            metric_paths = list((run_dir / "metrics").rglob("request_metrics.csv"))
            assert len(metric_paths) == 1, f"{label}: expected one request metrics file"
            metrics = metric_paths[0].parent
            rows = load_artifact(metric_paths[0])
            assert len(rows) == case["expected_requests"], (
                f"{label}: completed {len(rows)}, expected {case['expected_requests']}"
            )
            result["runs"][label]["completed_requests"] = len(rows)
            artifacts[label] = {
                str(path.relative_to(metrics)): load_artifact(path)
                for path in sorted(metrics.rglob("*"))
                if path.suffix in (".csv", ".jsonl") or path.name == "system_metrics.json"
            }
            assert "frontier_stage_batch_ledger.jsonl" in artifacts[label]
            assert artifacts[label]["frontier_stage_batch_ledger.jsonl"]
        assert len(artifacts) == 2, "one or both simulator runs failed; see run logs"
        compare(artifacts["baseline"], artifacts["candidate"])
        result["status"] = "PASS"
        result["compared_artifacts"] = list(artifacts["candidate"])
        result["stage_ledger_rows"] = len(artifacts["candidate"]["frontier_stage_batch_ledger.jsonl"])
        result["request_metrics"] = artifacts["candidate"]["request_metrics.csv"]
    except (AssertionError, subprocess.TimeoutExpired) as error:
        result["status"] = "FAIL"
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", action="append", help="Run selected case IDs only")
    args = parser.parse_args()
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = scenarios()
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        assert len(cases) == len(set(args.case)), "unknown case ID"
    (output / "manifest.json").write_text(json.dumps({
        "roots": {key: str(value) for key, value in roots.items()},
        "revisions": {key: subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=value, text=True
        ).strip() for key, value in roots.items()},
        "cases": cases,
    }, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, case, roots, output) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{len(results)}/{len(cases)} {result['status']} {result['id']} "
                  f"{result.get('error', '')}", flush=True)
    failed = sum(result["status"] != "PASS" for result in results)
    print(f"{len(results) - failed} passed, {failed} failed; artifacts: {output}")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
