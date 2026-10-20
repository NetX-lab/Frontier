"""Package the Step linear-op smoke; the parent submits the staged launcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from collections import Counter
from pathlib import Path

from frontier.profiling.common.step5_job import (
    CLOUD_MOUNT, CLOUD_ROOT, IMAGE, STEPMIND_ENV, describe_job, stage_package,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "linear_smoke_20261020_01"
MODEL = "stepfun-ai/Step-5-Preview"
TOKENS = (1, 16, 64)
ATTN_TP, FFN_TP, MOE_TP = 8, 8, 1
METHODS = ("cuda_event", "record_function")


def profile_arguments(output, method):
    """Use the canonical producer, including its TP=1 replicated-row writer."""
    return [
        "--disable_ray", "--num_gpus", "1", "--models", MODEL,
        "--num_tokens_list", *map(str, TOKENS),
        "--num_tensor_parallel_workers", str(ATTN_TP),
        "--attn_tp", str(ATTN_TP), "--ffn_tp", str(FFN_TP), "--moe_tp", str(MOE_TP),
        "--precision", "BF16",
        "--profile_method", method, "--device", "h800", "--output_dir", str(output), "--yes",
    ]


def planned_profile(output):
    """Derive work and output rows from the producer's plan and row writer."""
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.main import (
        _build_profiling_plans, _materialize_profile_result_rows, _plan_summary,
    )
    from frontier.profiling.utils.replicated_ops import deduplicate_tp1_rows

    config = ModelConfig.from_model_name(MODEL)
    profile = config.get_model_architecture_profile()
    args = argparse.Namespace(
        attn_tp=[ATTN_TP], ffn_tp=[FFN_TP], moe_tp=[MOE_TP],
        num_tensor_parallel_workers=[ATTN_TP], disable_replicated=False, is_moe=False,
        include_target_embedded_mtp=False, sequence_parallel_moe_topology=None,
    )
    plans = _build_profiling_plans(args, config)
    rows = []
    for plan in plans:
        if not plan["enabled_ops"]:
            continue
        replicated = set(plan["replicated_ops"])
        for tokens in TOKENS:
            rows.extend(_materialize_profile_result_rows(
                {"time_stats": dict.fromkeys(plan["enabled_ops"]),
                 "num_tokens": tokens, "num_tensor_parallel_workers": plan["tp_size"],
                 "model_arch": config.model_arch,
                 "typed_operator_contracts": plan["typed_operator_contracts"]},
                measurement_type="CUDA_EVENT",
                should_split=bool(plan["tp_size"] > 1 and replicated),
                replicated_op_names=replicated, model_config=config,
            ))
    rows = deduplicate_tp1_rows(
        rows, tp1_key_fields=("num_tokens", "model_arch", "measurement_type"))
    return {
        "model": MODEL, "num_tokens": list(TOKENS),
        "profiling_precision": ModelConfig._dtype_to_str(config.dtype),
        "planned_activation_variants": {
            "dense": profile.resolve_ffn_activation(config, share_expert=False),
            "shared": profile.resolve_ffn_activation(config, share_expert=True),
        },
        "planned_normalization": {
            "implementation": profile.linear_norm_impl.value,
            "norm_dtype": config.norm_dtype,
            "zero_centered": config.zero_centered,
        },
        "csv_paths": {method: str(path) for method, path in csv_paths(output).items()},
        "plans": [_plan_summary(plan, list(TOKENS)) for plan in plans],
        "profiled_steps_per_method": sum(len(TOKENS) for p in plans if p["enabled_ops"]),
        "expected_rows_per_method": len(rows),
        "expected_rows_by_tp_per_method": dict(sorted(Counter(
            str(row["num_tensor_parallel_workers"]) for row in rows).items())),
        "planned_csv_rows": [
            {"num_tokens": row["num_tokens"],
             "num_tensor_parallel_workers": row["num_tensor_parallel_workers"],
             "operators": sorted(row["time_stats"]),
             "typed_operator_contracts": row["typed_operator_contracts"]}
            for row in rows],
    }


def csv_paths(output):
    from frontier.profiling.utils import build_profile_method_output_path

    return {method: build_profile_method_output_path(
        output_root=output, profiling_type="compute", hardware="h800",
        model_name=MODEL, op_name="linear_op", profile_method=method)
        for method in METHODS}


def describe_smoke(stage, run_id=RUN_ID, *, gpus=1):
    if gpus != 1:
        raise ValueError("The linear smoke measures TP-local shapes on one isolated GPU")
    output = CLOUD_ROOT / run_id
    q = shlex.quote
    worker = [
        "set -euo pipefail",
        f"export TMPDIR={q('/data/ycfeng/tmp/step5_linear_worker/' + run_id)}",
        'mkdir -p "$TMPDIR"',
        f"test ! -e {q(str(output))}",
        f"mkdir -p {q(str(output.parent))}", f"mkdir {q(str(output))}",
        'export PYTHONPATH="$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"',
        "python3 -c 'import pandas, dateutil, pytz, tzdata, yaml, tqdm, six; "
        "print(\"OFFLINE_DEPENDENCIES_READY\", pandas.__version__, pandas.__file__)'",
        "nvidia-smi -L",
        "if ! python3 -m frontier.profiling.linear_op.step5_job "
        f"--write-plan {q(str(output / 'planned_metadata.json'))}; then",
        '  echo "LINEAR_PROFILE_PLAN_UNVERIFIED" >&2', "fi",
    ]
    worker.extend(shlex.join([
        "bash", "tests/integration/step5_profiling/run_on_worker.sh",
        "-m", "frontier.profiling.linear_op.main", *profile_arguments(output, method),
    ]) for method in METHODS)
    # Reporting is not an extra GPU-workload acceptance gate.
    worker += [
        "if ! python3 -m frontier.profiling.linear_op.step5_job "
        f"--summarize {q(str(output))}; then",
        '  echo "LINEAR_PROFILE_SUMMARY_UNVERIFIED" >&2', "fi",
    ]
    manifest = describe_job(stage, run_id, worker_command="\n".join(worker), gpus=gpus,
                            cloud_root=CLOUD_ROOT, credential_env=STEPMIND_ENV)
    plan = planned_profile(output)
    return {
        **manifest, "profile_methods": list(METHODS), "profile_plan": plan,
        "expected_csv_rows": len(METHODS) * plan["expected_rows_per_method"],
        "csv_paths": plan["csv_paths"],
    }


def stage_smoke(stage, run_id=RUN_ID, *, dependency_site, gpus=1):
    manifest = describe_smoke(stage, run_id, gpus=gpus)
    return stage_package(stage, manifest, repo_root=REPO_ROOT, dependency_site=dependency_site)


def _typed_contracts(value):
    contracts = json.loads(value) if isinstance(value, str) else value
    if not isinstance(contracts, dict):
        raise ValueError("typed_operator_contracts must be a JSON object or dictionary")
    return contracts


def _positive_finite(value, *, integral=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return (not isinstance(value, bool) and math.isfinite(number) and number > 0
            and (not integral or number == int(number)))


def _row_key(row):
    values = [row[name] for name in ("num_tokens", "num_tensor_parallel_workers")]
    if not all(_positive_finite(value, integral=True) for value in values):
        raise ValueError("num_tokens and num_tensor_parallel_workers must be positive integers")
    if not isinstance(row["measurement_type"], str) or not row["measurement_type"]:
        raise ValueError("measurement_type must be a nonempty string")
    return (*[int(float(value)) for value in values], row["measurement_type"])


def summarize_csv(path):
    """Read only ordinary CSV output; never import torch or inspect a GPU."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        operators = sorted({name.split(".")[1] for name in columns if name.startswith("time_stats.")})
        records = list(reader)
    rows = []
    for record in records:
        key = _row_key(record)
        timings = {name: {stat: record[f"time_stats.{name}.{stat}"] for stat in ("mean", "count")
                          if f"time_stats.{name}.{stat}" in record} for name in operators}
        rows.append({
            "num_tokens": key[0], "num_tensor_parallel_workers": key[1], "measurement_type": key[2],
            "operators": [name for name, stats in timings.items()
                          if _positive_finite(stats.get("mean")) and _positive_finite(stats.get("count"), integral=True)],
            "operator_stats": timings,
            "typed_operator_contracts": _typed_contracts(record["typed_operator_contracts"]),
            "linear_backends": json.loads(record.get("linear_backends") or "{}"),
        })
    return {
        "path": str(path), "num_rows": len(rows), "rows": rows,
        "operators": sorted({name for row in rows for name in row["operators"]}),
        "rows_by_tp": dict(sorted(Counter(
            str(row["num_tensor_parallel_workers"]) for row in rows).items())),
    }


def _coverage_errors(plan, tables):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    errors = []
    for label, methods in (("planned", plan["csv_paths"]), ("read", tables)):
        if set(methods) != set(METHODS):
            errors.append({"issue": f"{label} methods differ", "expected": list(METHODS), "actual": sorted(methods)})
    for method, measurement in (("cuda_event", "CUDA_EVENT"), ("record_function", "KERNEL_ONLY")):
        expected, actual = {}, {}
        for row in plan["planned_csv_rows"]:
            key = _row_key({**row, "measurement_type": measurement})
            if key in expected:
                errors.append({"method": method, "row_key": key, "issue": "duplicate planned row key"})
            expected[key] = row
        for row in tables.get(method, {}).get("rows", []):
            key = _row_key(row)
            if key in actual:
                errors.append({"method": method, "row_key": key, "issue": "duplicate actual row key"})
            actual[key] = row
        for key in sorted(set(expected) - set(actual)):
            errors.append({"method": method, "row_key": key, "issue": "missing row"})
        for key in sorted(set(actual) - set(expected)):
            errors.append({"method": method, "row_key": key, "issue": "unexpected row"})
        for key, planned in expected.items():
            context = {"method": method, "row_key": key}
            operators = set(planned["operators"])
            contracts = _typed_contracts(planned["typed_operator_contracts"])
            for operator in sorted(operators ^ set(contracts)):
                errors.append({**context, "operator": operator, "issue": "planned operator/contract coverage differs"})
            if key not in actual:
                continue
            observed = actual[key]
            for operator in sorted(operators ^ set(observed["operators"])):
                errors.append({**context, "operator": operator, "issue": "measured operator coverage differs",
                               "timing": observed["operator_stats"].get(operator, {}),
                               "required": "positive finite mean and positive integral count"})
            observed_contracts = _typed_contracts(observed["typed_operator_contracts"])
            for operator in sorted(operators ^ set(observed_contracts)):
                errors.append({**context, "operator": operator, "issue": "actual operator/contract coverage differs"})
            for operator in sorted(operators & set(contracts) & set(observed_contracts)):
                if canonical(contracts[operator]) != canonical(observed_contracts[operator]):
                    errors.append({**context, "operator": operator, "issue": "typed contract differs from plan"})
    return errors


def summarize_profiles(output):
    """Separate readable artifacts from planned coverage; neither controls a job."""
    output = Path(output)
    result = {"csv": {}, "errors": {}, "coverage_errors": [], "admitted": False,
              "read_complete": False, "complete": False, "row_counts_match_plan": False}
    plan_path = output / "planned_metadata.json"
    try:
        result["plan"] = json.loads(plan_path.read_text())
        paths = result["plan"]["csv_paths"]
        if not isinstance(paths, dict):
            raise ValueError("csv_paths must be a dictionary of method paths")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["errors"][str(plan_path)] = str(error)
        paths = {}
    for method, path in paths.items():
        try:
            result["csv"][method] = summarize_csv(path)
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            result["errors"][str(path)] = str(error)
    result["read_complete"] = not result["errors"]
    if "plan" in result:
        try:
            expected = result["plan"]["expected_rows_per_method"]
            if not _positive_finite(expected, integral=True):
                raise ValueError("expected_rows_per_method must be a positive integer")
            result["row_counts_match_plan"] = set(result["csv"]) == set(METHODS) and all(
                item["num_rows"] == int(expected) for item in result["csv"].values())
            result["coverage_errors"] = _coverage_errors(result["plan"], result["csv"])
        except (KeyError, ValueError, TypeError) as error:
            result["coverage_errors"].append({"issue": f"invalid planned coverage: {error}"})
    result["complete"] = (result["read_complete"] and result["row_counts_match_plan"]
                          and not result["coverage_errors"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write-plan", type=Path)
    action.add_argument("--summarize", type=Path)
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dependency-site", type=Path)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args(argv)
    if args.write_plan:
        result = planned_profile(args.write_plan.parent)
        with args.write_plan.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    elif args.summarize:
        result = summarize_profiles(args.summarize)
        with (args.summarize / "summary.json").open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print("LINEAR_PROFILE_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
        return
    else:
        if args.stage is None or (not args.dry_run and args.dependency_site is None):
            parser.error("--stage is required; staging also requires --dependency-site")
        result = (describe_smoke(args.stage, args.run_id, gpus=args.gpus) if args.dry_run
                  else stage_smoke(args.stage, args.run_id, dependency_site=args.dependency_site,
                                   gpus=args.gpus))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
