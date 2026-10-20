"""CPU-only Step linear smoke commands, source packages and CSV reporting."""

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from frontier.profiling.linear_op import step5_job as job


def rewrite_csv(path, edit):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    edit(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def update_csv_value(path, column, value):
    def update(rows):
        rows[0][column] = value

    rewrite_csv(path, update)


def test_plan_uses_existing_tp_domains_and_writer(tmp_path):
    plan = job.planned_profile(tmp_path)
    assert plan["num_tokens"] == [1, 16, 64]
    assert plan["profiled_steps_per_method"] == plan["expected_rows_per_method"] == 6
    assert plan["expected_rows_by_tp_per_method"] == {"1": 3, "8": 3}
    by_tp = {row["num_tensor_parallel_workers"]: set(row["operators"])
             for row in plan["planned_csv_rows"]}
    assert by_tp[1] == {"emb", "input_layernorm", "post_attention_layernorm", "step5_ffn_merge"}
    assert by_tp[8] == {"mlp_up_proj", "mlp_down_proj", "mlp_act",
                        "share_expert_up_proj", "share_expert_down_proj", "share_expert_act"}
    assert plan["profiling_precision"] == "BF16"
    assert json.loads(json.dumps(plan["planned_activation_variants"])) == {
        "dense": ["swiglustep", 7.0], "shared": ["swiglustep", 7.0]}
    assert plan["planned_normalization"] == {
        "implementation": "step5", "norm_dtype": "float32", "zero_centered": True}
    for row in plan["planned_csv_rows"]:
        assert set(row["typed_operator_contracts"]) == set(row["operators"])
    dense = plan["plans"][1]["operator_contracts"]["mlp_up_proj"]
    shared = plan["plans"][1]["operator_contracts"]["share_expert_up_proj"]
    assert dense["effective_ffn_width"] // 8 == 1728
    assert shared["effective_ffn_width"] // 8 == 192


@pytest.mark.parametrize("method", job.METHODS)
def test_commands_parse_as_canonical_linear_cli(tmp_path, monkeypatch, method):
    from frontier.profiling.linear_op import main as linear_main

    monkeypatch.setattr(sys, "argv", ["linear_op.main", *job.profile_arguments(tmp_path, method),
                                      "--dry_run"])
    args = linear_main.parse_args()
    assert args.models == [job.MODEL]
    assert args.num_gpus == 1 and args.disable_ray and args.skip_confirmation
    assert args.num_tokens_list == [1, 16, 64]
    assert args.num_tensor_parallel_workers == args.attn_tp == args.ffn_tp == [8]
    assert args.moe_tp == [1]
    assert args.precision == "BF16"
    assert not args.is_moe and not args.include_target_embedded_mtp and not args.disable_replicated
    dry_run = linear_main._dry_run(args, args.num_tokens_list)
    assert dry_run["models"][job.MODEL]["plans"] == job.planned_profile(tmp_path)["plans"]


def test_description_preserves_job_resources_and_separate_csvs(tmp_path):
    stage = tmp_path / "not_created"
    manifest = job.describe_smoke(stage)
    output = Path("/mnt/codesign-exp/ycfeng/frontier/step5_profiling/linear_smoke_20261020_01")
    assert manifest["image"].endswith(
        "sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
    assert manifest["gpu_tag"] == "H800" and manifest["charged_group"] == "codesign"
    assert (manifest["gpus"], manifest["cpus"], manifest["mem_gb"]) == (1, 8, 32)
    assert manifest["mounts"] == [f"local_nfs:{stage}:{stage}", job.CLOUD_MOUNT]
    assert manifest["cloud_output"] == str(output)
    assert manifest["credential_env"] == "/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh"
    assert not manifest["preemptible"] and manifest["timeout_seconds"] == 3600
    assert manifest["expected_csv_rows"] == 12
    assert manifest["csv_paths"] == {
        "cuda_event": str(output / "compute/h800/stepfun-ai/Step-5-Preview/linear_op.csv"),
        "record_function": str(output / "compute/h800/stepfun-ai/Step-5-Preview/linear_op_kernel_only.csv"),
    }
    worker = manifest["worker_command"]
    assert f"test ! -e {output}" in worker and f"mkdir {output}" in worker
    assert "OFFLINE_DEPENDENCIES_READY" in worker
    assert "--is_moe" not in worker and "--include_target_embedded_mtp" not in worker
    assert worker.count("-m frontier.profiling.linear_op.main") == 2
    assert worker.index("--profile_method cuda_event") < worker.index("--profile_method record_function")
    submission = shlex.split(manifest["submission_command"])
    assert submission[-1] == worker
    subprocess.run(["bash", "-n"], input=worker, text=True, check=True)
    assert not stage.exists()


def test_job_cli_dry_run_creates_no_stage(tmp_path, capsys):
    stage = tmp_path / "not_created"
    job.main(["--dry-run", "--stage", str(stage)])
    result = json.loads(capsys.readouterr().out)
    assert result["expected_csv_rows"] == 12
    assert result["profile_plan"]["profiled_steps_per_method"] == 6
    assert not stage.exists()


@pytest.mark.parametrize("run_id", ["", "../escape", "/absolute", ".", ".."])
def test_invalid_run_id_is_rejected_before_planning(tmp_path, monkeypatch, run_id):
    def unexpected(_):
        raise AssertionError("invalid paths must not build plans")

    monkeypatch.setattr(job, "planned_profile", unexpected)
    with pytest.raises(ValueError, match="run_id"):
        job.describe_smoke(tmp_path, run_id)


@pytest.mark.parametrize("gpus", [0, 2, 8])
def test_job_does_not_allocate_a_distributed_gpu_job(tmp_path, gpus):
    with pytest.raises(ValueError, match="one isolated GPU"):
        job.describe_smoke(tmp_path, gpus=gpus)


def test_shared_package_hashes_sources_dependencies_and_both_submit_scripts(tmp_path, monkeypatch):
    root, dependencies, stage = (tmp_path / name for name in ("repo", "site", "stage"))
    payload = {
        "frontier/probe.py": "# Source fixture\n",
        "frontier/__pycache__/probe.pyc": "ignored cache",
        "data/config/models/config.json": "{}\n",
        "tests/integration/step5_profiling/submit_profiling_job.py": "# Submit fixture\n",
        "tests/integration/step5_profiling/run_on_worker.sh": "# Worker fixture\n",
    }
    for name, content in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    dependencies.mkdir()
    (dependencies / "offline_dependency.py").write_text("# Offline fixture\n")
    (dependencies / "ignored.pyc").write_text("ignored cache")
    monkeypatch.setattr(job, "REPO_ROOT", root)
    monkeypatch.setattr(job, "STEPMIND_ENV", tmp_path / "credentials_not_read.sh")
    manifest = job.stage_smoke(stage, dependency_site=dependencies)
    assert not Path(manifest["credential_env"]).exists()
    assert {"submit.sh", "submit_with_env.sh", "frontier/probe.py",
            "offline_site/offline_dependency.py",
            "tests/integration/step5_profiling/submit_profiling_job.py",
            "tests/integration/step5_profiling/run_on_worker.sh"} <= manifest["sha256"].keys()
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in manifest["sha256"])
    assert json.loads((stage / "manifest.json").read_text()) == json.loads(json.dumps(manifest))
    assert set(manifest["sha256"]) == {
        str(path.relative_to(stage)) for path in stage.rglob("*")
        if path.is_file() and path.name != "manifest.json"}
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == digest
    wrapper = (stage / "submit_with_env.sh").read_text()
    assert f"source {manifest['credential_env']}" in wrapper
    assert "export STEPMIND_BACKEND=rjob" in wrapper
    assert "set +x" in wrapper
    subprocess.run(["bash", "-n", str(stage / "submit.sh")], check=True)
    subprocess.run(["bash", "-n", str(stage / "submit_with_env.sh")], check=True)
    with pytest.raises(FileExistsError):
        job.stage_smoke(stage, dependency_site=dependencies)


@pytest.mark.parametrize("failure,returncode,unverified", [
    ("", 0, ""), ("cuda_event", 23, ""), ("record_function", 23, ""),
    ("--write-plan", 0, "LINEAR_PROFILE_PLAN_UNVERIFIED"),
    ("--summarize", 0, "LINEAR_PROFILE_SUMMARY_UNVERIFIED"),
])
def test_worker_propagates_native_failure_but_not_reporting_failure(
        tmp_path, monkeypatch, failure, returncode, unverified):
    """Execute only disposable shell stubs, never the real worker or credentials."""
    monkeypatch.setattr(job, "CLOUD_ROOT", tmp_path / "cloud_fixture")
    manifest = job.describe_smoke(tmp_path / "not_staged")
    worker = manifest["worker_command"].replace(
        f"export TMPDIR=/data/ycfeng/tmp/step5_linear_worker/{job.RUN_ID}",
        f"export TMPDIR={shlex.quote(str(tmp_path / 'worker_tmp'))}")
    scripts = tmp_path / "tests/integration/step5_profiling"
    scripts.mkdir(parents=True)
    (scripts / "run_on_worker.sh").write_text(
        'echo "NATIVE $*" >> "$CALLS"\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --profile_method ]; then\n'
        '    [ "$2" != "$FAILURE" ] || exit 23\n'
        '  fi\n'
        '  shift\n'
        'done\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, content in {
        "python3": '#!/bin/bash\necho "PYTHON $*" >> "$CALLS"\n'
                   'if [ -n "$FAILURE" ] && [ "${3-}" = "$FAILURE" ]; then exit 17; fi\nexit 0\n',
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
    }.items():
        path = binaries / name
        path.write_text(content)
        path.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
           "CALLS": str(calls), "FAILURE": failure}
    result = subprocess.run(["bash", "-c", worker], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode == returncode, result.stderr
    recorded = calls.read_text()
    if failure == "cuda_event":
        assert "--profile_method record_function" not in recorded
    if returncode:
        assert "--summarize" not in recorded
    else:
        assert recorded.count("NATIVE ") == 2 and "--summarize" in recorded
    if unverified:
        assert unverified in result.stderr
    if not failure:
        again = subprocess.run(["bash", "-c", worker], cwd=tmp_path, env=env,
                               text=True, capture_output=True)
        assert again.returncode != 0
        assert calls.read_text() == recorded


@pytest.fixture
def csv_output(tmp_path):
    plan = job.planned_profile(tmp_path)
    (tmp_path / "planned_metadata.json").write_text(json.dumps(plan))
    operators = sorted({name for row in plan["planned_csv_rows"] for name in row["operators"]})
    for method, path in job.csv_paths(tmp_path).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "num_tokens", "num_tensor_parallel_workers", "measurement_type",
                "typed_operator_contracts", "linear_backends",
                *(f"time_stats.{name}.{stat}" for name in operators for stat in ("mean", "median", "count")),
            ])
            writer.writeheader()
            for row in plan["planned_csv_rows"]:
                writer.writerow({
                    "num_tokens": row["num_tokens"],
                    "num_tensor_parallel_workers": row["num_tensor_parallel_workers"],
                    "measurement_type": "CUDA_EVENT" if method == "cuda_event" else "KERNEL_ONLY",
                    "typed_operator_contracts": json.dumps(row["typed_operator_contracts"]),
                    "linear_backends": json.dumps({name: "cpu_fixture_not_measurement"
                                                   for name in row["operators"]}),
                    **{f"time_stats.{name}.{stat}": "20" if stat == "count" else ".01"
                       for name in row["operators"] for stat in ("mean", "median", "count")},
                })
    return tmp_path


def test_summary_reports_rows_operators_contracts_and_backend_without_admission(csv_output):
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and result["complete"] and result["row_counts_match_plan"]
    assert not result["admitted"] and not result["errors"] and not result["coverage_errors"]
    for method, item in result["csv"].items():
        assert item["num_rows"] == 6 and item["rows_by_tp"] == {"1": 3, "8": 3}
        for row in item["rows"]:
            assert set(row["operators"]) == set(row["typed_operator_contracts"])
            assert set(row["linear_backends"].values()) == {"cpu_fixture_not_measurement"}
            assert row["measurement_type"] == ("CUDA_EVENT" if method == "cuda_event" else "KERNEL_ONLY")


def test_summary_import_and_read_need_only_python_stdlib(csv_output):
    script = """
import builtins, sys
original = builtins.__import__
def stdlib_only(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'vllm', 'pandas', 'numpy'}:
        raise AssertionError('Unexpected runtime dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = stdlib_only
from frontier.profiling.linear_op.step5_job import summarize_profiles
assert summarize_profiles(sys.argv[1])['complete']
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(csv_output)], check=True,
                   cwd=job.REPO_ROOT, env={**os.environ, "PYTHONPATH": str(job.REPO_ROOT)})


def test_missing_csv_is_reported_without_raising(tmp_path):
    (tmp_path / "planned_metadata.json").write_text(json.dumps(job.planned_profile(tmp_path)))
    result = job.summarize_profiles(tmp_path)
    assert not result["read_complete"] and not result["complete"] and not result["row_counts_match_plan"]
    assert not result["csv"] and len(result["errors"]) == 2


def test_row_count_difference_is_reported_not_used_as_an_extra_gate(csv_output):
    path = job.csv_paths(csv_output)["record_function"]
    rewrite_csv(path, lambda rows: rows.pop())
    result = job.summarize_profiles(csv_output)
    assert not result["complete"] and not result["row_counts_match_plan"]
    assert result["read_complete"] and result["coverage_errors"]
    assert not result["admitted"] and not result["errors"]
    assert result["csv"]["record_function"]["num_rows"] == 5


@pytest.mark.parametrize("missing", ["timing", "zero_timing", "contract"])
def test_same_row_count_missing_merge_is_not_complete(csv_output, missing):
    path = job.csv_paths(csv_output)["record_function"]

    def remove_merge(rows):
        row = next(row for row in rows if row["num_tokens"] == "1" and row["num_tensor_parallel_workers"] == "1")
        if missing == "contract":
            contracts = json.loads(row["typed_operator_contracts"])
            contracts.pop("step5_ffn_merge")
            row["typed_operator_contracts"] = json.dumps(contracts)
        else:
            row["time_stats.step5_ffn_merge.mean"] = "0" if missing == "zero_timing" else ""
            row["time_stats.step5_ffn_merge.count"] = "20" if missing == "zero_timing" else ""

    rewrite_csv(path, remove_merge)
    result = job.summarize_profiles(csv_output)
    assert not result["complete"]
    assert result["read_complete"] and result["row_counts_match_plan"]
    assert any(error.get("operator") == "step5_ffn_merge" for error in result["coverage_errors"])
    assert not result["admitted"]


@pytest.mark.parametrize("scope", ["planned", "actual"])
def test_duplicate_row_keys_fail_even_with_matching_row_counts(csv_output, scope):
    if scope == "planned":
        path = csv_output / "planned_metadata.json"
        plan = json.loads(path.read_text())
        plan["planned_csv_rows"][1] = dict(plan["planned_csv_rows"][0])
        path.write_text(json.dumps(plan))
    else:
        def duplicate_row(rows):
            rows[1] = dict(rows[0])

        rewrite_csv(job.csv_paths(csv_output)["cuda_event"], duplicate_row)
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and result["row_counts_match_plan"] and not result["complete"]
    assert any(error["issue"] == f"duplicate {scope} row key" for error in result["coverage_errors"])


@pytest.mark.parametrize("field,value", [("num_tokens", "2"), ("num_tensor_parallel_workers", "2"),
                                         ("measurement_type", "KERNEL_ONLY")])
def test_actual_row_key_must_match_tokens_tp_and_measurement(csv_output, field, value):
    update_csv_value(job.csv_paths(csv_output)["cuda_event"], field, value)
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and result["row_counts_match_plan"] and not result["complete"]
    assert {error["issue"] for error in result["coverage_errors"]} >= {"missing row", "unexpected row"}


@pytest.mark.parametrize("stat,value", [("mean", "nan"), ("mean", "inf"), ("mean", "-1"),
                                       ("count", "0"), ("count", "nan"), ("count", "1.5")])
def test_expected_timing_needs_positive_finite_mean_and_integral_count(csv_output, stat, value):
    update_csv_value(job.csv_paths(csv_output)["cuda_event"], f"time_stats.step5_ffn_merge.{stat}", value)
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and not result["complete"]
    assert "step5_ffn_merge" not in result["csv"]["cuda_event"]["rows"][0]["operators"]
    assert any(error.get("operator") == "step5_ffn_merge" for error in result["coverage_errors"])


@pytest.mark.parametrize("field,value", [("selected_tensor_parallel_size", 8),
                                        ("selected_tensor_parallel_size", True), ("profile_id", "different")])
def test_typed_contract_must_match_plan_exactly_per_operator(csv_output, field, value):
    def change_contract(rows):
        contracts = json.loads(rows[0]["typed_operator_contracts"])
        contracts["step5_ffn_merge"][field] = value
        rows[0]["typed_operator_contracts"] = json.dumps(contracts)

    rewrite_csv(job.csv_paths(csv_output)["cuda_event"], change_contract)
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and result["row_counts_match_plan"] and not result["complete"]
    assert any(error.get("operator") == "step5_ffn_merge" and error["issue"] == "typed contract differs from plan"
               for error in result["coverage_errors"])


def test_plan_contracts_support_dictionary_or_json_object_representations(csv_output):
    path = csv_output / "planned_metadata.json"
    plan = json.loads(path.read_text())
    for row in plan["planned_csv_rows"][::2]:
        row["typed_operator_contracts"] = json.dumps(row["typed_operator_contracts"], sort_keys=True)
    path.write_text(json.dumps(plan))
    assert job.summarize_profiles(csv_output)["complete"]


def test_missing_timing_column_is_coverage_failure_not_silently_dropped(csv_output):
    def remove_count(rows):
        for row in rows:
            row.pop("time_stats.step5_ffn_merge.count")

    rewrite_csv(job.csv_paths(csv_output)["cuda_event"], remove_count)
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and not result["complete"]
    assert any(error.get("operator") == "step5_ffn_merge" and "count" not in error["timing"]
               for error in result["coverage_errors"] if "timing" in error)


@pytest.mark.parametrize("missing", [True, False])
def test_required_typed_contract_column_is_not_defaulted_to_empty(csv_output, missing):
    def break_contracts(rows):
        for row in rows:
            if missing:
                row.pop("typed_operator_contracts")
            else:
                row["typed_operator_contracts"] = ""

    rewrite_csv(job.csv_paths(csv_output)["cuda_event"], break_contracts)
    result = job.summarize_profiles(csv_output)
    assert not result["read_complete"] and not result["complete"] and result["errors"]


@pytest.mark.parametrize("missing", [True, False])
def test_complete_requires_exactly_both_planned_methods(csv_output, missing):
    path = csv_output / "planned_metadata.json"
    plan = json.loads(path.read_text())
    if missing:
        plan["csv_paths"].pop("record_function")
    else:
        plan["csv_paths"]["unsupported_method"] = plan["csv_paths"]["cuda_event"]
    path.write_text(json.dumps(plan))
    result = job.summarize_profiles(csv_output)
    assert result["read_complete"] and not result["complete"]
    assert any(error["issue"] == "planned methods differ" for error in result["coverage_errors"])


def test_inactive_zero_columns_are_not_measured_operators(csv_output):
    def add_inactive_columns(rows):
        for row in rows:
            row.update({"time_stats.inactive_op.mean": "0", "time_stats.inactive_op.count": "20"})

    rewrite_csv(job.csv_paths(csv_output)["cuda_event"], add_inactive_columns)
    result = job.summarize_profiles(csv_output)
    assert result["complete"]
    assert "inactive_op" not in result["csv"]["cuda_event"]["operators"]


def test_incomplete_coverage_keeps_summary_cli_nonblocking(csv_output, capsys):
    rewrite_csv(job.csv_paths(csv_output)["cuda_event"], lambda rows: rows.pop())
    assert job.main(["--summarize", str(csv_output)]) is None
    result = json.loads((csv_output / "summary.json").read_text())
    assert result["read_complete"] and not result["complete"] and not result["admitted"]
    assert "LINEAR_PROFILE_SUMMARY " in capsys.readouterr().out


def test_plan_and_summary_never_overwrite_existing_files(tmp_path, csv_output, capsys):
    path = tmp_path / "other_plan.json"
    job.main(["--write-plan", str(path)])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        job.main(["--write-plan", str(path)])
    assert path.read_bytes() == original
    job.main(["--summarize", str(csv_output)])
    assert "LINEAR_PROFILE_SUMMARY " in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        job.main(["--summarize", str(csv_output)])
