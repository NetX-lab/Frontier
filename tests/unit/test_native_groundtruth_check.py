"""Existing external checker admission tests; CPU fixtures are not native evidence."""

import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest
import yaml

from tests.comparison.calibration.native_pd_worker import node_plan
from tests.comparison.calibration.pd_replay import canonical_workload, instance_env, kv_transfer_config
from tests.comparison.dp_placement_pp.vllm_replay import V028_IMAGE, V028_SOURCE, server_command


def original_checkout_root(repo):
    common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


REPO = Path(__file__).resolve().parents[2]
CHECKER = original_checkout_root(REPO) / "task_memory/task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate/calibration/check_groundtruth_run.py"
BEFORE = REPO / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_glm_canonical_integration_20261021/check_groundtruth_run.before.py"
TRACE = REPO / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"

if not CHECKER.is_file():
    pytest.skip(f"External groundtruth checker is unavailable: {CHECKER}", allow_module_level=True)


def legacy_snapshot():
    if not BEFORE.is_file():
        pytest.skip(f"External legacy checker baseline is unavailable: {BEFORE}")
    return BEFORE


def load_checker(path):
    spec = importlib.util.spec_from_file_location("external_groundtruth_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def archived_harness(tmp_path_factory):
    """Use actual pinned reader bytes in a fresh subprocess, with fixture Git receipts."""
    root = tmp_path_factory.mktemp("native_checker_harness")
    archive = subprocess.run(["git", "archive", "9f58dbe", "tests/comparison"],
                             cwd=REPO, capture_output=True)
    if archive.returncode:
        pytest.skip(f"Pinned CPU fixture archive 9f58dbe is unavailable: {archive.stderr.decode().strip()}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as stream:
        stream.extractall(root, filter="data")
    return root


@pytest.fixture
def native_case(tmp_path, monkeypatch, archived_harness):
    checker = load_checker(CHECKER)
    root = tmp_path / "canonical"
    run = root / "runs/groundtruth_clean/native"
    run.mkdir(parents=True)
    shutil.copytree(TRACE, root / "inputs/trace")
    model = tmp_path / "model"
    write(model / "config.json", {"architectures": ["GlmMoeDsaForCausalLM"], "vocab_size": 256})
    engine = dict(vllm_version="0.28.0", native_reference=dict(source_sha=V028_SOURCE,
        instrumentation_sha="1" * 40, image=V028_IMAGE), load_format="dummy", skip_tokenizer_init=True,
        tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=4,
        enable_expert_parallel=True, enforce_eager=False, enable_chunked_prefill=True,
        enable_prefix_caching=False, dtype="bfloat16", max_num_batched_tokens=8192,
        max_num_seqs=16, block_size=16, max_model_len=7168, gpu_memory_utilization=.9, seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        kv_transfer=dict(kv_buffer_device="cuda", side_channel_host="10.0.0.1",
                         prefill_side_channel_port=32000, decode_side_channel_port=33000))
    config = dict(mode="clean", run_tag="native", engine=engine, replica=8, gpu=8, data_parallel_size_local=1,
        gpu_type="H200", image=V028_IMAGE, charged_group="step_main",
        model_dir=str(model), model_hashes={"config.json": digest(model / "config.json")},
        expected_source_hashes={"vllm/__init__.py": "a" * 64},
        http_ports=dict(prefill=31000, decode=31100), dp_rpc_ports=dict(prefill=31200, decode=31300),
        serve_observers=dict(prefill=None, decode=None))
    write(root / "inputs/engine.json", engine)
    write(root / "inputs/native_config.json", config)
    case = dict(case_id="canonical", groundtruth_commit=V028_SOURCE, groundtruth_branch="fixture-native",
                groundtruth_checkout_path=str(tmp_path / "source"), run_generation="fixture-generation-not-native")
    (root / "manifest.yaml").write_text(yaml.safe_dump(case))
    harness = archived_harness
    modules = {str(p.relative_to(harness)): digest(p) for p in (harness / "tests/comparison").rglob("*.py")}
    manifest = dict(case_id="canonical", run_tag="native", run_generation="fixture-generation-not-native",
        mode="clean", written_before_launch_utc="2026-01-01T00:00:00Z", authorization="CPU fixture only",
        groundtruth_checkout_path=str(tmp_path / "source"), groundtruth_commit=V028_SOURCE,
        groundtruth_branch="fixture-native", groundtruth_tree_dirty=False,
        harness_worktree=str(harness), harness_commit="9f58dbe", harness_branch="fixture-harness",
        harness_tree_dirty_paths=[], harness_sha256=modules,
        engine_config="inputs/engine.json", execution_config="inputs/native_config.json", trace_dir="inputs/trace",
        inputs_sha256={str(p.relative_to(root)): digest(p) for p in (root / "inputs").rglob("*") if p.is_file()},
        resources=dict(gpus=64, gpu_type="H200", replica=8, gpu=8, charged_group="step_main", image=V028_IMAGE))
    manifest["inputs_sha256"][str(model / "config.json")] = digest(model / "config.json")
    manifest["inputs_sha256"]["manifest.yaml"] = digest(root / "manifest.yaml")
    write(run / "run_manifest.json", manifest)
    workload, formal, warmup = canonical_workload(TRACE)
    records, boundaries = [], []

    def boundary(name, stamp, completed):
        roles = {role: dict(lanes={str(i): dict(num_requests_running=0, num_requests_waiting=0,
                      kv_cache_usage_perc=0) for i in range(4)}, completed_requests=completed, idle=True)
                 for role in ("prefill", "decode")}
        boundaries.append(dict(boundary=name, verified=True, observations=[
            dict(monotonic=stamp, roles=roles, errors=[])]))

    boundary("before_repeat01_warmup", 999., 0)
    completed = 0
    for repeat in (1, 2, 3):
        for phase, templates in (("warmup", warmup), ("formal", formal)):
            for template in templates:
                arrival = repeat * 1000 + (100 if phase == "formal" else 0) + template["arrived_at"]
                records.append(dict(template_request_id=template["request_id"],
                    request_id=f"{template['request_id']}-repeat{repeat:02d}", repeat=repeat,
                    arrived_at=template["arrived_at"], role=phase, segment=phase,
                    status="completed", http_status=200, validation_errors=[], stream_done=True,
                    finish_reason="length", num_prefill_tokens=6144, num_decode_tokens=1024,
                    output_token_count=1024, usage=dict(prompt_tokens=6144, completion_tokens=1024, total_tokens=7168),
                    dispatch_monotonic=arrival, first_token_monotonic=arrival + 2,
                    last_token_monotonic=arrival + 4.046, ttft_ms=2000., tpot_ms=2., e2e_ms=4046.))
            completed += len(templates)
            boundary(f"after_repeat{repeat:02d}_{phase}", records[-1]["last_token_monotonic"] + 1, completed)
    repetitions = dict(client_workload_verified=True, observation_errors=[], reference_admission="UNVERIFIED",
                       repetitions=[dict(repeat=i) for i in (1, 2, 3)], idle_boundaries=boundaries)
    peers = [dict(host=f"host{i}", ip=f"10.0.0.{i + 1}") for i in range(8)]
    urls = dict(prefill="http://10.0.0.1:31000", decode="http://10.0.0.5:31100")
    summary = dict(mode="clean", vllm_version="0.28.0", status=0, canonical_repetitions=repetitions,
                   engine_config=engine, num_requests=len(records), failed_request_ids=[],
                   instances={role: dict(http_url=url) for role, url in urls.items()})
    for peer in peers:
        host = peer["host"]
        directory = run / "nodes" / host
        plan = node_plan(config, peers, host, [str(i) for i in range(8)])
        role = plan["role"]
        command = server_command(plan["engine"], model, config["http_ports"][role]) + [
            "--kv-transfer-config", json.dumps(kv_transfer_config(plan["engine"], role, config["http_ports"][role]))]
        _, env = instance_env("clean", plan["engine"], directory, {}, "0,1,2,3,4,5,6,7", role=role)
        order = dict(drained_before_stop=True, all_nodes_drained_before_head_stop=[p["host"] for p in peers],
                     head_stopped_before_headless_stop=True, native_returned_before_requested_stop=False)
        stop = dict(signal="SIGINT", returncode=0)
        node = dict(host=host, stage="finished", plan=plan, versions=dict(vllm="0.28.0"),
                    nvidia_smi="\n".join(f"GPU {i}: NVIDIA H200 (UUID: fixture-{host}-{i})" for i in range(8)),
                    native_source_hashes=config["expected_source_hashes"], model_hashes=config["model_hashes"],
                    server_command=command, server_env_set=env, server_stop=stop, shutdown_order=order,
                    observation_errors=[], weight_audit={}, workload_returncode=0)
        write(directory / "ready.json", peer)
        write(directory / "result.json", node)
        if plan["lane"] == 0:
            write(directory / "server_ready.json", dict(role=role, url=urls[role]))
        write(directory / "drained.json", dict(role=role, lane=plan["lane"]))
        write(directory / "native_stopped.json", stop | {"shutdown_order": order})
    write(run / "run/replay_summary.json", summary)
    write(run / "run/canonical_repetitions.json", repetitions)
    (run / "run/client_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write(run / "replay_finished.json", dict(returncode=0))
    receipt = dict(job="fixture-not-real", creator="i-fengyicheng", charged_group="step_main",
        status=dict(phase="Succeeded", finishTime="2026-01-01T01:00:00Z"), tasks=dict(default=dict(replicas=8, resources=[
            dict(limits={"nvidia.com/gpu": "8"})])), replicas=[dict(name=f"fixture-{i}", node=p["host"],
            phase="Succeeded", containers=[dict(state=dict(terminated=dict(exitCode=0)), restartCount=0,
                                                imageID=V028_IMAGE)]) for i, p in enumerate(peers)])
    write(run / "platform_terminal_receipt.json", receipt)
    write(run / "launcher_receipt.json", dict(rjob_name=receipt["job"],
        run_generation=manifest["run_generation"], run_manifest_sha256=digest(run / "run_manifest.json"),
        execution_config_sha256=digest(root / manifest["execution_config"]),
        recorded_at_utc="2026-01-01T00:00:01Z"))
    original_run = subprocess.run

    def fixture_git(repo, *args):
        if args[:1] == ("status",):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "9f58dbe" if repo == str(harness) else V028_SOURCE
        if args == ("branch", "--show-current"):
            return "fixture-harness" if repo == str(harness) else "fixture-native"
        raise AssertionError(args)

    def fixture_run(args, **kwargs):
        if args[:3] == ["git", "-C", str(harness)]:
            args = ["git", "-C", str(REPO), *args[3:]]
        return original_run(args, **kwargs)

    monkeypatch.setattr(checker, "git", fixture_git)
    monkeypatch.setattr(checker.subprocess, "run", fixture_run)
    return checker, root, run, manifest, records, summary


def test_native_fixture_uses_actual_reader_and_never_fabricates_graph_pass(native_case):
    checker, root, _, _, _, _ = native_case
    result = checker.check(root, "native", "clean")
    assert result["status"] == "FAIL"
    assert [k for k, v in result["checks"].items() if v.startswith("FAIL")] == ["rank_runtime_graph"]
    assert "UNVERIFIED" in result["checks"]["rank_runtime_graph"]
    assert "run_manifest" in result and result["derived_num_blocks"] == {}


@pytest.mark.parametrize("fault", ["control", "wrong_job", "wrong_generation", "wrong_manifest_hash",
                                  "wrong_execution_hash", "missing_launcher_receipt", "stale_launcher_receipt"])
def test_native_terminal_receipt_binds_actual_launcher(native_case, fault):
    checker, root, run, _, _, _ = native_case
    launcher_path = run / "launcher_receipt.json"
    launcher = json.loads(launcher_path.read_text())
    if fault == "wrong_job":
        platform_path = run / "platform_terminal_receipt.json"
        platform = json.loads(platform_path.read_text())
        platform["job"] = "other-fixture-job-not-real"
        write(platform_path, platform)
    if fault == "wrong_generation": launcher["run_generation"] = "earlier-fixture-generation"
    if fault == "wrong_manifest_hash": launcher["run_manifest_sha256"] = "0" * 64
    if fault == "wrong_execution_hash": launcher["execution_config_sha256"] = "0" * 64
    write(launcher_path, launcher)
    if fault == "missing_launcher_receipt": launcher_path.unlink()
    if fault == "stale_launcher_receipt": os.utime(launcher_path, ns=(0, 0))
    result = checker.check(root, "native", "clean")
    expected = ["rank_runtime_graph"] + ([] if fault == "control" else ["terminal_status"])
    assert [k for k, v in result["checks"].items() if v.startswith("FAIL")] == expected, result
    assert result["status"] == "FAIL"  # CPU fixtures cannot establish native graph admission.
    assert result["rjob_name"] == ("fixture-not-real" if fault == "control" else None)


@pytest.mark.parametrize("layout", ["main_checkout", "worktree"])
def test_checker_path_resolves_from_both_git_layouts(layout):
    root = original_checkout_root(REPO)
    selected = root if layout == "main_checkout" else REPO
    assert original_checkout_root(selected) == root
    assert CHECKER == root / "task_memory/task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate/calibration/check_groundtruth_run.py"


def test_missing_external_checker_skips_module(tmp_path, monkeypatch):
    original_run = subprocess.run

    def git_common_dir(args, **kwargs):
        if args[-3:] == ["rev-parse", "--path-format=absolute", "--git-common-dir"]:
            return subprocess.CompletedProcess(args, 0, stdout=str(tmp_path / ".git") + "\n")
        return original_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_common_dir)
    with pytest.raises(pytest.skip.Exception, match="External groundtruth checker is unavailable"):
        load_checker(Path(__file__))


def test_missing_legacy_snapshot_does_not_skip_native_checker(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "BEFORE", tmp_path / "missing-before.py")
    with pytest.raises(pytest.skip.Exception, match="External legacy checker baseline is unavailable"):
        legacy_snapshot()
    checker = load_checker(CHECKER)
    assert checker.native_rank_graph_evidence({}, {}, {})[0] is False


def test_missing_pinned_archive_skips_native_fixture(tmp_path_factory, monkeypatch):
    original_run = subprocess.run

    def unavailable_archive(args, **kwargs):
        if args[:3] == ["git", "archive", "9f58dbe"]:
            return subprocess.CompletedProcess(args, 128, stdout=b"", stderr=b"Fixture commit unavailable")
        return original_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", unavailable_archive)
    with pytest.raises(pytest.skip.Exception, match="Pinned CPU fixture archive 9f58dbe is unavailable"):
        archived_harness.__wrapped__(tmp_path_factory)


def test_capture_inventory_and_claimed_replay_boolean_cannot_prove_runtime_coverage():
    checker = load_checker(CHECKER)
    passed, evidence = checker.native_rank_graph_evidence({"fixture": dict(inventory=[dict(
        groups={"world": {"rank": 0}}, moe_backends={"layer": "FLASHINFER_CUTLASS"},
        effective_graph="FULL_DECODE_ONLY", serving_replay_verified=True)])}, {}, {})
    assert not passed and "UNVERIFIED" in evidence


def update_native_stop(run, host, *, returned, signal, returncode=0, head_stopped=True):
    """Keep result and stop-marker observations identical when testing ordering."""
    directory = run / "nodes" / host
    node = json.loads((directory / "result.json").read_text())
    node["shutdown_order"].update(native_returned_before_requested_stop=returned,
                                   head_stopped_before_headless_stop=head_stopped)
    node["server_stop"] = dict(signal=signal, returncode=returncode)
    write(directory / "result.json", node)
    write(directory / "native_stopped.json", node["server_stop"] | {"shutdown_order": node["shutdown_order"]})


@pytest.mark.parametrize("returned,signal", [(True, None), (False, None), (False, "SIGINT")],
                         ids=["natural_headless_exit", "headless_poll_stop_race", "requested_headless_stop"])
def test_native_lifecycle_accepts_ordered_headless_completion(native_case, returned, signal):
    checker, root, run, _, _, _ = native_case
    update_native_stop(run, "host1", returned=returned, signal=signal)
    result = checker.check(root, "native", "clean")
    assert result["checks"]["native_lifecycle"].startswith("PASS"), result["checks"]["native_lifecycle"]
    assert [k for k, v in result["checks"].items() if v.startswith("FAIL")] == ["rank_runtime_graph"]


@pytest.mark.parametrize("fault", ["missing_head_marker", "head_not_stopped", "native_nonzero",
                                  "SIGTERM", "SIGKILL", "returned_with_signal", "early_head_exit"])
def test_native_lifecycle_rejects_unordered_or_failed_completion(native_case, fault):
    checker, root, run, _, _, _ = native_case
    update_native_stop(run, "host1", returned=True, signal=None,
                       returncode=23 if fault == "native_nonzero" else 0,
                       head_stopped=fault != "head_not_stopped")
    if fault in ("SIGTERM", "SIGKILL", "returned_with_signal"):
        update_native_stop(run, "host1", returned=fault == "returned_with_signal",
                           signal="SIGINT" if fault == "returned_with_signal" else fault)
    if fault == "early_head_exit":
        update_native_stop(run, "host0", returned=True, signal=None)
    if fault == "missing_head_marker":
        (run / "nodes/host0/native_stopped.json").unlink()
    result = checker.check(root, "native", "clean")
    assert result["checks"]["native_lifecycle"].startswith("FAIL"), result["checks"]["native_lifecycle"]


@pytest.mark.parametrize("fault,failed_check", [
    ("missing_provenance", "checkout"), ("stale_commit", "checkout"),
    ("dirty_harness", "harness_pinned_and_client_rows"),
    ("missing_harness_hash", "harness_pinned_and_client_rows"),
    ("changed_input", "inputs_unchanged"), ("missing_record", "harness_pinned_and_client_rows"),
    ("wrong_output_count", "harness_pinned_and_client_rows"),
    ("missing_lane", "requests_and_idle_lanes"), ("nonzero_gauge", "requests_and_idle_lanes"),
    ("stale_completions", "requests_and_idle_lanes"), ("idle_before_output", "requests_and_idle_lanes"),
    ("missing_role", "requests_and_idle_lanes"), ("wrong_node_lane", "native_workers"),
    ("observation_error", "native_workers"), ("native_error", "native_workers"),
    ("source_hash_mismatch", "native_workers"),
    ("missing_order", "native_lifecycle"), ("early_native_return", "native_lifecycle"),
    ("head_before_drain", "native_lifecycle"), ("headless_before_head", "native_lifecycle"),
    ("missing_platform", "terminal_status"), ("stale_output", "requests_and_idle_lanes"),
])
def test_native_negative_admission(native_case, fault, failed_check):
    checker, root, run, manifest, records, summary = native_case
    if fault == "missing_provenance": manifest.pop("groundtruth_commit")
    if fault == "stale_commit": manifest["groundtruth_commit"] = "0" * 40
    if fault == "dirty_harness": manifest["harness_tree_dirty_paths"] = ["modified.py"]
    if fault == "missing_harness_hash": manifest["harness_sha256"].pop("tests/comparison/calibration/client_metrics.py")
    write(run / "run_manifest.json", manifest)
    # Manifest fixture mutations should not make unrelated output files stale.
    os.utime(run / "run_manifest.json", ns=(1, 1))
    if fault == "changed_input": (root / "inputs/trace/trace.csv").write_text("changed")
    if fault == "missing_record": records.pop()
    if fault == "wrong_output_count": records[-1]["output_token_count"] = 1023
    (run / "run/client_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    boundary = summary["canonical_repetitions"]["idle_boundaries"][1]
    if fault == "missing_lane": boundary["observations"][-1]["roles"]["decode"]["lanes"].pop("3")
    if fault == "nonzero_gauge": boundary["observations"][-1]["roles"]["prefill"]["lanes"]["2"]["num_requests_waiting"] = 1
    if fault == "missing_role": boundary["observations"][-1]["roles"].pop("prefill")
    if fault == "stale_completions": boundary["observations"][-1]["roles"]["decode"]["completed_requests"] = 0
    if fault == "idle_before_output": boundary["observations"][-1]["monotonic"] = 1001
    write(run / "run/replay_summary.json", summary)
    write(run / "run/canonical_repetitions.json", summary["canonical_repetitions"])
    node_path = run / "nodes/host1/result.json"
    node = json.loads(node_path.read_text())
    if fault == "wrong_node_lane": node["plan"]["lane"] = 0
    if fault == "observation_error": node["observation_errors"] = ["fixture observation failed"]
    if fault == "native_error": node["workload_returncode"] = 23
    if fault == "source_hash_mismatch": node["native_source_hashes"] = {}
    if fault == "missing_order": node.pop("shutdown_order")
    if fault == "early_native_return": node["shutdown_order"]["native_returned_before_requested_stop"] = True
    if fault == "headless_before_head": node["shutdown_order"]["head_stopped_before_headless_stop"] = False
    write(node_path, node)
    if fault == "head_before_drain":
        node_path = run / "nodes/host0/result.json"
        node = json.loads(node_path.read_text())
        node["shutdown_order"]["all_nodes_drained_before_head_stop"].pop()
        write(node_path, node)
    if fault == "missing_platform": (run / "platform_terminal_receipt.json").unlink()
    if fault == "stale_output": os.utime(run / "run/replay_summary.json", ns=(0, 0))
    result = checker.check(root, "native", "clean")
    assert result["status"] == "FAIL" and result["checks"][failed_check].startswith("FAIL"), result


def test_legacy_function_bodies_and_cli_remain_unchanged():
    before, after = ast.parse(legacy_snapshot().read_text()), ast.parse(CHECKER.read_text())
    old = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    body = new["check"].body
    body[:] = [n for n in body if not (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "workload_manifest" for t in n.targets)) and not (
        isinstance(n, ast.If) and any(isinstance(v, ast.Name) and v.id == "workload_manifest" for v in ast.walk(n)))]
    for name, node in old.items():
        assert ast.dump(node) == ast.dump(new[name]), name


def test_legacy_missing_artifacts_keep_original_exception(tmp_path):
    write(tmp_path / "manifest.yaml", {})
    write(tmp_path / "runs/groundtruth_clean/legacy/run_manifest.json", {})
    write(tmp_path / "inputs/trace/request_ids.json", {})
    write(tmp_path / "inputs/engine.json", {})
    errors = []
    for module in (load_checker(legacy_snapshot()), load_checker(CHECKER)):
        with pytest.raises(FileNotFoundError) as error:
            module.check(tmp_path, "legacy", "clean")
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
