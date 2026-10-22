"""Offline S0 job, row-count and platform-resource contract checks."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/integration/step5_comm"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s0_preparation_and_submission_dry_run(tmp_path, monkeypatch):
    definition = json.loads((PACKAGE / "s0_run.json").read_text())
    prepare = load("prepare")
    model = tmp_path / prepare.MODEL_CONFIG
    model.parent.mkdir(parents=True)
    model.write_bytes((ROOT / prepare.MODEL_CONFIG).read_bytes())
    def git_output(command, **kwargs):
        return {"ls-files": prepare.MODEL_CONFIG + "\n", "status": "",
                "rev-parse": "test_commit\n"}[command[1]]
    monkeypatch.setattr(prepare.subprocess, "check_output", git_output)
    job = prepare.prepare(definition, tmp_path / "stage", "s0_test", repository=tmp_path)
    assert job["expected"] == {"scenarios": 10, "cases": 100, "measurements": 100,
                               "kernel_rows": 100, "wall_minutes": 20,
                               "gpu_hours": 8 / 3}
    submit = load("submit")
    resources = submit.resource_arguments(job)
    assert resources["gpu"] == 8 and resources["replica"] == 1
    assert resources["positive_tags"] == ["H800"]
    assert resources["preemptible"] is False
    assert resources["extra_requirements"] == []
    assert "custom_resources" not in resources  # Preserve the SDK's RDMA defaults.
    assert resources["task_anti_affinities"] == ["hard/default"]
    assert resources["envs"]["NCCL_DEBUG_SUBSYS"] == "INIT,NET,GRAPH"
    assert job["image"].endswith("69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
    assert job["output_dir"] == "/mnt/codesign-exp/ycfeng/frontier/step5_comm/s0_test"
    config = tmp_path / "stage/job.json"
    command = submit.job_command(job, config)
    assert "timeout 1800 bash" in command
    assert load("worker").collective_command(job)[-2:] == ["--config", job["benchmark"]]
    native = load("worker").probe_command(job)
    assert "frontier.profiling.collectives.step5_dispatch_probe" in native
    assert native[-3:] == ["--tokens-per-dp-rank", "8", "1"]
    assert (Path(job["native_probe"]["model_path"]) / "config.json").read_bytes() == model.read_bytes()
    result = subprocess.run([sys.executable, str(PACKAGE / "submit.py"),
                             "--config", str(config), "--dry-run"],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["expected"]["measurements"] == 100
    assert not config.with_name("submission_attempt.json").exists()


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "/absolute"])
def test_run_id_stays_inside_cloud_root(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        load("prepare").prepare(json.loads((PACKAGE / "s0_run.json").read_text()),
                                tmp_path, run_id)


def test_worker_does_not_install_or_replace_native_dependencies():
    text = (PACKAGE / "run_on_worker.sh").read_text()
    assert "pip install" not in text
    assert "python3 tests/integration/step5_comm/worker.py" in text
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1" in text
    assert "NCCL_IB_HCA=" not in text
    worker = (PACKAGE / "worker.py").read_text()
    assert worker.index("STEP5_COMM_WORKER_LOG") < worker.index("os.dup2")
    assert '"torch_nccl_version": torch.cuda.nccl.version()' in worker
    subprocess.run(["bash", "-n", str(PACKAGE / "run_on_worker.sh")], check=True)


def test_preparation_rejects_uncommitted_source(tmp_path, monkeypatch):
    prepare = load("prepare")
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda *args, **kwargs: " M worker.py\n")
    with pytest.raises(ValueError, match="Commit the source"):
        prepare.prepare(json.loads((PACKAGE / "s0_run.json").read_text()),
                        tmp_path / "stage", "s0_test", repository=tmp_path)
    assert not (tmp_path / "stage").exists()


def test_selected_prepare_propagates_frozen_config_and_separate_local_counts(tmp_path,monkeypatch):
    prepare=load("prepare")
    definition=json.loads((PACKAGE/"s2_run.json").read_text())
    model=tmp_path/prepare.MODEL_CONFIG;model.parent.mkdir(parents=True)
    model.write_bytes((ROOT/prepare.MODEL_CONFIG).read_bytes())
    frozen=tmp_path/"network.json";frozen.write_text('{"evidence_kind":"cpu_fixture"}')
    definition["frozen_model_config"]="network.json"
    names=["moe_dispatch_all_gatherv_fp8","moe_combine_reduce_scatterv"]
    definition["grid"]["native_call_names"]=names
    definition["grid"]["local_work"]=[{"evidence_kind":"cpu_fixture"}]
    monkeypatch.setattr(prepare.subprocess,"check_output",lambda command,**kwargs:
        {"ls-files":prepare.MODEL_CONFIG+"\n","status":"","rev-parse":"cpu_fixture\n"}[command[1]])
    job=prepare.prepare(definition,tmp_path/"stage","selected",repository=tmp_path)
    benchmark=json.loads(Path(job["benchmark"]).read_text())
    assert job["expected"]["cases"] == 2*job["expected"]["scenarios"]
    assert job["expected"]["local_work_rows"] == definition["passes"]
    assert benchmark["grid"]["native_call_names"] == names
    assert benchmark["frozen_model_config_sha256"] == prepare.sha256(frozen)


@pytest.mark.parametrize("stage,scenarios,cases,nodes,dp,minutes", [
    ("s1", 50, 500, 4, 4, 25), ("s2", 46, 460, 2, 2, 12)])
def test_multinode_preparation_freezes_exact_rows_and_launchers(tmp_path, monkeypatch, stage, scenarios, cases, nodes, dp, minutes):
    prepare = load("prepare")
    definition = json.loads((PACKAGE / f"{stage}_run.json").read_text())
    model = tmp_path / prepare.MODEL_CONFIG
    model.parent.mkdir(parents=True)
    model.write_bytes((ROOT / prepare.MODEL_CONFIG).read_bytes())
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda command, **kwargs:
                        {"ls-files": prepare.MODEL_CONFIG + "\n", "status": "",
                         "rev-parse": "frozen\n"}[command[1]])
    job = prepare.prepare(definition, tmp_path / "stage", stage + "_offline", repository=tmp_path)
    assert job["expected"]["scenarios"] == scenarios
    assert job["expected"]["measurements"] == cases * 3
    assert job["expected"]["kernel_rows"] == cases  # Profile each callsite once, not each pass.
    assert job["expected"]["gpu_hours"] == pytest.approx(8 * nodes * minutes / 60)
    assert job["resources"]["gpu"] == job["gpus_per_node"] == 8
    assert job["resources"]["replica"] == nodes
    assert job["engine"]["tensor_parallel_size"] == 8
    assert job["engine"]["data_parallel_size"] == dp
    assert len(job["native_probe"]["tokens_per_dp_rank"]) == dp
    assert job["rendezvous"]["master_port"] != job["rendezvous"]["probe_port"]
    script = Path(job["submission_script"])
    assert prepare.sha256(script) == job["submission_script_sha256"]
    text = script.read_text()
    assert text.index("source ") < text.index("cd ") < text.index('exec "$STEPMIND_PYTHON"')
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert not script.with_name("submission_attempt.json").exists()


def test_generated_launcher_loads_credentials_before_the_blocking_submitter(tmp_path, monkeypatch):
    prepare = load("prepare")
    definition = json.loads((PACKAGE / "s0_run.json").read_text())
    auth = tmp_path / "auth environment.sh"
    auth.write_text(f'export STEPMIND_PYTHON="{sys.executable}"\nexport AUTH_LOADED_FIRST=yes\ncd /\n')
    definition["credential_environment"] = str(auth)
    model = tmp_path / prepare.MODEL_CONFIG
    model.parent.mkdir(parents=True)
    model.write_bytes((ROOT / prepare.MODEL_CONFIG).read_bytes())
    fake_submit = tmp_path / "tests/integration/step5_comm/submit.py"
    fake_submit.parent.mkdir(parents=True)
    fake_submit.write_text('import os,sys\nassert os.environ["AUTH_LOADED_FIRST"] == "yes"\n'
                           f'assert os.getcwd() == {str(tmp_path)!r}\n'
                           'assert sys.argv[1] == "--config"\nprint("CREDENTIALS_FIRST_PASS")\n')
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda command, **kwargs:
                        {"ls-files": prepare.MODEL_CONFIG + "\n", "status": "",
                         "rev-parse": "frozen\n"}[command[1]])
    job = prepare.prepare(definition, tmp_path / "stage", "offline", repository=tmp_path)
    result = subprocess.run(["bash", job["submission_script"]], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "CREDENTIALS_FIRST_PASS"
