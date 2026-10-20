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
    assert resources["envs"]["NCCL_DEBUG_SUBSYS"] == "INIT,NET"
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
