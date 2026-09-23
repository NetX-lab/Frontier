"""Submit one DP-placement ground-truth run (plan section 18.5 G3/G4)."""

import os
import shlex
from pathlib import Path

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import spawn_tasks

MOUNT = "/data/ycfeng/Frontier"
TREE = f"{MOUNT}/.worktrees/issue26-correctness-pr"
CASE = f"{TREE}/task_memory/task_2026-09-21_issue26_correctness_pr/calibration/dp_pp_case_001"
IMAGE = "artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2"
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"

run_tag = os.environ["RUN_TAG"]
engine = os.environ["ENGINE_INPUT"]
trace = os.environ["TRACE_INPUT"]
gpus = int(os.environ["NUM_GPUS"])

assert os.environ["STEPMIND_BACKEND"] == "rjob"
assert os.environ["BRAINPP_ACCESS_KEY"] and os.environ["BRAINPP_SECRET_KEY"]
os.chdir(MOUNT)
for path in (f"{CASE}/inputs/{engine}", f"{CASE}/inputs/{trace}/trace.csv",
             f"{CASE}/runs/groundtruth_clean/{run_tag}/run_manifest.json",
             f"{TREE}/tests/comparison/dp_placement_pp/run_vllm_worker.sh",
             f"{MOUNT}/.real-engine/vLLM-BS/vllm/v1/frontier_trace.py"):
    assert Path(path).is_file(), path
os.environ["EXP_ID"] = run_tag

payload = " ".join([
    f"RUN_TAG={run_tag}",
    f"FRONTIER_TREE={TREE}",
    f"GROUNDTRUTH={MOUNT}/.real-engine/vLLM-BS",
    f"CASE_DIR={CASE}",
    f"ENGINE_CONFIG={CASE}/inputs/{engine}",
    f"TRACE_DIR={CASE}/inputs/{trace}",
    f"ARCHIVE_DIR=/mnt/codesign-exp/ycfeng/frontier/dp_pp_calibration/{run_tag}",
    f"bash {TREE}/tests/comparison/dp_placement_pp/run_vllm_worker.sh",
])
command = "bash -c " + shlex.quote(payload)
print("WORKER_COMMAND", command, flush=True)

cfg = ResourceConfig(
    cpu=int(os.environ["NUM_CPUS"]),
    gpu=gpus,
    mem_gb=int(os.environ["MEM_GB"]),
    replica=1,
    image=IMAGE,
    positive_tags=["H800"],
    extra_requirements=[],
    mounts=[CLOUD_MOUNT],
    custom_resources=[],
    envs={},
    command="{COMMAND}",
    task_specs={"default": {"is_critical": True}},
)
workers = spawn_tasks(
    cfg,
    command=command,
    # User instruction: GPU workers use codesign only (steptron_ci paused).
    charged_group="codesign",
    use_image=True,
    code_mount_point=MOUNT,
)
print("RJOB_NAME", workers.rjob.meta.name, flush=True)
print("FINAL_STATUS", workers.poll(), flush=True)
