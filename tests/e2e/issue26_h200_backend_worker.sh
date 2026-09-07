#!/usr/bin/env bash
# Verify the communication backend in the exact H200 calibration runtime.
set -euo pipefail
source "$(dirname "$0")/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"

BACKEND_ROOT="$REPO_ROOT/frontier/cc_backend/backends/collective-sim"
make -B -j8 -C "$BACKEND_ROOT/sim" > "$PROBE_ROOT/backend_build.log" 2>&1
ldd "$BACKEND_ROOT/sim/datacenter/htsim_ndp"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PROBE_ROOT
"$PY" - <<'PY'
import json
import math
import os
from pathlib import Path

from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.types import ClusterType

root = Path(os.environ["PROBE_ROOT"])
config = CollectiveSimCCBackendConfig(
    cluster_servers=1, cluster_gpus_per_server=8, parallel_tp=4, parallel_dp=2,
    runtime_num_replicas=1, runtime_num_pipeline_stages=1,
    runtime_attn_tensor_parallel_size=4, runtime_attn_dp=2,
    runtime_moe_tensor_parallel_size=1, runtime_moe_expert_parallel_size=8,
    intra_server_model="nvlink_analytic",
    runner_out_dir=str(Path(os.environ["TMPDIR"]) / root.name / "htsim"),
    runner_end_us=100000, runner_stop_on_finished=True,
)
backend = CollectiveSimCCBackend(config, ClusterType.MONOLITHIC, "h200", "h200", 8)
values = {
    "purpose": "structural runtime verification; defaults are not calibrated",
    "message_bytes": 16777216,
    "tp4_allreduce_ms": backend.predict_allreduce(16777216, 4, comm_domain="ATTN_TP"),
    "ep8_alltoall_ms": backend.predict_all_to_all(16777216, 8, comm_domain="MOE_EP"),
}
for key in ("tp4_allreduce_ms", "ep8_alltoall_ms"):
    assert math.isfinite(values[key]) and values[key] > 0, (key, values[key])
(root / "backend_smoke.json").write_text(json.dumps(values, indent=2) + "\n")
print(json.dumps(values))
print("H200_BACKEND_RUNTIME_PASS")
PY
