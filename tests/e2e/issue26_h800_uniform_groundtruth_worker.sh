#!/usr/bin/env bash
# Verify the existing uniform router before measuring clean server TTFT.
set -euo pipefail
UNIFORM_RUN_ROOT="${1:?Provide a fresh run parent directory.}"
export VLLM_MOE_UNIFORM_ROUTING=1
source /data/ycfeng/tmp/issue26_h800_environment_probe.sh "$UNIFORM_RUN_ROOT/uniform-preflight"
"$PY" - "$PROBE_ROOT/routing_check.json" <<'PY'
import json
import sys
import torch
from vllm import envs
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

assert envs.VLLM_MOE_UNIFORM_ROUTING
records = []
for device in range(torch.cuda.device_count()):
    for tokens in (1, 4096, 4097):
        hidden = torch.zeros((tokens, 2048), device=f"cuda:{device}", dtype=torch.bfloat16)
        logits = torch.zeros((tokens, 128), device=hidden.device, dtype=torch.float32)
        weights, ids, _ = fused_topk(hidden, logits, 8, True)
        counts = torch.bincount(ids.flatten().long(), minlength=128).cpu().tolist()
        quotient, remainder = divmod(tokens * 8, 128)
        expected = [quotient + int(expert < remainder) for expert in range(128)]
        assert counts == expected, (device, tokens, counts, expected)
        assert bool(torch.all(weights == 0.125)), (device, tokens)
        records.append({"device": device, "tokens": tokens, "counts": counts})
with open(sys.argv[1], "x") as stream:
    json.dump({"status": "PASS", "uniform_routing": True, "records": records,
               "scope": "Direct runtime router; not a distributed batch trace"}, stream, indent=2)
print("UNIFORM_ROUTING_RUNTIME_PASS", len(records))
PY
bash "$REPO_ROOT/tests/e2e/issue26_h800_official_ttft_worker.sh" "$UNIFORM_RUN_ROOT/runtime"
echo UNIFORM_GROUNDTRUTH_EXECUTION_COMPLETE
