#!/bin/bash
# Usage: run_units.sh <log path>
cd /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
FILES=$(ls tests/unit | grep -E "dsv4|nixl|kernel_gap|pd_transfer|kv_transfer|glm53_structural|comm_operator|operator_registry|operator_spec|transfer|dp_dummy" | sed "s#^#tests/unit/#")
ulimit -v 31457280
export PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
timeout 5400s /data/ycfeng/envs/frontier-py310/bin/python -m pytest $FILES -q -p no:cacheprovider --no-header --continue-on-collection-errors -rfE > "$1" 2>&1
echo "exit=$?" >> "$1"
