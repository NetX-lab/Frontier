#!/usr/bin/env bash
# Run tests/unit and tests/integration of one tree with JUnit output.
set -u
tree="$1"; out="$2"
cd "$tree"
export PYTHONPATH="$tree" WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
for suite in unit integration; do
  timeout 1800 /data/ycfeng/envs/frontier-py310/bin/python -m pytest "tests/$suite" -q -p no:cacheprovider --continue-on-collection-errors \
    --basetemp="$out/tmp-$suite" --junitxml="$out/$suite.xml" > "$out/$suite.log" 2>&1
  echo "$suite exit $?" >> "$out/$suite.log"
done
