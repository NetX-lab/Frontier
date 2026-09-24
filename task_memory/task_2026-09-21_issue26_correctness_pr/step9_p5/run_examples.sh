#!/bin/bash
# Step 9 P5: the 16 Step 8 architecture examples on one exported tree, metrics under OUT.
tree="$1"; out="$2"
cd "$tree" || exit 1
export PYTHONPATH="$tree" WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PATH="/data/ycfeng/envs/frontier-py310/bin:$PATH"
SCRIPTS="
examples/architecture/co-location/offline/dense_model_basic.sh
examples/architecture/co-location/offline/moe_model_basic.sh
examples/architecture/co-location/offline/thinking_mode_basic.sh
examples/architecture/co-location/offline/moe_spec_dec.sh
examples/architecture/co-location/offline/moe_prefix_caching.sh
examples/architecture/co-location/online/dense_model_basic_online.sh
examples/architecture/co-location/online/moe_model_basic_online.sh
examples/architecture/pdd/offline/dense_model_basic.sh
examples/architecture/pdd/offline/moe_model_basic.sh
examples/architecture/pdd/online/dense_model_basic_online.sh
examples/architecture/pdd/online/moe_model_basic_online.sh
examples/architecture/pd-af-disagg/offline/dense_model_basic.sh
examples/architecture/pd-af-disagg/offline/moe_model_basic.sh
examples/architecture/pd-af-disagg/offline/moe_model_ep.sh
examples/architecture/pd-af-disagg/offline/moe_cuda_graph.sh
examples/architecture/pd-af-disagg/online/moe_cuda_graph_online.sh
"
pass=0; fail=0
for s in $SCRIPTS; do
  name=$(echo "$s" | sed 's#examples/architecture/##; s#/#_#g; s#\.sh$##')
  mkdir -p "$out/$name"
  if METRICS_OUTPUT_DIR="$out/$name/metrics" bash "$s" > "$out/$name/run.log" 2>&1; then
    echo "PASS $s"; pass=$((pass + 1))
  else
    echo "FAIL $s"; fail=$((fail + 1))
  fi
done
echo "=== examples: $pass passed, $fail failed ==="
