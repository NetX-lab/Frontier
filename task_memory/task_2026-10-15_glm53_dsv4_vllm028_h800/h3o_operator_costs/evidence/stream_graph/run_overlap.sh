#!/bin/bash
# Kernel-overlap diagnostics of S1 over the retained native decode traces (TP rank 0 of DP lane 0).
cd /data/ycfeng/tmp/glm53_dsv4/h3o_b/s1
P=/data/ycfeng/envs/frontier-py310/bin/python
ulimit -v 31457280
declare -A TRACE=(
  [p2_32]=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942701901162313.pt.trace.json.gz
  [p2_64]=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792946239761662380.pt.trace.json.gz
  [h800_p0]=/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842748090329211.pt.trace.json.gz
)
run() {  # <output name> <case> <patterns>...
  local out=$1 case=$2; shift 2
  { echo "trace: ${TRACE[$case]}"; printf 'patterns:'; printf " '%s'" "$@"; echo
    timeout 900 $P kernel_overlap.py "${TRACE[$case]}" "$@" 2>&1; echo "rc=$?"; } > overlap_${out}_${case}_decode_dp0.txt
}
rm -f shared_overlap_*_decode_dp0.txt overlap_*_decode_dp0.txt
for c in p2_32 p2_64 h800_p0; do
  run shared_expert $c 'swapAB<4096u, 4096u' 'swapAB<4096u, 2048u' 'act_and_mul_kernel'
  run hca_compressor $c 'SparseAttnCompressC128Block8' 'SparseAttnNormRopeStoreKernel'
  run csa_attn_compressor $c 'SparseAttnCompressNormRopeStoreC4'
done
echo done > run_overlap.done
