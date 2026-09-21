## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded exact AR bypass, startup failures, runtime asset repair and pending fresh run. |

# Minimal post-MoE AR bypass validation

## Test script information

- Standard suite: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h800_replay_worker.sh` -> `issue26_h800_uniform_groundtruth_worker.sh` -> `issue26_h800_official_ttft_worker.sh`, followed by `issue26_h800_diagnostics_worker.sh batch` -> `issue26_token_id_client.py`.
- Reproducible launch: `bash /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-post-moe-bypass-03/launch.sh` (fresh output paths and job name required for a new run).
- Source: `/data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910`, commit `e29a8f925216d517bcbf7ffad47b93970d72d1f5`, parent `0d633a946e6600c77a251bef8b5553ec7f43f7e7`.
- GPU environment: fixed image digest b97a2fdc..., conda `vllm-bs-0.10.2`, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0. CPU asset verification: Python 3.12.3.
- Allocation: codesign + h800, 8 GPUs, 64 CPUs, 409600 MiB, KV blocks 176000. H200 results remain separate.

## Validation criteria

- Exactly one post-MoE TP AR invocation commented; DP combine preserved; no added clone/scalar/event/file I/O.
- Standard 3 x 100 fully drained warmups + 100 formal requests per server mode; 400 complete client rows each.
- Uniform routing 24 direct checks; standard eight-worker identity validator PASS.
- Select formal request cmpl-pf4096_dc1024:0-0, DP0 TP0-3, batch size 1, request_num_tokens [4096], prefill 4096, decode 0.
- Report median, rank P90, rank max, rank spread, and delta versus qualified normal reference. Rank P90 over four ranks is distinct from repeated-run P90.
- Observed normal reference median 77.483665466 ms, max 77.486846924 ms, spread 0.026206970 ms; prior normal median 80.654880524 ms. These two runs establish approximately 80 ms scale, not a statistical variability interval.
- Runtime success requires actual worker exit 0, completion markers, full clients and valid batch logs. Platform Succeeded alone is insufficient.

## Results and evidence

- Exact single-call source diff and Python AST: PASS.
- Two worker shell syntax checks and git diff whitespace: PASS. Worker source selection preserves the default normal source, and uses isolated source with pinned commit for this experiment.
- Runtime asset repair: all eight missing ignored FlashAttention files copied from verified normal source and compared byte-for-byte, PASS. See runtime_assets.json. This restores the same dependency content; no new library version.
- Job -01: STOPPED before requests. Local timeout interrupted rlaunch while it awaited worker startup; rlaunch sent stop. Rerun launch uses no short process lifetime limit.
- Job -02: FAIL before requests. Platform Succeeded, worker exit 1. Model registry import failed with `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'`. Git clone had preserved only the tracked .gitkeep; eight runtime files were ignored and missing. GPU probes and uniform router check passed, server never became ready. Zero measured client/batch results.
- Job -03: COMPLETED on gpu-h800-0600, actual worker exit 0 at 2026-09-10 09:34:52 UTC. Clean and batch each 400 rows (300 warmup + 100 formal), three drained 100-request warmups, all 4096 prompt/1024 output. Standard eight-worker identity validator PASS; 24 direct router checks PASS. Actual server args equal normal reference; selected source and boundary/operator logging OFF verified.

## Causal limits

Old 522.214 ms and 675.207 ms medians were normal AR diagnostic runs with altered workload, boundary logging and unqualified uniform routing. They do not prove AR bypass caused 5x slowdown. Complete attribution of their excess is open. A successful single bypass result gives an ablation delta; same-cluster repeated paired normal/skip runs and completion evidence remain necessary for a stable collective/queue causal claim. This timing-only experiment cannot close production CUDA or E2E parity gates.

Direct standard-result analysis command after completion: `python tests/e2e/issue26_standard_bypass_result.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-post-moe-bypass-03 --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-post-moe-bypass-03/standard_result.json`. Run the existing `issue26_diagnostic_identity_analysis.py --mode batch` independently for all eight workers.

## Final measured result — 2026-09-10

Formal identity `cmpl-pf4096_dc1024:0-0`, DP0 batch4718, TP0-3; each batch_size1, request_num_tokens[4096], prefill4096,decode0, DP token counts[4096,1]. Normal reference was DP0 batch4711 on gpu-h800-0496; bypass on gpu-h800-0600. Both use H800/codesign/176000.

| Metric | Normal ms | Minimal bypass ms | Bypass minus normal ms |
| --- | ---: | ---: | ---: |
| Rank median | 77.483665466 | 77.935920715 | +0.452255249 |
| Rank P90 | 77.486233521 | 78.081732941 | +0.595499420 |
| Rank max | 77.486846924 | 78.142173767 | +0.655326843 |
| Rank spread | 0.026206970 | 0.222427368 | +0.196220398 |

Rank spans TP0/1/2/3: 77.940704346 / 77.919746399 / 77.931137085 / 78.142173767 ms. Median delta +0.583678%; max delta +0.845727%. Normal-scale gate PASS; no 5x slowdown in minimal bypass. This is not an AR operator-accuracy gate or a repeated-run causal conclusion.

Analysis failure/fix: initial direct analyzer selected target request membership plus whole-batch prefill>0, yielding47rows on TP0 because target decode overlapped later requests' prefill. Fixed selection to the target request's own scheduled tokens4096. Actual artifact validation rerun PASS, retaining all single-request/DP0/rank predicates. Standard identity validator independently PASS.

Evidence: `analysis/h800-standard-post-moe-bypass-03/standard_result.json`, `identity_validation.json`, `execution_status.json`, `runtime_assets.json`, and raw clean/batch logs. Launch log `/data/ycfeng/tmp/issue26-h800-bypass-03-launch.log` includes both completion markers. Shutdown TCPStore warnings occur after requests complete; worker exit0 verified.
