# W128 native stream probe source delivery

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Root completed final-source checks, sealing and successful single-H800 execution; retained-data acceptance is in progress. |
| 2026-10-24 | Archived262-line reviewed-P2 repairs and identical original-byte RED/repaired-source GREEN controls. |
| 2026-10-24 | Archived the scoped source/CPU delivery and root-owned remaining checks. |

## Deliverables

- Current source: `tests/comparison/calibration/native_profile_probe.py`,262/275gross lines, SHA256 `2830d15c79ccf8f3a6280bc388c93ac423ecacefe9d73e053412a405ae4fd7c3`.
- Original source snapshot: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/original_native_profile_probe.py`,249lines, SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7`.
- Records: `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/{requirements.md,plan.md,progress.md,test_report_2026-10-24_stream_probe.md,checkpoint.md,summary.md}`.
- Task ID: `W128-NATIVE-STREAM-PROBE`.

The probe invokes the actual imported collector stream APIs and native `TorchProfilerWrapper`/`ProfilerConfig`. It keeps one wrapper/underlying profiler across two cycles, targets default+auxiliary eager operations, and separately retains priming evidence. Two labeled operations are captured and automatically instantiated by native default `CUDAGraph.capture_end` during cycle1, then replayed in both cycles. Complete native traces, raw joins, graph/node IDs, stream snapshots, errors/logs and source/library/runtime/GPU identities are retained. Graph correspondence requires direct exact identities and otherwise stays UNKNOWN. No model/serving cost is claimed.

Both independently reviewed P2 acceptance defects are repaired: graph PASS now requires the actual successful capture and both replay-completion flags; eager status is stored before optional parsing, and priming/graph exceptions are separately isolated through the existing error recorder. No native/GPU mechanism, identity inference, CLI, configuration or fallback was added. Repair diff is `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/source_repair.diff` (21additions/8deletions versus original).

## Observed validation

Original AST/native-source config-keyword witness/help/fresh local imports passed without CUDA initialization. Original read-only numstat was249 additions/0 deletions. Collector SHA256 matches root's `17d6dea11d35a2dc9849a11986fe5df46b45adb84d2d2bdeb958642a851b26ef`; root accepted its CPU source at `0d745706`/records `6ef05dd8`.

Before production repair, retained temporary real-function controls on original bytes were RED2PASS/6FAIL, reproducing both P2s while preserving graph/eager positives. The identical controls on repaired bytes were GREEN8PASS/0FAIL. Missing capture/replay completion is UNKNOWN; malformed graph schemas preserve all valid eager rows/cycle verdicts as PASS and retain graph UNKNOWN plus exact parser exceptions. Final AST and help pass;262gross lines are within275. Temporary control code, raw RED/GREEN receipts and all synthetic traces are retained under `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/`. These controls prove CPU verdict behavior only.

Local native-checkout import failed because the available Transformers4.57.6 is below the checkout's Transformers5 requirement. Installed vLLM0.16.0 config construction rejects native0.28 warmup/wait fields. Exact commands/errors remain in the test report; no dependency change or fallback was applied. Local Torch2.9.1+cu128 is not the locked image's Torch2.13.0+cu130.

## Source-owner handoff — historical

Root original-image receipt `/data/ycfeng/tmp/glm53_dsv4/native_stream_probe_image_oj706s8g/receipt.json` already passed actual config/wrapper construction, help and the original full CLI entry reaching expected GPU absence/exit2 on pinned2286e853. The owner only read that receipt and did not rerun or rewrite it. Root owns re-review and final repaired source/package binding, launch receipts,1H800/16CPU/64GB/600s submission and retained-data acceptance. Package the unchanged siblings listed in the test report and preserve existing driver/CUPTI configuration.

No GPU mapping/capture correspondence was measured; both remain unverified. No GPU/platform/network/Docker/Git write or delegation was performed by this owner. Original CPU failures are unchanged historical evidence. Parent requirements, including the275-line P2 repair allocation, were re-read before completion. The owner stops writing after delivery.

## Root execution milestone

The final source and package passed independent re-review, root RED/GREEN reproduction and actual locked-image/SDK checks. Root sealed seven inputs at source5f7d0950 and completed `exp-1024-020317-012085`:1H800/codesign/non-preemptible,16CPU/mem_gb64/600s ceiling; Succeeded2026-10-23T18:06:33Z,23container seconds,exit0,restart0. Receipts are `root_cpu.json`, `root_verdict_controls.json`, `matrix_binding.json`, `root_seal.json`, `root_submission.json`, `platform_terminal.json` and `root_run_check.json` in this directory.

All five outputs/228175bytes were downloaded hash-identically (`download.json`). Producer-reported eager PASS and graph UNKNOWN await independent raw-trace audit/review and root acceptance; execution success establishes no serving cost. The root inspector's initial GPU UUID prefix mismatch is retained with its exact-UUID correction in `root_run_check.json`.
