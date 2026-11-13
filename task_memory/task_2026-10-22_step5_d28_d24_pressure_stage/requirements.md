# D28 D24 pressure staging requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded verbatim D28 approval and local-only capture03 ownership. |
| 2026-10-22 | Added balanced native routing and workflow/GPU-before-CPU steering. |

## [Original Request / Decision] D28 — 2026-10-22

Parent's exact question:

> 是否将D-only --num-gpu-blocks-override32000合并到本次D24诊断？父端已复现native CPU allocator的真实拒绝；需要接入D-only启动参数并生成新包。保持原workload、max_num_seqs128、P容量、原生拒绝判据，以及H80032/90分钟/48 GPU-hour上限。

YC's verbatim answer:

> 合并压力控制 (Recommended)

## [Original Request] D28-D24-PRESSURE-STAGE

Resume in `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`. Own minimal D-only existing native flag wiring, recorder/gate capture only if actually necessary, affected tests and fresh exclusive `/data/ycfeng/tmp/step5_d24_native_20261022/capture03`. Capture02 is sealed, immutable and unsubmitted; never relabel it as already using the override. Record new D28 facts here without changing old receipts.

Parent owns every GPU/platform/authentication/cloud/submission/stop operation. Local/offline only; no further delegation or backend instantiation. D26 P lifecycle and D27 SWA are separate CPU streams, and no files or commits from those streams may be edited/merged. The pressure-capacity proposal branch may be read for evidence, not merged wholesale.

Preserve workload, role topology, ports, API count, request count, prompt/output, native runtime, original criteria and H80032 non-preemptible `codesign` 5400-second /48GPUh ceiling. P remains automatic; only D uses32000. Independent references run on the same accepted D pool. No numerical tolerance, retries, connection change, SWA or P-lifecycle repair. All50 native pins remain unchanged. Auxiliary validators remain soft; native failure semantics stay intact.

Parent-verified files: `/data/ycfeng/tmp/d24_parent_pressure_{baseline,denial}_dp{0,1}_01.json` and `/data/ycfeng/tmp/d24_parent_pressure_startup01.json`. Native CPU denial at computed6400: actual demand4 > contemporaneous available3, free0→3 after native skipped retirement, final free31999. Admission of64 CPU-synchronized requests does not prove GPU queueing/preemption/denial timing. Actual decode None, demand/free/watermark/reserved, ownership and READ correlation remain required native evidence.

## [Original Request / Steering] Native balanced routing and calibration order

YC's verbatim requests, supplied by the parent during this task:

> routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定）

> 先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量，不要在当前阶段就引入cpu overhead的大批分析、测量和suits，推迟到矫正最后。

Inspect actual native Step route and existing `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` accepted assignments before freezing or accepting the next package. Incorporate a verified balanced setting or report the unsupported mechanism. Do not confuse uniform random probabilities or EPLB placement with balanced token assignment. Parent owns Frontier semantics; this child performs native source/CPU assignment checks only. No CPU overhead harness or extra platform permission. A new custom strategy/native patch requires a separately approved design; no speculative environment flag is set.

## Acceptance checklist

1. One existing role configuration mechanism; new package D argv includes32000 and P argv/effective expected config remains unchanged. Default/no-override control remains supported and byte-equivalent. No metadata mislabel.
2. Existing estimate-result hook retains actual failed-allocation pregrant demand, proved at the real native call seam on CPU/fault injection, with no granted-delta substitute/formula/new GPU read or sync.
3. Existing builder/node verifier, source/import origins, current committed source and hash pins are consistent; fresh bounded capture03 passes scoped tests, staged collector fixtures, compilation/bash syntax and offline ResourceConfig/dry-run. Never instantiate backend.
4. Capture02 and original evidence unchanged; no capture03 job/attempt/submitted marker. Code/tests committed after verification, records separately, clean branch, no push. Provide diff artifact/changed list for primary review, exact paths/identities/results/blockers and live-check requirements.
