## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Updated fresh runtime, workload and profile evidence; retained unresolved comparability gates. |
| 2026-09-07 | Recorded the preliminary semantic audit before formal measurements. |

# Semantic Alignment

- analysis_state: INCOMPLETE
- status: INSUFFICIENT_EVIDENCE
- correction_state: pending
- next_action: Complete identified diagnostic runs; match routing vectors, actual communication primitives and runtime operator shapes, then evaluate the new Frontier baseline against official server TTFT.

The parallel-domain mapping is structurally validated by nine tests. Backend D001 is confirmed. Formal numeric parity is not admitted while UNSET/MISMATCH rows remain. This preliminary audit does not prove a new H200 performance root cause or authorize a shared-contract repair.

The old compute profiling used /usr/bin/python3 (Python 3.12.3) and /data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2, while old vLLM used the pinned image's Python 3.10.16. The new profile environment must be validated against the effective runtime; old timing datasets remain excluded.

Current evidence supersedes the preliminary gaps above: H200 topology and collective_sim/nvlink_analytic runtime passed; D001-D004/D006-D007 confirmed; three full drained clean warmups and100 unique formal requests passed; fresh H200 BF16/CUDA_EVENT profile files and shared runtime versions passed. D005 is deferred. No old numeric profiling or cache is used.

The queued Frontier run is a diagnostic baseline for the requested comparison/RCA. It is not a semantic PASS or accepted E2E result. Remaining gates: exact operator/phase/shape coverage, routing abstraction, communication primitive/domain agreement, and independently evidenced missing official-TTFT critical-path work. No new user choice is pending for the current diagnostic repair.

## D011/D012 current uniform execution scope

D011 supersedes the traced-routing import plan for this calibration. D012 is explicitly approved and committed 9e3d1874 after 74 relevant checks. Frontier uses balanced loads with explicit uniform_topk; the current vLLM run uses VLLM_MOE_UNIFORM_ROUTING=1. On eight H200 workers, actual vLLM router outputs match Frontier materialization at identical populations 1,4096,4097 (24/24 checks, zero per-expert count distortion). Fresh uniform profiles contain 774 rows and matching router metadata. The new trace contains exactly the current 100 formal queue arrivals.

CPU generation01 is admitted as the requested diagnostic baseline. It retains H200 hardware parameters and current-task GPU measurements, with fresh CPU-trained predictors (Python3.13.13/sklearn1.9.0). No historical trained models or old arrival traces are inputs. Full request validation follows completion. This does not claim matched distributed batch populations or close operator/communication/missing-host-work evidence gaps. Uniform routing removes the implementation/load-distribution mismatch at equal population; source-DP/dummy scheduling remains a separate question.
