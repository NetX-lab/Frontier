## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded W08 owner rerun resolving both trace export failures: 18 passed in 12.06 s. |
| 2026-09-16 | Recorded source review and direct CPU verification for the W02 slot-owner cleanup, including rejected zero-KV scope premise and remaining W08 trace failure. |

# W02 slot-owner boundary verification

Reviewer/implementer: `/root/w09_artifact_review`. This report covers the follow-up removal of six production fallbacks, the corresponding narrow fixture migration, and the attempted all-GDN boundary reproduction. It supplements the existing W02 automatic-capacity and Simulator evidence.

## Changes and acceptance criteria

`VLLMv1EngineReplicaScheduler.__init__` unconditionally initializes `_gdn_state_slot_manager` to `None` before installing the supported MONOLITHIC GDN owner. Six admission/continuation/release reads now use that invariant directly. The lightweight-fixture compatibility comment was removed. Three method-level test fixtures explicitly initialize the absent owner in `test_pdaf_decode_attn_preemption.py` and `test_prefix_cache_scheduler_frontier.py`. Existing rollback, queue state, slot retention, idempotent release, and allocation numeric assertions remain intact.

PASS requires normal hybrid and non-GDN construction, automatic capacity arithmetic, owner retention/reuse, rollback on slot failure, preemption rejection before mutation, and unchanged prefix/PD-AF focused behavior. CPU Simulator validation must additionally reach all request completion and export assertions. Native AMD/ROCm profiling parity is outside this evidence.

## Rejected scope premise: all-GDN zero-KV scheduler

A normal `dataclasses.replace` construction of the supported Qwen3.5 MoE config with either all-linear `layer_types` or an interval beyond the model's layer count fails in `BaseModelConfig.__post_init__` -> `ModelArchitectureProfile.validate_structural_requirements` -> `_requires_qwen3_5_hybrid_gdn_contract`. The exact failure requires **both GDN and full-attention layers**. It occurs before the scheduler's initial block-budget check.

Consequently, the suspected all-GDN zero-KV failure is not reachable through the supported normal configuration. `BaseReplicaScheduler`, zero-KV admission, and memory-percent behavior were not changed; no fake positive KV allocation was introduced. Two constructor rejection regressions were added to existing `tests/unit/test_gdn_runtime_guards.py`. Raw investigation log: `/data/ycfeng/tmp/pr33-w02-zero-kv-red.log` (two expected-to-reproduce scheduler tests instead failed at the authoritative model guard; replaced by explicit guard tests).

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Interpreter: `/usr/bin/python`, Python **3.12.3**. No conda activation was used. Tests are CPU tests; GPU tools were not invoked.

Common environment and invocation prefix:

```bash
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest
```

Broad fixture/lifecycle verification appended these arguments:

```bash
tests/unit/test_gdn_runtime_guards.py tests/unit/test_gdn_scheduler_slots.py tests/unit/test_pdaf_decode_attn_preemption.py tests/unit/test_prefix_cache_scheduler_frontier.py tests/unit/test_prefix_cache_identity_ledger.py tests/unit/test_pdaf_cluster_scheduler_invariants.py tests/unit/test_mtp_token_ledger_repair.py tests/unit/test_mtp_terminal_overshoot_ep_replay.py tests/unit/test_kv_transfer_completion_contract.py tests/unit/test_pdaf_decode_attn_online_cohort.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-slot-owner-2 > /data/ycfeng/tmp/pr33-w02-slot-owner-2.log 2>&1
```

After the W08 EP interfaces were installed, the focused rerun appended:

```bash
tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_real_simulator_cpu_e2e 'tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_production_constructor_cpu_e2e[1]' 'tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_production_constructor_cpu_e2e[3]' tests/unit/test_gdn_scheduler_slots.py tests/unit/test_gdn_runtime_guards.py tests/unit/test_pdaf_decode_attn_preemption.py tests/unit/test_prefix_cache_scheduler_frontier.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-slot-owner-final > /data/ycfeng/tmp/pr33-w02-slot-owner-final.log 2>&1
```

## Observed evidence

| Check | Result | Interpretation |
| --- | --- | --- |
| Initial six-fallback removal, before fixture repair | 463 passed, 3 failed, 19 skipped | Exactly three lightweight fixtures lacked the now-required owner initialization. |
| Broad run after fixture repair | 478 passed, 3 failed, 19 skipped, 8.98 s | All three failures were the concurrent missing `MetricsStore.ep_wave_reporting_enabled` interface at `ep_wave_schedule.py:69`; no slot-owner failure remained. |
| Rerun after EP interface implementation | 38 passed, 2 failed, 10.91 s | `real_simulator_cpu_e2e` and all slot/guard/prefix/PD-AF focused tests passed. Both production-constructor E2Es completed lifecycle assertions but failed at the trace assertion `assert dense_events` (line 982). |
| W08 owner final reporting/hybrid rerun | 18 passed, 12.06 s | All three previously blocked hybrid nodes passed; log `/data/ycfeng/tmp/pr33-w08-reporting-hybrid-2.log`. |
| W11 combined attention/config/memory/artifact test run | 223 passed, 14.91 s | Includes all slot-owner and guard tests plus memory arithmetic and actual artifact fits; log `/data/ycfeng/tmp/pr33-w11-attention-training.log`. |
| Changed-file whitespace check | PASS | `git diff --check` passed for all lane files. |

The two remaining trace failures were reported to the W08 owner with the actual JSONL. The single-request output contained 251 trace rows but only five expanded rows, all layer-0 GDN FFN/normalization operations; all attention operator rows were aggregate. This rules out merely adding `attn_prefill` to the dense-event filter as a sufficient fix. Lifecycle acceptance was observed before the export assertions: all requests admitted, continuations retained identity, completed owners were released, and final allocation/queue/running counters were empty. The W08 owner subsequently corrected expansion ownership and reran both `test_stage_reporting_contract.py` and `test_gdn_hybrid_e2e_increment14ab.py`: **18 passed in 12.06 s**, including all three originally blocked hybrid nodes. This closes the observed export failures; it is attributed owner-run evidence, not a second execution by this reviewer. The reporting file subsequently gained two quota tests, so its current collection is larger than this historical run.

No native state tensors, ROCm kernels, measured GPU latency, or hardware performance parity were validated here. No production edits were made during the W11 review phase.

W08 owner exact successful command (current worktree, Python 3.12.3, no conda):

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/unit/test_stage_reporting_contract.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w08-reporting-hybrid-2.log 2>&1
```
