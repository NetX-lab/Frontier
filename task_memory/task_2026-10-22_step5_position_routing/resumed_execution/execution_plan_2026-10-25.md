# Current-source joint timing and worker execution plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Archive the completed single readiness worker and exact parent verification; refresh current worktree and workflow status without new pricing or shape calls. |
| 2026-10-24 UTC / 2026-10-25 HKT | Close preparation Standards/Spec follow-ups with No findings; freeze task-only scripts before one readiness worker. |
| 2026-10-24 UTC / 2026-10-25 HKT | Verify the final copied Git and task-scoped configuration in the fixed image; preserve prior failures and affected static06 controls. |
| 2026-10-24 UTC / 2026-10-25 HKT | Repair preparation review findings and the SDK field/mount checker; retain original failures and zero runtime calls. |
| 2026-10-24 UTC / 2026-10-25 HKT | Prepare the exact two-wave timing driver, full parity checks and one zero-GPU worker; no pricing or platform operation. |

The production source is `2ca5e57113b0bc99ab84e926e5bb492b3670072e` and equals frozen `1461573d` outside `task_memory`. The existing116-path/126-binding budget remains CORE4377/4404, PRICING3955/3956, native1350/1582, source9682/9942 and global9696/10000. These are single-task scripts in the retained evidence bundle, not new production modules or source allocations.

Preparation has two independent branches:

`source admission + existing runtime PASS -> {joint script + finite pricing decision, worker scripts + original H800/resource authority} -> current-wave and worker observations -> unchanged complete6144/1024 CLI -> final T2/source/package -> native/G3/E2E`.

The first two preparation deliverables and the single authorized job `exp-1025-050336-389487` for worker import, reference append and lifecycle are complete. Actual joint timing remains unperformed pending its finite decision. See test_report_2026-10-25_worker_readiness.md and evidence/worker_readiness_2026-10-25/. Do not rerun source regressions, local/image probes, copy/init diagnoses, old five-pricing/fifteen-emission controls, exhausted required-shape6/6 or the completed readiness job.

## Joint timing: two-call proposal

The scripts live under `evidence/execution_preparation_2026-10-25/`:

- `joint_setup.py` contains the retained setup prefix before the old `MetricsStore.__new__` assignment. It loads existing checksum-checked models but never prices a wave.
- `joint_wave.py` contains exactly the retained `phases = predict_ep_wave_phase_times(...)` AST. Source and trace-module bindings are updated to current admitted bytes.
- `joint_timing.py` creates the real `MetricsStore`, initializes the cached manager's original constructor-owned `CollectiveProfileInputs`, binds the real backend, and performs cold/warm pricing once each in the same process. There is no warmup, profiling loop or automatic retry.
- `joint_bindings.json` freezes25 source paths, six exact input hashes,262 fixture metadata entries, the original wave AST and complete pricing/ledger/operator references. Reference operators are exactly the208 rows after the header; the rest of the historical multi-emission file is not scanned.

Each wave gets new phases, timing, plan, MetricsStore and output writer. The timed interval includes production pricing, plan/timing construction, `on_ep_wave_schedule`, complete16-lane ledger appends with each native flush, and final trace/ledger close. TraceStore keeps the original5000-event threshold. MetricsStore construction and trace header initialization are outside the wave interval and must not be presented as included startup time.

Validation follows the timed interval. It compares all208 operators,144 compute proofs, complete pricing bytes and all16 ledger rows with exact values, keys, proof content and order. New ledger framing is independently required to be sorted JSONL. Newly allocated EP batch IDs and synthetic destination request IDs must follow the original contiguous allocation and bind to each current lane record; captured source batches, routing identities and all other fields remain exact. It rejects stale cold-plan reporting, dropped rows, reordered lanes, changed proofs, changed durations, extra events and changed captured source IDs.

Proposed authority: **at most two new ordinary captured D1/layer60 CPU pricing calls**, one cold and one warm, under one60s/30GiB execution. Required-shape calls remain0; the already consumed6/6 ceiling is unchanged. The task's root plan expressly states that a new label or resume grants no extra pricing calls. The proposal therefore remains unexecuted until a separate finite decision is recorded.

Exact proposed command, with an unused output directory:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G \
  runuser -u brainpp -- env \
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -B \
  task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/joint_timing.py \
  --bindings task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/joint_bindings.json \
  --output-root /data/ycfeng/tmp/d29_current_joint_20261025_01
```

Record the command, environment, attempted calls, cold/warm components and joint totals, complete output parity, actual probe RSS and source/input bindings. A timeout or failed parity remains a raw failure and consumes attempted calls; never silently retry. Do not multiply this one-layer observation into a claimed whole-CLI PASS or combine it with older components.

## Worker: one zero-GPU readiness execution (completed; retained plan)

`worker_plan.json`, `launch_worker.py`, `worker_entry.py`, `worker_append.py` and `run_worker.sh` prepare one codesign worker:4CPU,0GPU,64 `mem_gb`, one replica, H800 placement, explicit published imagef037d958, no extra requirements, no RDMA requests and no retry. Root requirements' existing H800 task-resource authority applies; this preparation requests no new GPU allocation or native/profile collection.

The local launcher must originate on `kun-workspace-vgen2`. Original `spawn_tasks()` resolves the current host address and mounts the local `/data/ycfeng` parent at the same path, covering Frontier/Git metadata, fixture inputs and both copied runtime prefixes. `/etc/exports` already covers `/data`; actual worker connectivity and platform source annotation remain to be checked. Do not substitute a remote mount or edit mount encoding.

The cloud mount remains `juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp`; all access is confined to `/mnt/codesign-exp/ycfeng`. OSS quota is USER_CONFIRMED_SUFFICIENT, not a new measurement. A fresh test directory is selected in the plan. The worker runs the unchanged import probe once in its own environment and one serialized-reference append control; pricing/shape/fullCLI/native calls are0.

Append preserves native row-flush and5000-event trace-open/close patterns using complete retained serialized rows, with a30s/1GiB upper bound and60s child limit. It records actual bytes, append/close, fsync, full read-back digests and scoped cleanup. Its timing includes replay/digest work and excludes production pricing/emitter JSON serialization; it is not isolated storage bandwidth or full-workload performance. The historical469GB/mode projection gives an approximate195.4MB/s minimum if all2400s were available to writes, not a readiness guarantee.

The launcher stays alive through the original `spawn_tasks()` wait. Capture the exact created job from its log, inspect creator/source/image/resources/replica/container state through the original personal Python API, and require both business artifacts and terminal success. A queued job is retained rather than resubmitted. Any required stop uses `/data/ycfeng/steptron/playground/users/ycfeng/tools/rjob_stop.sh`. Archive selected local evidence, then remove only the two completed cloud test files after hash verification; preserve unverified files for bounded diagnosis.

Prepared submit entry:

```bash
bash task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/run_worker.sh
```

This entry was not run during preparation; its single authorized execution has now completed. It configures existing personal credentials before any client and performs exactly one local Python RJobBackend submission. It does not install packages, change credentials/VPN/v1 configuration or run native profiling.

## Original SDK materialization and review boundary

Preparation follow-up repairs full trace-header identity and frozen cache-name handling without pricing. Finalstatic05 verifies10 expected output corruptions; finalworker_spec03 verifies the exact original SDK objects without client creation. The first field error and intermediate mutation-tainted mount check remain raw evidence.

The original ResourceConfig.__iter__ appends juicefs+s3://oss.i.shaipower.com/tenant:/mnt/shared-storage/tenant for every task. The requested mount list is copied, and the checker verifies the requested codesign mount plus only this documented SDK default and an unchanged plan. The worker reads/writes no tenant data; all task cloud reads/writes remain under /mnt/codesign-exp/ycfeng. This records original SDK behavior and changes no SDK, submitted mount request or task scope. Follow-up review closes before the one readiness submission; the two new ordinary pricing calls remain unapproved.

## Final copied-runtime Git verification

The fixed image lacks git, and its root UID differs from the owned source UID10250. Retain Git01's FileNotFoundError, Git02's original ownership failure and Git03's exact diagnostic child stderr. The copied host Git2.34.1 is already in the mounted runtime; the task-only configuration file contains exactly the two owned repository paths. worker_entry.py verifies both asset hashes, selects that runtime PATH and sets GIT_CONFIG_GLOBAL before the real source binding call. Host/image global configuration and production source are unchanged.

worker_git_preflight04 passes these exact assets/configuration and two real check_bindings calls in the same pinned image, read-only mount, network none and60s/30GiB configuration. Static06 passes the final prepared hashes, full retained outputs, all10 expected rejections and no-operation worker dry run. Do not add Git probes after the passing affected checks. Both existing follow-ups now have No findings. Commit the task-only changes and hold that records HEAD during the one readiness worker; its before/after binding check requires the HEAD to remain stable. The single actual worker now supplies import/append and platform lifecycle evidence separately from these retained preparation results.

## Complete workload follows observations

Use the existing `tests/integration/run_step5_measured_main.py` with the unchanged fixture and one fresh personal cloud output root. Its own `frontier.main` child remains capped at2400s for each offline/online mode; complete outputs and all validation checks remain required. The current runner validates after each child, outside that child's cap. Keep the local launcher alive throughout; do not reuse a5100s outer timeout that could terminate large post-run validation.

Record real full-workload RSS and memory/controller evidence while that unchanged workload runs. The64 `mem_gb` readiness resource is a requested allocation only; import/append RSS cannot establish the full-workload bound. Stage/event ledger retention remains part of actual workload validation. Final T2/source/package, native/G3 and each E2E error<=10% remain unfinished and keep their original requirements.
