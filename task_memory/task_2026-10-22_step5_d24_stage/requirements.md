# D24 native staging requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded parent-owned D24 offline staging scope and acceptance. |

## [Original Request] D24-NATIVE-STAGE

Own `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`, created clean on branch `feat/step5-d24-stage` at `ebd5baf`, and this new task directory only. Finish an independent fresh D24 native diagnostic package and offline preflight using the existing canonical G3 builder. No production code changes; report blockers before any required code change. No GPU, platform, authentication, cloud access, SSH, submission, stop, or further delegation. Never instantiate `RJobBackend`. Offline `ResourceConfig` validation is permitted only without network/API/authentication calls. The parent performs every actual StepMind operation.

Authority: primary `task_memory/task_2026-10-15_step5_preview_h800/{requirements,plan,spec}.md`, particularly D24 and plan lines 123–132. Native staging is independent of CPU actual-main. D25 H80032/64 full workloads are completed and must not be repeated.

## [Original Request / Decision] D24

YC selected `补采后定数值判据 (Recommended)` after the parent's proposal to combine bounded raw-value capture with the next H80032 diagnostic under the original 90-minute / 48 GPU-hour ceiling.

Retain the computed-token logical page and its predecessor raw/mean values at `first_local_tail_complete` for actual and independent-reference requests, using existing host copies. Preserve complete hashes elsewhere. State disagreement remains **UNVERIFIED**, transferred content remains bit-exact, and numerical-rule approval is a separate decision.

## Frozen scope

- Reference package: `/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01/stage`; all original stage/evidence bytes are immutable.
- Fresh exclusive candidate: `/data/ycfeng/tmp/step5_d24_native_20261022/capture02`.
- Cloud destination string only: `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d24_capture02`; no cloud access.
- H800 / `codesign`, 32 non-preemptible GPUs, P16/D16, TP8/DP2/EP16/PP1. Native settings, resource shape, diagnostic waves, ports, windows and ceilings match capture01.
- Canonical 6144/1024 prompt/output settings, 7168 context, 8192 token budget, 128 sequence capacity; existing adjacent boundary/reference controls remain unchanged.
- No D-only 32000 block override, P lifecycle/backoff repair, SWA repair, tolerance, retry, connection change, or waiver. Pressure-control proposals do not delay existing staging.
- All auxiliary validation at this GPU size remains a soft diagnostic; native I/O and native workload failures retain their original semantics.

## Acceptance

Fresh complete stage, actual imported-module origins/checkouts bound against Git (not just nested hashes), no root/cwd shadowing, exact 50 installed native source pins, configured scripts/bounds/mounts/creator expectations, source compilation and bash syntax PASS, staged HTTP/startup/two-page raw-value capability, static all-rank/reference/budget readiness. Reuse existing tests/preflight rather than adding a broad harness. Commit only task records/receipts after checks, no push, clean branch. Deliver source/stage/manifest identities, commands/interpreters, PASS/FAIL/UNVERIFIED boundaries, parent live checks and submission command, and explicit absence of a submission marker/job.
