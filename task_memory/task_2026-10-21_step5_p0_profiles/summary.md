# D21-P0-PROFILES completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive committed native producer support,validated independent P0 role packages and parent-only native handoff. |

## Overview

Source commit `a7551a3a26ec84c66af9a88d282019f47e5ab52b` adds the missing native one-slot request-free idle-DP producer path and a bounded independently callable P0 PREFILL/DECODE measurement package. It does not substitute ordinary decode,history or zero work. Native metadata and slot mapping are None;source-defined projection/norm/zero-fill scopes remain positive;cache mutation is rejected. No predictor/simulator/admission or linear/MoE protocol source was changed. Existing D22 timing/replay/scopes modules are byte-identical to starting bc8ae27.

## Deliverables

- Existing `frontier/profiling/step5/{inputs,rows,vllm_wrapper,main}.py`: explicit metadata identity,full physical table/capacity,role labels,native prefix/reference page preservation and no-metadata invocation.
- Existing `frontier/profiling/step5/device_activity.py`: actual per-iteration kernel counts;wrapper exports root count separately from timing sample count/aggregate trace count.
- `frontier/profiling/step5/{p0_profile_plan,p0_prefill_inputs,p0_decode_inputs}.json` and existing `job.py`: exact bounded grids,native P0 runtime policy,independent role producer commands and one-H800 non-preemptible packages.
- Focused CPU tests and existing local preflight/judge wiring;only the old boundary CPU description matcher/default-field expectation was updated for the expanded input type. No boundary native experiment/retry occurred.
- `task_memory/task_2026-10-21_step5_p0_profiles/{requirements,plan,progress,handoff,test_report_2026-10-21_p0_profiles}.md` and `evidence/`:requests,checks,exact commands,baseline comparison,manifests,source/hash audit and limits.

## Observed verification

- Actual native metadata/forward seam RED→GREEN,cache-mutation rejection,physical448-column/page-offset/8192-budget checks and unchanged consumer loader checks.
- 448 final focused tests PASS. Latest per-ID combined coverage998PASS/4 starting-commit failures/1002total;all980 starting IDs retained,22 added,no new failure. The4 old execution-disabled assertions are independently reproduced from bc8ae27. One Git-less archive-only baseline limitation is explicitly recorded.
- Both sealed stage preflights PASS:3415 payload hashes/exact inventory and533 committed source matches per role;all10 actual sealed CLI dry-runs PASS. Original D18/D22 stages retain all3407 original payload hashes each;old verdicts not recomputed.
- PREFILL stage `/data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01`,manifest `82dfc1146321e92a8fffdffc41e27e2572d5d5993bf95c0d5be44a4cf1c0496c`:4attention event+4GPU-active,24linear rows total,42MoE steps/method.
- DECODE stage `/data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01`,manifest `f09b6fc0436889042dabb6fb0d322b9a9d9c17b9b4860f425342b8e9a50a5ca5`:20event+4GPU-active,28linear rows total,48MoE steps/method.
- Each package:1H800,codesign,non-preemptible,8CPU/32GiB,3600s cap. Planning estimate30–60minutes per role;maximum2GPUh combined. Zero local platform/GPU/auth/network/cloud/SSH/Docker actions and no delegation.

## Open work and acceptance limits

Parent alone submits and monitors using `handoff.md`. New H800 native timings remain UNVERIFIED;linear quality remains unqualified. Prepared attention layers0/3 are not all92 identities;decode widths1/2 are not a reached-width census;DSA partial-summary contexts,graph padding,mixed zero/nonzero history and actual routing distributions are not qualified or substituted. Separate first input norm/input cast/final residual,embedding/sampler,host/launch/gap and communication owners still need compatible measured inputs. This is executable producer support and a bounded first native package,not full-profile admission,32-GPU serving-capacity,canonical/G3 or <=10% accuracy qualification.
