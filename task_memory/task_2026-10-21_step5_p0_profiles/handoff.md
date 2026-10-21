# D21-P0-PROFILES parent-only native handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Prepare independent P0 role commands, bounded coverage, native budget and admission limits. |

## Outcome and ownership

Missing request-free producer support is implemented in the existing input/metadata/forward/profile path. It constructs no attention metadata or slot mapping, performs no prefix preparation, requires unchanged cache state and records the existing source-defined positive native scopes. Actual candidate/reference sampling remains the approved D22 independent protocol. No simulator, predictor, typed admission or linear/MoE protocol changed.

The parent alone checks quota/auth/priority, submits and monitors. This task ran no GPU, platform, auth, network, cloud, SSH or Docker operation. D22 H20018/18 candidate,22/22 functional,18/18 prefix PASS is protocol evidence only; no H200 time is relabelled as H800.

## New stages and resource budget

Source commit `a7551a3a26ec84c66af9a88d282019f47e5ab52b`. Each stage passed its sealed verifier,all3415 payload hashes/exact inventory,all5 producer CLI dry-runs,shell syntax and533 committed source payload matches. No source changes followed sealing.

| Role | Sealed manifest SHA256 |
| --- | --- |
| PREFILL | `82dfc1146321e92a8fffdffc41e27e2572d5d5993bf95c0d5be44a4cf1c0496c` |
| DECODE | `f09b6fc0436889042dabb6fb0d322b9a9d9c17b9b4860f425342b8e9a50a5ca5` |

Retained preflight/audit:`evidence/{prefill,decode}_package_preflight01.json`,`evidence/source_stage_audit01.json`. Original D18 and passed D22 stages each retain all3407 original payload hashes and unchanged manifest;no verdict was recomputed.

Both new packages use one isolated H800 TP-local rank, codesign, **non-preemptible**,8CPU/32GiB,3600s cap. They feed the32-GPU serving case(P16/D16,eachTP8/DP2/EP16/PP1) but do not reserve or emulate32 serving GPUs. Parent can run the two packages independently if that does not delay P0 serving. Expected planning range30–60minutes per role; hard maximum1GPUh per role,2GPUh total. This is an unmeasured planning estimate,not a runtime claim; no automatic expanded-budget retry is authorized.

```bash
bash /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01/submit_with_env.sh
bash /data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01/submit_with_env.sh
```

Keep each local launcher alive for its worker lifetime. The unchanged personal-auth launcher owns submission. Stop only through the existing approved rjob stop helper. Do not source credentials or invoke either submit command during CPU preflight. Logs and retained native traces/CSV outputs go exclusively below the manifest's `/mnt/codesign-exp/ycfeng/frontier/step5_profiling/p0_h800_{prefill,decode}_20261021_01/`.

## Prepared measurement coverage

Source grids: `frontier/profiling/step5/p0_{prefill,decode}_inputs.json`. Explicit native policy/topology/limits: `frontier/profiling/step5/p0_profile_plan.json`.

| Item | PREFILL package | DECODE package |
| --- | --- | --- |
| Global attention layers | 0 SWA,3 sparse GQA | 0 SWA,3 sparse GQA |
| Native no-metadata workload | one physical slot,context0,eager,each layer | independently collected same identities,not P samples |
| Live attention | q6128/context0,eager | q16/context6128 in separate eager and PIECEWISE cells;decode widths1/2 at contexts6144,6656,7160;SWA also7167 |
| Event rows | 4 | 20 |
| GPU-active rows | 4 | 4 |
| Strict candidates | 4 | 20 |
| Functional row checks | 8 | 24 |
| Prefix-required row checks | 0 | 20 |
| Linear token points | 1,2,6128,6129,6144,6145 | 1,2,3,4,16,17,32 |
| Linear rows,per method | 12 | 14 |
| MoE token points | 1,2,11,6128,6129,6144,6145 | 1,2,3,4,11,16,17,32 |
| MoE source groups/EP | both actual FP8/BF16 routed groups,EP16=22 local experts | independent same source groups/EP16 |
| MoE profiled steps,per method | 42 | 48 |

Each package runs attention plus both original linear methods and both original MoE methods:5 producer invocations per role. Full commands are in each sealed `manifest.json` under `producer_commands` and `worker_command`,and retained `evidence/producer_commands.json`. MoE's existing CLI executes requested token points in descending order. Neither uniform routing load points nor widths1/2 are labelled observed P0 scheduling distributions.

Native configuration matches retained H800 init evidence: max_model_len7168,max_num_seqs128,max_num_batched_tokens8192,block16,full physical448 columns,35 actual capture buckets1..256,FULL_AND_PIECEWISE,chunked prefill off,prefix cache off,dummy weights,noMTP. Request-free rows alone have table extent0 and `has_attention_metadata=0`. Prefix/reference preparation preserves page IDs and splits requests outside timing when their combined history exceeds8192. Explicit eager references are not relabelled PIECEWISE; they remain a separate measurement family and cell identity.

## Parent qualification after collecting worker outputs locally

Use the exact sealed judge with unchanged5% CV/agreement thresholds. These are local receipt commands,not GPU submissions; replace `<local-role-output>` with the downloaded role directory containing `summary.json` and the standard CSV paths.

```bash
PYTHONPATH=/data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01/tests/integration/step5_profiling/judge_attention_smoke.py \
  <local-prefill-output>/attention --expected-event-rows 4 --expected-gpu-active-rows 4 \
  --max-time-cv 0.05 --report <new-prefill-receipt.json>
PYTHONPATH=/data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01/tests/integration/step5_profiling/judge_attention_smoke.py \
  <local-decode-output>/attention --expected-event-rows 20 --expected-gpu-active-rows 4 \
  --max-time-cv 0.05 --report <new-decode-receipt.json>
```

No boundary probe or diagnostic/retry is requested. Do not overwrite any prior stage or receipt. A native quality failure remains FAIL; do not retry based on quality,subtract overhead,alter thresholds or reinterpret historical receipts.

## Remaining admission gaps

- Native execution of these new H800 rows is still UNVERIFIED; linear quality remains unqualified even if positive rows appear.
- Exact consumer requires all92 global layer identities. This bounded package measures0/3;the other90 remain missing. The callable accepts explicit other layer IDs from the real source schedule;measured timings are never copied across layer IDs.
- Real P0 reached decode arrays,variable widths and graph padding are not yet collected. Prepared widths1/2 are benchmark points,not a census. Exact unpadded capture widths and heterogeneous positive contexts are callable;bucket-gap padding and mixed zero/nonzero histories are explicitly unsupported,never silently rounded or replaced.
- The idle-DP native workload is exactly `_dummy_run(1)`:one physical slot,no requests/history/attention metadata. Wider synthetic dummy cells are rejected. Attention rows alone do not measure its complete dense/shared/MoE/communication cost.
- DSA repeated timing only supports region-aligned contexts. Actual partial-summary tails remain missing;prepared aligned points do not substitute for them.
- MoE EP16 profile load is uniform and does not qualify actual reached routing distributions or EP communication. Ordinary linear token points do not establish missing strict quality admission.
- Existing separate owners still need compatible first input norm/input cast/final residual,sharded embedding,sampler,host/launch/gap and communication measurements. No full-path or <=10% accuracy claim is made.
- The old execution-disabled assertions in `test_step5_linear_op_ownership.py` conflict with starting bc8ae27 consumer enablement and are reported separately,not fixed by disabling Step execution.

Native census -> exact additional role inputs -> strict compatible full-grid measurements -> unchanged consumer admission is the remaining parent route. This delivery supplies the missing callable and a runnable bounded first package,not a new inventory-only stopping point.
