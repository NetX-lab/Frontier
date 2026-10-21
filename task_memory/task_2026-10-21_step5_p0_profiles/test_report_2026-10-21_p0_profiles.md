# D21-P0-PROFILES CPU and package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record native-seam RED→GREEN,448 focused passes,unchanged inherited failures and independent role CLI preparation. |

## Result and limits

Producer/source checks PASS. Final focused suite: **448PASS**. Latest-result-per-test-ID union across the broad affected suite and final focused rerun: **998PASS,4 inherited FAIL,1002total**. No new failure versus starting `bc8ae27`;all980 starting IDs retained and22 added. These counts overlap;do not sum test runs. The broad command is not an all-green run.

The4 inherited failures are `test_step5_linear_op_ownership.py:66,160`,each parametrized for profiling/runtime. They assert attention/FFN execution disabled although `bc8ae27` already enables these consumers. An actual `git archive bc8ae27` source run reproduces all4 failures. That archive additionally lacks `.git`,so one existing boundary package description test fails only in the archive environment;it passes in the actual worktree. No simulator/predictor/admission change was made to satisfy stale assertions.

Compared with the original D18's851IDs:846PASS,4 now-inherited enablement FAIL,and one test was already renamed before this assignment (`test_smoke_grid_and_runtime_stays_disabled` -> current runtime-enabled smoke contract). No test was dropped by this change. See `evidence/baseline_comparison_final01.json` for exact IDs and environment limitation.

No native H800 timing,platform/auth/network/cloud/SSH/Docker operation or delegation occurred. Parent still owns GPU submission and native admission. D22 H200 native PASS is reused only as timing-protocol evidence;historical FAIL receipts are unchanged.

## Environment

- CWD `/data/ycfeng/Frontier/.worktrees/step5-p0-profiles`;starting commit `bc8ae27`.
- `/data/ycfeng/envs/oneshot-opd-py312/bin/python`,Python3.12.13;existing `/data/ycfeng/tmp/step5_profiling_20261020/extra_site`;no installs or conda environment creation.
- `PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1`.
- `PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-p0-profiles:/data/ycfeng/tmp/step5_profiling_20261020/extra_site`.
- Existing site-packages shadows the repository's `tests` namespace. As in D18/D22,the full commands explicitly bind `sys.modules['tests'].__path__` to local `tests/`. A standalone boundary pytest invocation without this bootstrap failed collection;it is a known command-environment issue,not a code failure.

## Meaningful RED→GREEN

Actual RED command:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-p0-profiles:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m pytest \
  tests/unit/test_step5_request_free_profiling.py::test_request_free_native_forward_receives_no_attention_metadata_or_slot_mapping \
  -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/p0_request_free_red01.xml
```

Expected/actual RED: missing native no-metadata input capability;`TypeError: Step5ProfileInput.__init__() got an unexpected keyword argument 'has_attention_metadata'`. Raw error retained in `evidence/red_seam01.json`;raw XML path above preserved. GREEN executes real `_metadata` and `_forward` functions with only native/CUDA boundaries replaced by CPU doubles,and asserts metadata=None,slot_mapping=None,one zero position,no metadata-builder call,positive physical scopes and distinct no-metadata feature/fusion identity. Injecting any cache mutation into real `profile()` is rejected before timing.

Additional CPU checks settle the actual P0 seams:

- Both SWA and sparse request-free rows from actual `build_row`/`rows_to_dataframe` load through unchanged consumer validation;active work is positive,inactive scopes exactly zero,root launch count separate from timing sample count.
- Full serving448-column physical table is actually allocated in native metadata,including unequal6144/6656 prefixes. Offset preparation preserves the second request's physical pages.
- Real `profile()` reaches its timing boundary with prefix/full-reference forwards each<=8192 tokens,without inflating the128-seq/8192 budget;long combined history is prepared request-by-request outside timing.
- Existing trace association counts root kernels per replay and excludes restore/prime activity,driver/runtime aliases and memcpy from kernel counts. Aggregate trace count>=10 is not substituted for the per-replay count1 in its fixture.
- No-metadata history,multiple slots,FULL graph and physical table extent are rejected. Explicit eager references and native PIECEWISE are separate cell identities/measurement families.
- Actual main loop writes independently labelled PREFILL/DECODE no-metadata rows. The strict judge retains5% CV/agreement and D22 independent-reference checks;no quality retry/overhead subtraction/new threshold.
- More than50 existing Step shape-boundary scenarios remain exercised in the affected suite. CPU values are explicitly fixtures,not H800 timing evidence.

## Exact verification commands and observed results

The complete executed child command arrays,including every filename,namespace bootstrap and JUnit path,are retained in:

- `evidence/commands_final01.json`: broad original D18 file list + fixed/boundary probes + request-free +8 measured-consumer regression files. Actual broad run997PASS,5FAIL in179.06s;one failure was the newly exposed optional-field assertion in the old boundary CPU description fixture.
- `evidence/commands_focused_final02.json`: final8 focused files,including all producer,timing,trace,judge,fixed and boundary CPU checks. Actual **448PASS in15.60s** after the narrow typed-default matcher and expected-field update. This rerun replaces matching IDs in the broad result,yielding998PASS/4 inherited FAIL.
- `evidence/baseline_commands01.json`: actual starting-source980-ID comparison,975PASS/4 inherited semantic FAIL/1 archive-no-git environment FAIL in191.21s.

Raw logs/XML under `/data/ycfeng/tmp/step5_p0_profiles_20261021/`: `affected_final01.{log,xml}`,`focused_final02.{log,xml}`,`bc8ae27_baseline01.{log,xml}`. Initial combined run timed out at120s;the recorded full runs use shell timeout0. A one-time wrong scheduler fixture import was corrected to the surrounding consumer tests' SimpleNamespace seam,not product compatibility code.

The old boundary experiment was **not run**. Its CPU-only typed input matcher needed to include dataclass defaults after adding metadata/physical-width fields;its timing/schedule/native experiment remains unchanged. No new diagnostic or native retry was introduced.

Byte equality to `bc8ae27`: `timing.py`,`replay.py`,`scopes.py`. No linear/MoE producer or predictor source changes. Evidence:`evidence/protocol_preservation01.json`. Whitespace/source-scope checks precede commits.

## Executable role package preparation

All10 actual producer CLI dry-runs returned0:attention +linear(cuda_event,record_function)+MoE(cuda_event,record_function) independently for PREFILL and DECODE. Exact worker invocations:`evidence/producer_commands.json`;grid/step counts:`evidence/coverage01.json`.

Actual retained H800 initialization policy was parsed and compared with the committed configuration:35 capture buckets,7168 max context,TP8/DP2. Source log SHA256 `0faf38c3df0b32977616d5664a11be038840f82e490b1fd313a428a0b1f87943`;full path,line29 and exact buckets in `evidence/native_policy_evidence01.json`. This proves configuration only,not reached widths.

PREFILL:4event+4GPU-active,12linear rows/method,42MoE steps/method. DECODE:20event+4GPU-active,14linear rows/method,48MoE steps/method. Each role is one H800/codesign/non-preemptible worker with3600s cap. All no-metadata cells have one slot;normal physical table width448;all logical shapes remain independently sampled. Prepared layers0/3 and widths1/2 are bounded measurement points,not complete92-layer/variable-shape admission.

## Sealed package verification

Source commit `a7551a3a26ec84c66af9a88d282019f47e5ab52b`. Both actual sealed preflights **PASS**:3415 payload hashes and exact inventory,common workload identity,attention shapes/capacity/policy,both linear and both MoE dry-runs,launcher shell syntax. Each stage's533 tracked payloads match that committed source;remaining payloads are reused offline dependencies and generated launch wrappers. The manifest is hashed separately. No source edits followed sealing.

| Role | Stage | Manifest SHA256 | Linear rows | MoE steps/method |
| --- | --- | --- | --- | --- |
| PREFILL | `/data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01` | `82dfc1146321e92a8fffdffc41e27e2572d5d5993bf95c0d5be44a4cf1c0496c` | 24 | 42 |
| DECODE | `/data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01` | `f09b6fc0436889042dabb6fb0d322b9a9d9c17b9b4860f425342b8e9a50a5ca5` | 28 | 48 |

Actual stage commands differed only by role/run ID/stage:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-p0-profiles:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m frontier.profiling.step5.job \
  --p0-role PREFILL --stage /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01 \
  --run-id p0_h800_prefill_20261021_01 \
  --dependency-site /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/offline_site
```

DECODE exact substitutions:`--p0-role DECODE --stage /data/ycfeng/tmp/step5_p0_profiles_20261021/decode_stage01 --run-id p0_h800_decode_20261021_01`.

Actual sealed verification commands used the role-specific stage on both PYTHONPATH entries and script path:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01:/data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01/offline_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01/tests/integration/step5_profiling/verify_fixed_probe.py \
  --stage /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_stage01 \
  --output /data/ycfeng/tmp/step5_p0_profiles_20261021/prefill_package_preflight01.json
```

The DECODE command replaces every `prefill_stage01` with `decode_stage01` and output basename with `decode_package_preflight01.json`. Both commands returned0;retained result JSONs are in `evidence/`. The verifier executes all5 actual producer dry-runs from each sealed stage,not mutable workspace code. Logs remain in the temporary evidence directory;do not overwrite existing results when rerunning.

Additional original-stage hash-only audit PASS:all3407 D18 stage02 payloads,manifest `42545748def4f40947037a61ca68541d4f3736cd5e8faaa60a8ec3698be91077`;all3407 D22 stage01 payloads,manifest `a1dc28ceca71235b000e7f7448464a5ff76d27e76d2d51ee63fe0811908b5312`. No old receipt/verdict was recomputed. Full source/hash/resource audit:`evidence/source_stage_audit01.json`.

`git diff --check` and staged whitespace check PASS before source commit. See `handoff.md` for exact parent-only submission/judgment commands,budget and remaining gaps. Native H800 timing/full admission remains UNVERIFIED;no serving accuracy or canonical/G3 qualification is claimed.
