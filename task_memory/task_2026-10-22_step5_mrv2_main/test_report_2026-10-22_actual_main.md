# D23-ACTUAL-MAIN CPU Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record fixture tests and exact main acceptance conditions. |
| 2026-10-22 | Retain successful fit and record actual CLI prerequisite failures. |
| 2026-10-23 | Record actual layer3 failure and exact compute refresh. |
| 2026-10-22 | Record paired actual04 core rejection and demonstrated blocker. |
| 2026-10-22 | Primary passes8 fixture checks, verifies merged bytes and reproduces the root ledger failure. |
| 2026-10-22 | Parent167 PASS and accepted root/owner repair; start actual05 at26db654 with8 unchanged input hashes. |
| 2026-10-22 | Root/owner clean comparison77/77 identical; zero cache/provenance findings. |
| 2026-10-22 | Actual05 fails by2400s timeout after partial decode; retain phase timings and missing-artifact evidence. |
| 2026-10-22 | Parent repeats real-cache CPU RED and collective/trace controls; source correction remains in progress. |
| 2026-10-22 | Accept fixed-contract source with parent timings, exact costs/artifacts and 255 passing checks plus two baseline-identical failures. |
| 2026-10-22 | Accept loader reuse and merged224 checks; launch actual06 on frozen7e8bea3 with unchanged inputs and execution bound. |
| 2026-10-22 | Integrated CPU reuse strict77 passes; actual06 full request remains pending. |

## Environment and scope

Base:d8e3e941; tested core:`c9e9388f117c625c6d546d1d083f8aaf3f824f51`.
Worktree:`/data/ycfeng/Frontier/.worktrees/step5-mrv2-main`.
Python:`/data/ycfeng/envs/frontier-py310/bin/python`3.10.6; numpy2.2.6,
pandas2.3.3, sklearn1.7.2. Torch not installed; CPU DES does not need it.
CPU only, no platform/auth/GPU/cloud work. A default dependency init failed at
the local proxy; successful initialization used only the pinned local checkout.
All generated durations/counts and3.5ms READ cost are synthetic CPU fixtures,
not real measurements, native-admitted profiles or accuracy evidence.

## Scoped regression

Exact command is recorded in `progress.md`.
`tests/integration/test_step5_measured_main.py`:4 PASS in1.64s.
Logs/XML:`/data/ycfeng/tmp/d23_main_fixture_unit.{log,xml}`.
Final and corrected-source fixture tests:6 PASS, final1.83s.
Logs/XML:`/data/ycfeng/tmp/d23_main_fixture_final_unit.{log,xml}` and
`/data/ycfeng/tmp/d23_main_correction_unit.{log,xml}`.
Complete row equality checks compare template context reuse against independently
built workloads on both variants at6145/6500/7166. Generated CUDA_EVENT94392
and KERNEL_ONLY184 rows retain exact scope/kind identity and1023 contexts.
Environment receipt:`/data/ycfeng/tmp/d23_main_environment.json`.

## Fit and input evidence

Original generation02 completed in1068s; no duplicate generation/fit was launched.
Manifest:`/data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json`.
SHA256:`0ed8dd433a49e324804cb50980055e010c3af90f5c9b0cf1220490b940a3fbe3`.
CUDA_EVENT94392 and KERNEL_ONLY184 typed rows;92 layers;1023 real contexts from
6144 through7166 inclusive. Dummy rows are metadata-bearing MRv2, never isolated.
Cache144 files/161346978 bytes at observed snapshot;4 existing bundle families.
Selected receipts:`fixture02_bundle_identity.json`, `actual03_config_receipt.json`.
Exact CSV/count/kind receipts:`/data/ycfeng/tmp/d23_main_fixture02_{rows,kernel_rows}.json`.

Local dependency gitlink:`d28fe917370c2d55230c3ea226e5660785b44aca`.
Local htsim binary SHA256:
`812fb3d86489d76a4d50e2f0ea936bdb6d25dfe740c06050a0d0c1a1f9f4631e`.

## Primary fixture acceptance and core reproduction

Primary command, cwd=`/data/ycfeng/Frontier/.worktrees/step5-mrv2-main`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_step5_measured_main.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_main_parent_fixture01.xml > /data/ycfeng/tmp/d23_main_parent_fixture01.log 2>&1
```

Observed8PASS/6.45s, exit0, Python3.10.6. Primary merged at `8e1d0a1` and
byte-compared all three integration files to `e54f9457`; all match.
Primary reran the exact minimal command from [core_blocker.md](core_blocker.md)
in `step5-preview`, obtaining exit1 and the identical KeyError at4504;
`/data/ycfeng/tmp/d23_ledger_parent_red01.log` retains the stack. Thus fixture
acceptance is PASS while actual-main acceptance remains FAILED/BLOCKED.

## Actual subprocess acceptance

Required for **each** offline and online subprocess:
- Actual `python -B -m frontier.main`, no simulator patching; full6144/1024 completion.
- P16/D16 TP8/DP2/EP16/PP1,7168/8192/128,0.92 utilization, no prefix/chunk/spec.
- FULL_AND_PIECEWISE,capture1/2/4/8..256, correctly bound real/dummy first calls.
- P6144, one D local-tail16 using cached6128,1023 subsequent D query1 forwards.
- NIXL reserved target READY and source release at completed3.5ms READ; D after READY.
- Finite positive E2E/client-TTFT/TPOT and stage costs, exact transfer payload.
- No loosened workload/capacity, defaults, scaling or native qualification claims.

## Actual attempts

| Attempt | Offline | Online | Cause/evidence |
| --- | --- | --- | --- |
| actual01 | FAIL before events | FAIL before events | Mutually exclusive PD-AF global graph flag in runner; corrected to explicit no-use_cuda_graph while keeping native PD graph policy |
| actual02 | FAIL1.366s | FAIL1.366s | Uninitialized collective-sim gitlink; ModuleNotFoundError before events; pinned local dependency restored/built |
| actual03 | FAIL1141.94s, exit1 | FAIL1160.34s, exit1 | First actualP layer3 EP wave rejects share_expert_up_proj num_tokens6145 above old fixture maximum6144; no timeout |
| actual04 | FAIL1032.6621s, exit1 | FAIL1032.4320s, exit1 | P-stage completion ledger KeyError: step5_layer_e2e; no timeout |

All exact commands are in `progress.md` and per-case `command.{json,bash}`.
Logs:`/data/ycfeng/tmp/d23_main_actual03_{offline,online}/<mode>/run.log`.
Input hashes:`/data/ycfeng/tmp/d23_main_actual03_{offline,online}/input_hashes.json`.
READ cost SHA256:
`3975d5ec0bf247d0c961bd74af47bd9411de6db59abe37a6251f8e0575d101fa`.

Current status: actual05 timed out in both modes after the root/owner repair was
accepted. No full request/READ/final-metrics acceptance has been claimed.
A zero returncode or initialized config does not establish completion. On timeout,
preserve whether setup or DES was active; do not silently expand2400s.

## Exact compute-source correction

Actual03 reaches real `prefill_sync_event.py -> sync_entry.py ->
ep_wave_schedule.py:101 -> expert_parallel.py:117` for layer3. The existing
native-bound NONE aggregate preserves local useful6144 and dummy1, so its
shared-expert input is6145. Both modes reject this missing coverage with the
same exact exception; costs/defaults/capacity are not changed to bypass it.

The fixture now uses local widths1/16/6144 and aggregate2/32/6145. Each linear
CSV has24 rows. Each routed CSV has12 distinct exact nonempty lane rows from
the existing balanced Hamilton materializer over352 experts, EP16, topk8,
separately for3-87 FP8 and88-90 BF16 typed contracts. Source tokens stay2/32/6145;
the rows retain actual22-expert local counts. Zero-load lanes keep the existing
zero-load execution skip. Kernel count7 is explicitly synthetic per replay.

Refresh command (existing fixture only, no duplicate attention generation/fit):
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/step5_measured_compute_fixture.py --output-dir /data/ycfeng/tmp/d23_main_fixture02 --refresh-compute > /data/ycfeng/tmp/d23_main_compute_refresh.log 2>&1
```
Four original compute CSVs retained as `*.csv.before_mrv2_compute`. Existing
attention hashes are unchanged from actual03:
- CUDA_EVENT94392 rows: `27baaf5c63315a41e5a78096aa318d50c4c134cae5c338abd8a940cae61b7179`.
- KERNEL_ONLY184 rows: `b00a44e2e62efde9313f3e38f04d9d320558d78d15dbae13be747926c7c7dd81`.

Selected old/new input hashes and row counts:`compute_refresh_receipt.json`.
Refresh manifest SHA256:`417492f6223f83bc702b815cd8a4280323afb60600fd68fe9b9636cc0077fffd`.
Original fixture bundle receipts remain in `fixture02_bundle_identity.json`;
manifest marks them stale until actual CLI regenerates affected compute bundles.

Scoped regression7 PASS in2.34s; exact command:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_step5_measured_main.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_main_compute_green.xml > /data/ycfeng/tmp/d23_main_compute_green.log 2>&1
```
The new regression compares complete exact feature keys against actual
`materialize_layer_ep_workload` output for2/32/6145 and both precision groups;
CSV round-trip preserves float values. The initialRED log is a missing fixture
function; actual03 supplies the decisive pre-fix runtimeRED evidence.

Actual04 exact commands:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual04_offline --mode offline > /data/ycfeng/tmp/d23_main_actual04_offline.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual04_online --mode online > /data/ycfeng/tmp/d23_main_actual04_online.log 2>&1
```

Final scoped8 PASS in5.84s:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_step5_measured_main.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_main_exact_compute_replay.xml > /data/ycfeng/tmp/d23_main_exact_compute_replay.log 2>&1
```
The final test fits the existing typed FFN cache, invokes existing Step FFN
duration/count lookup for exact balanced EP lane features at2/32/6145 and both
precision groups, preserves zero-load skip, and verifies missing7168 coverage
rejects. No generated timing or domain changes follow from this test. Actual04
runner/fixture hashes remain unchanged while children execute.

## Actual04 observed output and limits

Both exact subprocesses ended within2400s on testedc9e9388f. Last stack:
`dense_layer_complete_event.py:50 -> prefill_collective.py:250 ->
metrics_store.py:4504`, `KeyError: 'step5_layer_e2e'`. Nine saved P scheduler
records/mode, one useful6144-token batch, no D scheduler records. Last scheduler
simulation times85.20541516799993s offline/85.45541516799992s online. These are
not final request E2E metrics. All terminal request/system/event/transfer/stage
artifact files are absent; READ,1024 useful D forwards and final positive finite
metrics remain UNVERIFIED. Full6144/1024 criterion is not passed.

`actual04_failure_receipt.json` contains execution values, commands/config/log
SHA256 and sizes, exact useful scheduler record and absent-artifact checks.
`actual04_root_source_receipt.json` retains existing exact cache source keys,
kind-bound physical provenance, duration.6244ms/count23 and cache SHA256. The
actual event cost object was not saved, so this is labelled cache source evidence.

`core_blocker.md` contains an executable isolated reproduction and smallest
source request: preserve scalar-free inclusive root once in existing metrics
and trace projection, with no diagnostic-child double counting. Layer and stage
ledger KeyError both reproduced; direct structural projection already exposes
the subsequent root-rejection ValueError. Reproduction log:
`/data/ycfeng/tmp/d23_main_ledger_repro.log`. No core edits, ledger disablement,
cost substitutes or duplicate fit/runs are used to bypass this defect.

Both actual04 logs emit mixed-precision generic metadata warnings at
`quantization_manager.py:810`. This was unresolved at the original handoff.
Parent subsequently confirmed by source and factor-injection controls that
the generic0.5 factor does not rescale this typed Step path; see
`quantization_warning.md` and `precision_factor_receipt.json`. No accuracy claim.

Implementation commit:`e54f9457aed7e31336cb6cb553aec016a737ae2a`; source receipt
proves runner/fixture match actual04 hashes after commit. Tested production core
remainsc9e9388f; no core edits are part of this integration slice.

## Parent actual05 launch

Stable source: `26db654cbbb38b361990aa1aa8ae1dad5db8e000` in the existing
`step5-mrv2-main` worktree. Parent verified that all production source and changed
tests match reviewed7c70461. Root/owner focused verification:167 PASS,1 explicitly
inherited deselection,46.71s; independent final review No findings. Evidence:
`root_metrics_review.md` and `/data/ycfeng/tmp/d23_root_p1_parent01.{log,xml}`.

Preflight compared all8 retained fixture/profile hashes against actual04 and
verified all three integration files match e54f9457. Receipt:
`/data/ycfeng/tmp/d23_main_actual05_input_preflight.json`. Manifest SHA256 remains
`417492f6223f83bc702b815cd8a4280323afb60600fd68fe9b9636cc0077fffd`.
The first read-only helper attempt used Python3.11's `hashlib.file_digest` on
Python3.10 and failed before hashing; the corrected streaming-hash check passed.
No fixture regeneration or separate fit was launched.

Exact commands, cwd=`/data/ycfeng/Frontier/.worktrees/step5-mrv2-main`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual05_offline --mode offline > /data/ycfeng/tmp/d23_main_actual05_offline.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d23_main_fixture02/fixture_manifest.json --output-root /data/ycfeng/tmp/d23_main_actual05_online --mode online > /data/ycfeng/tmp/d23_main_actual05_online.log 2>&1
```

Both commands run independently, each with the existing2400s child timeout.
Expected acceptance remains the complete list above; no terminal result yet.
The new clean77 candidate26db654 and baseline0cdd6cb, both using the fixed32fb934
driver, completed and passed strict comparison:77/77 identical; zero failures,
missing evidence, differing definitions or predictor cache/provenance findings.
Full report `root_metrics_fidelity.json` and exact commands in
`root_metrics_review.md`. Actual05 acceptance failed by timeout, as below.

## Actual05 result: TIMEOUT, not accepted

The exact commands above ran on unchanged26db654 and reached the existing2400s
child bound. Neither log contains the previous ledger exception. The following
values were recomputed from each scheduler JSONL, execution receipt and command
file timestamp; files are only1.6MiB and were streamed with a60s inspection limit.

| Observation | Offline | Online |
| --- | ---: | ---: |
| Child wall time (s) | 2400.130515 | 2400.111418 |
| Useful P batches scheduled | 1 | 1 |
| Useful D batches scheduled / batch-end records | 43 / 42 | 42 / 41 |
| Highest observed emitted output tokens | 42 | 41 |
| Command written to first scheduler event (s) | 998.503711 | 1004.875261 |
| Median useful D iteration wall gap (s) | 32.486104 | 36.116395 |
| Minimum / maximum wall gap (s) | 18.613046 / 209.662127 | 18.518195 / 209.332491 |
| Last recorded simulation time (s) | 87.717413 | 87.924543 |

The first useful D schedule is16 tokens; later ones are1 token. Both simulations
advance through the prior P-stage ledger failure and partial decode, so complete
event-loop stagnation is excluded by the observed interval. The wall-gap values
include all work between useful scheduling events; they are not per-function CPU
profiles or native GPU token latency. The startup interval is anchored at the
command-file write, not an instrumented initializer boundary.

Both receipts have `timed_out=true` and null child returncode; wrapper exit1
reports that timeout. `system_metrics.json`, `request_metrics.csv`,
`event_trace.json`, `frontier_transfer_ledger.jsonl` and
`frontier_stage_batch_ledger.jsonl` were not written. Thus complete1024-token
execution, final READ/readiness/release proof and final metrics remain unverified.
No partial counter is substituted for accepted terminal metrics.

Full census, event counts and command/log/scheduler/receipt hashes are retained
in `actual05_timeout_receipt.json`. CPU profiling/minimization is now assigned
to `D23-ACTUAL-MAIN/PERF`; the execution bound, inputs and modeled costs remain
unchanged. No additional full attempt has been launched.

Parent has now rerun the delivered short feedback loop and controls. It confirms
the repeated fixed metadata/contract cost at the real query boundary while exact
source values remain intact. See `cpu_performance_review.md` for exact commands,
timings, source review scope and the independent initializer diagnosis. These
diagnostics do not change the failed full-workload verdict.

## Fixed-contract correction: parent acceptance

Target `ee28bf7`, records `7834dd5`, cwd
`/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`. Python3.10.6. Exact command:

```bash
env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_interfaces.py tests/unit/test_on_demand_prediction_contract.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_measured_attention_root_metrics.py tests/unit/test_metrics_stage_execution_time.py tests/unit/test_step5_mrv2_forward.py tests/unit/test_step5_structural_execution.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_contract_parent01.xml > /data/ycfeng/tmp/d23_contract_parent01.log 2>&1
```

Actual: 255 PASS, 2 FAIL, zero errors/skips, 108.370s, exit1. Before merging the
optimization, the parent also ran the following from `step5-preview`:

```bash
env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_structural_execution.py -k first_serving_event_includes_input_scope_and_preserves_launch_start -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_contract_baseline_structural01.xml > /data/ycfeng/tmp/d23_contract_baseline_structural01.log 2>&1
```

Baseline: the same two failures, 2.014s. XML comparison verifies identical test
IDs and `AttributeError: 'types.SimpleNamespace' object has no attribute
'take_forward_start_batch'` messages. No new failures were observed; the full
selected suite is not all-pass. The original fixture failure remains recorded.

The unchanged short feedback loop reports 0.532956s -> 0.073913s for 64 real lane
queries (7.210579x). Parent comparison confirms all corresponding cost dictionaries
and artifact descriptors exactly equal; trace persistence passes for 456 events.
The original maximum-query gate remains RED. Exact benchmark/loader commands and
source-review findings are in `cpu_performance_review.md`; the selected comparison
is retained in `contract_parent_receipt.json`. Independent review accepts this
source phase with No findings. Loader reuse has since been accepted as detailed
below; merged strict77 and original6144/1024 terminal acceptance remain pending.

## Actual06 launch after CPU reuse corrections

Source `7e8bea3` combines both independently reviewed corrections. Parent
verification:187 loader/bundle/manager checks,19 CPU torch-helper checks,224
combined checks and8 frozen-checkout fixture checks pass. These overlapping
suites are reported separately. The original input probe is GREEN; baseline
frame/attrs/JSON and four root/count hashes match exactly. Commands and limits
are in `test_report_2026-10-22_cpu_reuse.md`, with selected evidence in
`load_parent_receipt.json`.

The full offline/online commands are recorded in `progress.md` under Actual06.
Both run in `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-7e8bea3`, whose
HEAD remains fixed. Each generated input-hash receipt matches actual05 at all8
shared fixture/source paths. No fixture regeneration, timeout change or weakened
terminal validation is used. The retained clean26db654 matrix is the baseline
for the new77-case candidate run. That run and its strict comparison have now
passed77/77, including predictor cache filenames and provenance. See
`cpu_reuse_fidelity.json`. Both full CLI terminal results remain pending;
no full-request or native/accuracy acceptance is claimed.
