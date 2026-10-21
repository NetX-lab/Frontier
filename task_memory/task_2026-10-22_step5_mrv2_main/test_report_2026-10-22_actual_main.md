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

Current status: actual05 is running after the root/owner repair was accepted.
No full request/READ/final-metrics acceptance has been claimed yet.
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
The new clean77 candidate is also running from26db654 against0cdd6cb, both using
the fixed32fb934 driver. Baseline77/77 execution alone is not strict comparison.
