**1. Assignment and outcome**

Implemented the Q5 preparation and CPU post-run tools. The harness changes are uncommitted in `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`, initially clean at `b62c155abd981a71e12e62aa0c0fe3990741d2f8`. The real package is intentionally not built or submitted: Claude reviews and commits the harness, then runs the real builder. Final native admission and ticket 19 acceptance remain with Claude.

The task root is `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`. Writes were limited to the prepared harness worktree, `h200_mainline/build_package.py`, `h3o_operator_costs/formal_p2_64_native_20261102/`, and the private executor scratch directory. No Git mutations, dependency installations, platform/cloud-volume actions, GPU work, submissions, or commits were performed. Read-only Git inspection and the builder's existing Git reads were used. The unrelated `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories remain intact.

Requirements used: D-P2-64-FORMAL-B, D-WARM-TARGET, D-P2-INPUT-QUALIFICATION and the standing resource policy in `requirements.md`; ticket 19 `promotion_32_to_64`, `final_acceptance` and Acceptance; `h200_mainline/design.md`; the formal P2 assessment and warm-reading reports; applicable AGENTS.md. No applicable CLAUDE.md was found. Source text was used to establish implementation behavior, without expanding the assignment.

The new builder case is `p2_64_formal`, with run/job tag `dsv4-h200-p2-64-formal-20261102-01`. It retains P32+D32, TP8/DP4/EP32/PP1 per role, eight nodes of eight H200, `step_main`, and three repetitions of 16 warmups plus 128 formal requests, each 6144 input / 1024 output tokens at QPS 1. The only behavioral changes are warmup pair targeting, the wholly preemptible resource request, and promotion receipt validation. Necessary generation tags, paths and hash bindings also change.

The receipt is bound to its committed task path and SHA-256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`. Its issued time is `2026-11-02T07:38:49.523696+00:00`. The final dry build recorded `2026-11-02T08:07:59.788333+00:00`, strictly later. It does not retrospectively cover `exp-1025-155441-668927`.

**2. Changes and native-source findings**

Every edited source file is listed below. Paths beginning `tests/` in this table are in the prepared harness worktree; the other paths are relative to the task root.

| File | Relevant symbols / lines | Change and reason |
| --- | --- | --- |
| `tests/comparison/calibration/pd_replay.py` | `warmup_pair_targets`, line 406; `replay_repetitions`, line 421; CLI, line 519 | Generates the configured P×D Cartesian product, preserves `intended_pair`, and applies it only when `--pair-warmup` is selected. Fewer warmups than pairs fails clearly. First product cycle is one-to-one; surplus warmups cycle the product. Formal templates containing a target are rejected. |
| `tests/comparison/dp_placement_pp/vllm_replay.py` | `request_headers`, line 496; `replay`, line 511 | Sends warmup-only proxy headers carrying distinct P/D ranks. Formal headers remain Content-Type and X-Request-Id. |
| `tests/comparison/calibration/pd_proxy.py` | `warmup_pair`, line 59; `upstream_headers`, line 72; `build_app`, line 79 | Records the intended pair and forwards separate `X-data-parallel-rank` values to P and D, including the streaming decode path. Targets require an explicit warmup segment and the native v0.28 adapter. Native formal requests receive no target. |
| `tests/comparison/calibration/native_pd_worker.py` | `validate_promotion_receipt`, line 19; `write_run_manifest`, line 38; `run`, line 196 | Checks the receipt's bytes, timezone-aware issuance/build order and true promotion predicate. Retains binding/resource class in the pre-launch manifest and passes the opt-in replay flag. Existing serving and I/O failure handling stays intact. |
| **New** `tests/unit/test_native_pair_warmup.py` | 12 CPU tests | Covers configured products, insufficient warmups, formal-target refusal, three-repeat HTTP forwarding, pinned native header/routing methods, multi-API routing state and negative receipt bindings. |
| `h200_mainline/build_package.py` | case, line 53; `formal_documents`, line 370; `compare_formal`, line 458; `launch_path`, line 693 | Adds the formal case, restages only the four changed harness modules, reconstructs exact allowed package bytes, and rejects all other differences. Inserts receipt validation before staged launch's check-only branch and adds `preemptible=config.get("preemptible", False)` to ResourceConfig. Updates policy/cost fields. Adds the package root to offline launch imports because the staged worker is at that root. |
| **New** `h3o_operator_costs/formal_p2_64_native_20261102/coverage.py` | `ownership`, `idle_evidence`, `analyze`, `safe_analyze` | Derives actual P/D pairs from retained ownership and remote-engine records, binds compatibility cohorts, checks targets and idle boundaries, and emits compatible openers. Missing evidence yields NOT_COVERED and no fabricated empty exclusion map. |
| **New** `h3o_operator_costs/formal_p2_64_native_20261102/compare.py` | `verify_download`, `persist`, `run_pipeline` | Verifies a local cloud copy, prepares an isolated R4 case, checks new-run receipt/preemptible bindings, persists native inputs, and runs unchanged all-request/warm tools against the unchanged formal replay. Errors are recorded as UNVERIFIED; CLI exits zero. |
| **New** `h3o_operator_costs/formal_p2_64_native_20261102/verify_control.py` | `verify`, line 14 | Verifies exact semantic readings and all 12 P2-64 assessment CSV rows, including counts, exclusions, token totals and original/retained endpoints. Only provenance fields may differ. |
| **New** `h3o_operator_costs/formal_p2_64_native_20261102/test_tools.py` | 25 CPU tests | Tests coverage/failure isolation, absent cohorts and configured nodes, exact package mutation rejection, receipt checks before mkdir and actual staged check-only, and the staged ResourceConfig expression with a CPU stand-in. |
| **New** `h3o_operator_costs/formal_p2_64_native_20261102/report.md` | This report | Review handoff and actual verification results. |

All other new output files are evidence under this output directory's `scratch/` and are enumerated, with hashes, in `artifact_inventory.json`. `scratch/control_verified/` contains isolated copies and generated R4/native-input/gate/reading outputs. Earlier scratch artifacts belong to this same executor session; they are not unrelated user work. No pre-existing edits to the writable source files were reported at the session's start. The harness tracked diff is four files, 100 insertions and eight deletions, plus the new test file. The builder tracked diff is 148 insertions and eight deletions. Every edited Python module remains below 2,000 lines.

The final dry candidate differs from the retained package in exactly 12 files: `config.json`, `case/inputs/native_config.json`, `case/manifest.yaml`, `evidence_readiness.json`, `launch.py`, `native_pd_worker.py`, `overlay.json`, `source_lock.json`, `submit_native_diagnostic.py`, and the staged `pd_proxy.py`, `pd_replay.py`, `vllm_replay.py`. Overlay/source-lock bytes change for relocated paths and reviewed harness hashes; their underlying native/C26 implementation remains unchanged. Engine JSON, Frontier case input, model, trace, weights/bootstrap and existing observers remain the retained bytes. Mutation tests reject changes to engine settings, timeout, Frontier input, trace, submission script and bootstrap.

**Header support is confirmed on both roles.** Pinned native source reads the header in `vllm/entrypoints/generate/base/serving.py:206`, extracts it in completion serving at line 155, and passes it into engine generation. `vllm/v1/engine/core_client.py:133` selects DPLBAsyncMPClient for internal DP; line 1473 honors an explicit rank. The package's head commands use internal DP4 with local size one, without external/hybrid LB flags; other lanes are headless. The retained resolved parallel records confirm external/hybrid LB false on both role heads. No alternate route is needed.

**Routing-state limitation for review:** the role heads each start four API processes. Native `vllm/entrypoints/cli/serve.py:103–125` defaults internal-DP API count to the full DP size, even when local DP size is one; retained lane-zero records confirm `_api_process_count=4`. Headless lanes have no API frontends (their config field defaults to one). Each API client has a separate `eng_start_index`, initialized at `core_client.py:1467`; targeted requests skip its increment at line 1516. Sixteen total untargeted warmups do not ensure a zero modulo-four increment in every API process. The new source-method test demonstrates counts `[5,5,3,3]`: after idle, untargeted scan indices are `[1,2,1,2]`, targeted indices `[0,1,2,3]`, and the next untargeted selections differ. This distribution is a CPU example, not an observed distribution from the retained job.

Thus targeting **can change persistent tie-break state seen by formal requests**. Finished outputs release inflight ownership (`core_client.py:1539`), but the idle boundary does not reset API scan indices. Coordinator load snapshots and wave state are traffic-dependent and asynchronous; metrics proving idle do not establish their exact delivery to each API process. Formal routing still uses the unchanged native algorithm, with no target. No API-count change or routing reset was introduced. Claude should review this interpretation explicitly; exact retained formal lane assignments cannot be promised under the allowed warmup change. A requirement for identical post-warmup router state would need a separately agreed native/API design, beyond the current three changes.

NIXL's remote-engine TTL is 3,600 seconds (`base_worker.py:508–509`); eviction requires inactivity greater than TTL (line 2557). The package's KV extra config does not override it. The unchanged replay limit is 1,800 seconds and outer worker timeout is 3,600 seconds. Connections opened after startup should therefore survive the campaign; repeated warmups reuse them. TTL/restarts or missing/extra handshake evidence cause post-run NOT_COVERED rather than an invented proof. Source hashes and retained parallel-record evidence are in `scratch/native_routing_findings.json`.

**3. R3/R4 compatibility and post-run behavior**

No predicate that must fail solely due to the approved changes was found. This is source inspection plus a retained R4 control, not admission of an unrun new campaign.

| Predicate / source | Dependence and finding |
| --- | --- |
| R2 inherited `harness_pinned_and_client_rows`, `check_groundtruth_run_successor_r2.py:830` onward | Requires actual clean harness HEAD/branch and committed/staged file hashes. It does not hard-code b62c155a. Changed modules must be committed and the worktree must remain clean and at that commit through admission. Builder/source lock rebind every staged module. |
| R2 inherited replay/idle checks, lines 982 onward; `native_idle_evidence`, lines 706 onward | Verify the same 432 client identities, exact token/arrival contracts, three repetitions and seven idle boundaries. Extra warmup `intended_pair` fields are allowed; formal records retain the existing contract. The new coverage checker reuses this idle predicate and the pinned dedicated-service proof. |
| R2 raw client/worker predicates | Read required fields and exact request ownership rather than rejecting additional warmup/proxy fields. The unchanged client reader emits only the formal population. Proxy pair fields are additional post-run evidence. |
| R3 `verify_balanced_frontier`, lines 25 onward; balanced module check, line 118 | Uses declared harness revision and actual Frontier input/hashes. The fixed balanced-routing SHA remains `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`. Frontier source/input are unchanged. Full reconstruction at the new committed revision is deferred to the real builder. |
| R2 terminal/platform predicate, lines 1106 onward | Checks launcher binding, Succeeded status, eight replicas, eight GPUs each, image, zero exits and no container restarts. It does not inspect `preemptible_label`. New compare wrapper additionally requires manifest/config preemptible true and platform label `yes`, and matching promotion binding/chronology. |
| R4 source | Hash-binds R3 and replaces only the routing-owner census. Pair headers, warmup fields and resource class do not affect that replacement. No checker was edited. |

`coverage.py` is a CPU post-run soft gate. It reuses `warm_target_20261101/identify_openers.py` and adds both-role request ownership, P-lane/remote-engine consistency, complete configured-node/domain checks and a complete matching between each D lane's first contacts and eight-TP compatibility cohorts. Every warmup's client and proxy intended pair must agree with actual ownership. Formal requests must be untargeted. Both-role idle/freshness samples are rechecked. Compatibility logs have one-second resolution; ambiguous warmup cohorts retain candidate lists and a full matching rather than asserting a unique remote identity unsupported by the logs. Every pair must first appear in a warmup before the formal window; later repetitions must reach every pair again but need not handshake again. Any formal opener is retained in the compatible `--openers` map. Missing/inconsistent evidence gives NOT_COVERED; derivation failure gives `openers: null`, never false empty exclusions. CLI errors are isolated and always return zero.

On retained P2-64, repeat01 warmups opened these 11 `(P,D)` pairs: `(0,2),(0,3),(1,0),(1,1),(1,3),(2,1),(2,2),(2,3),(3,0),(3,1),(3,3)`. Missing pairs were `(0,0),(0,1),(1,2),(2,0),(3,2)`. Exactly five repeat01 formal openers reproduce the retained evidence: r006, r017, r027, r035, r036. There are 128 compatibility lines, 16 eight-TP events, 11 warmup events and five formal events. Repeats02/03 have zero formal openers. P2-32 has 32 compatibility lines, four warmup events, zero formal events, and all four pairs reached in every repetition. Both controls verify all seven idle boundaries. Both correctly report NOT_COVERED under the full new contract because legacy runs did not record intended targets; P2-64 also misses warmup pair coverage. This does not alter their retained historical admission.

`compare.py` rehashes the cloud-copy inventory and all recorded local artifact bytes, permitting only local launcher/terminal receipts as additions. It keeps copied mtimes, runs the unchanged R4 checker in an isolated case and persists 12 files in `inputs/p2_64r/` with a manifest matching the existing diagnostic layout. It overrides a local copy of the A10 case entry, reuses h4a4 `native_layout`, and calls the unchanged normalize/gate/warm-reading tools. The formal replay `/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64` and its command record are read-only and hash-checked. It produces all-request, event-excluded warm, and empty-exclusion control readings. TOOLS_FINISHED describes tool completion, not final acceptance; COVERED/NOT_COVERED, R4 and metric verdicts remain separate.

**4. Executed verification**

Aliases below expand to exact paths; commands were executed from the indicated working directories. Python 3.10.6 is the assignment environment. Its missing aiohttp/torch were not installed. Existing Python 3.12.13 at `openmopd-py312` supplied aiohttp for HTTP tests and torch for the unchanged R4 checker. Source-method tests execute pinned methods with CPU stand-ins; they are not native GPU tests.

```bash
TASK=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800
HARNESS=/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup
TOOLS=$TASK/h3o_operator_costs/formal_p2_64_native_20261102
SCRATCH=/data/ycfeng/tmp/codex-implement/54cbb041-c825-4706-b25e-75530b995d23
REPO=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
PY310=/data/ycfeng/envs/frontier-py310/bin/python
PY312=/data/ycfeng/envs/openmopd-py312/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

| Check | Expected / observed | Result |
| --- | --- | --- |
| Pair/worker tests, command A | Pair routing, receipt refusal, native state and existing worker behavior; **28 passed** | PASS |
| Python 3.10 pair tests, command B | Non-HTTP tests; **11 passed, one skipped** because aiohttp is absent; HTTP executes in A | PASS with stated skip |
| Package/post-run tests, command C | Exact-byte rejection, resource constructor and actual staged launch refusal, coverage and soft errors; **25 passed** | PASS |
| Existing client/trace regressions, command D | Frozen trace and unchanged metric contracts; **27 passed, one skipped** because Python <3.11 lacks asyncio.timeout_at | PASS with stated skip |
| Formal dry builder, command E | Exact package comparison, eight-host plan and issued receipt; **DRY_CHECK_PASS**, receipt PASS, no unapproved harness dependencies | PASS |
| Full P2-64 control, command F | Hash-identical input, R4, 12 persisted native inputs and unchanged gate/readings; **TOOLS_FINISHED, R4 PASS, errors []**, all five subprocesses exit zero | PASS |
| Exact control verifier, command G | Both reading documents equal beyond source provenance; all **12 CSV rows × 26 non-provenance fields exactly equal** | PASS_EXACT |
| P2-32 coverage control, command H | Four warmup-opened pairs and zero formal openers; **evidence_verified true**, idle true, zero formal openers in all repeats | PASS control; NOT_COVERED for absent legacy intentions |
| P2-64 coverage in F and independent retained-opener comparison | Five exact retained repeat01 formal identities, 11 initial warmup pairs, idle true | PASS control; NOT_COVERED as expected |
| Cost recomputation from receipt clocks | Eight durations sum 10,945 seconds; multiply by 8/3600 equals receipt **24.322222222222223 GPUh** | PASS |
| Pinned native source / retained runtime inspection | Both role heads internal DP4, four API processes, rank header honored; TTL 3600 | PASS source/runtime inspection |
| Existing PY312 config probe in HARNESS | Reconstruct retained flat Frontier input with config writes mocked; printed **Total world size: 64** and success marker | PASS environment probe, not R3 admission |
| Real build/seal, preflight run manifest, new-commit R3 reconstruction and real-package check-only | Require Claude's clean committed harness and write scope for the real package | NOT_RUN, explicitly deferred |
| New native run / post-run new-run admission | Claude's submission and completed local copy are not available | NOT_RUN, outside executor authorization |

Commands A–H (and the additional arithmetic/schema probes):

```bash
# A: cwd HARNESS
PYTHONPATH="$HARNESS" timeout 120 "$PY312" -m pytest tests/unit/test_native_pair_warmup.py tests/unit/test_native_pd_worker.py -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_harness_verified"
# B: cwd HARNESS
PYTHONPATH="$HARNESS" timeout 60 "$PY310" -m pytest tests/unit/test_native_pair_warmup.py -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_pair310_verified"
# C: cwd TOOLS
 timeout 120 "$PY310" -m pytest test_tools.py -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_tools_verified"
# D: cwd HARNESS
PYTHONPATH="$HARNESS" FRONTIER_TMP_ROOT="$SCRATCH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 120 "$PY310" -m pytest tests/unit/test_canonical_client_metrics.py tests/unit/test_canonical_fixed_manifest.py -q -p no:cacheprovider --basetemp="$SCRATCH/pytest_client_fixed"
# E: cwd TASK
 timeout 90 "$PY310" h200_mainline/build_package.py p2_64_formal --dry-check "$SCRATCH/formal_dry_final" > "$TOOLS/scratch/dry_final.json"
# F: cwd TOOLS
PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 "$PY310" compare.py --control --r4-python "$PY312" --output scratch/control_verified
# G: cwd TOOLS
PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 60 "$PY310" verify_control.py --control scratch/control_verified --output scratch/control_verified_exact.json
# H: cwd TOOLS
 timeout 60 "$PY310" coverage.py --run /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-canonical-20261025-01/artifacts --output scratch/coverage32_verified.json --openers scratch/openers32_verified.json
# Cost arithmetic: cwd TASK; expected and observed PASS, 10945 seconds / 24.322222222222223 GPUh
 timeout 30 "$PY310" "$TOOLS/scratch/check_cost.py"
# Config/environment probe: cwd HARNESS; writes mocked; expected/observed 64-world reconstruction and PASS marker
PYTHONPATH="$HARNESS" timeout 30 "$PY312" "$TOOLS/scratch/check_config.py"
```

Exact child command arguments, interpreters, working directory, stdout/stderr and return codes for F are recorded in `scratch/control_verified/result.json`. The ordinary all-request gate remains **FAIL** because retained repeat01 TTFT relative error is `0.12254940665008353`; this is expected control reproduction, not a tool failure. Warm repeat01 TTFT error is `0.045998114700243405`; all 12 warm metric checks PASS. Retained counts are 123/128/128, versus unchanged all-request counts 128/128/128. Both warm and empty readings exactly match `formal_p2_20261102/readings/p2_64_{warm,empty}_reading.json` beyond provenance.

Failures encountered and resolved: the first dry plan could not import the root-staged worker; `launch_path` now inserts the package root. Python 3.10 could not import torch for R4; the existing Python 3.12 interpreter completed it, without installation. An initial client/trace regression run failed with `OSError: [Errno 30] Read-only file system` at `/data/ycfeng/tmp/canonical_fixed_*`; rerun D supplied the existing `FRONTIER_TMP_ROOT` override and passed. A config probe from `/data/ycfeng/Frontier` imported that root's different schema and rejected `measured_cc_backend_config_cache_dir`; the same probe from HARNESS reconstructed successfully. An initial runtime assertion incorrectly expected the API-count field to be four on headless nodes too; inspection showed four on heads and default one on headless nodes, and the corrected check passed. The missing-cohort coverage edge case was corrected and regression-tested. Earlier assertion/import errors in synthetic tests were corrected before the final runs. Tests that create fixture Git commits (`test_native_pd_manifest.py` and its canonical-HTTP fixture consumer) were inspected but not executed under the no-Git-mutations instruction. The new HTTP test does not create repositories or commits.

**5. Cost and resources**

The retained platform receipt for `exp-1025-155441-668927` has SHA-256 `634a6e7380909a0992876407a067da1fb96506dd88f6f7f17a12b3094040256a`. Its eight container durations are 1311, 1452, 1314, 1304, 1452, 1473, 1316 and 1323 seconds. They sum to **10,945 container-seconds**. With eight GPUs per container, `10945 * 8 / 3600 = 24.322222222222223 container GPU-hours`; mean container duration is 1368.125 seconds. This includes startup, warmup, serving and shutdown, but is not a claim about queue time or a new campaign's actual charge. Full arithmetic and source binding are in `scratch/cost.json`.

Request: **8 replicas × 8 H200 = 64 H200**, P32+D32, `step_main`, all replicas preemptible. Per-replica CPU/memory remain 64 CPUs and configured 512 GB. The non-preemptible running-plus-queued ceiling is **32**, consistently recorded in manifest policy and readiness. This 64-GPU job is wholly preemptible and outside that non-preemptible ceiling; it is not a mixed resource request.

Timeouts remain outer worker 3600 seconds, startup 1800 seconds, replay 1800 seconds and idle observation 180 seconds. Outer timeout ceiling is **64 GPU-hours per clean generation**. Readiness retains scheduled phase arrival spans totaling 426 seconds, minimum schedule cost 7.573333333333333 GPU-hours, and maximum payload cost 64 GPU-hours, while adding the 24.322222222222223 estimated container cost. The retained campaign from first warmup origin to final token lasted 644.8770877560601 seconds.

Targeting adds **zero requests and zero scheduled arrival seconds**. It moves all lazy pair first contacts into warmup if native execution follows the targets. Warmup handshakes and changed warmup batching/routing can affect phase completion and total elapsed time; no new-run timing was measured, and no exact elapsed-time delta is claimed. A preempted/infrastructure-failed job keeps its evidence and starts a new clean generation, without combining fragments; Claude applies the standing three-job-per-case loop cap and accounts for every attempt.

**6. Commands and review handoff for Claude**

After reviewing and committing the harness, keep that checkout clean and at the recorded commit through launch and R4. Run the real CPU build with the existing Python 3.12 interpreter because the unchanged R3/R4 stack requires torch. This builds the candidate in the real target, compares files, seals, preflights the manifest on scratch, reconstructs with R3, plans all hosts and invokes check-only. These commands were not executed here:

```bash
export PYTHONDONTWRITEBYTECODE=1
cd "$TASK"
timeout 300 "$PY312" h200_mainline/build_package.py p2_64_formal --package package_01 --harness "$HARNESS"
PKG=$TASK/h200_mainline/p2_64_formal/package_01
timeout 60 "$PY312" "$PKG/launch.py" --check-only
# Expected: INPUT_HASHES_PASS_NO_SUBMISSION
```

Claude reviews `p2_64_formal/package_01_build.json` and the sealed manifest, then submits through `/ds4-exec` after the quota/resource read. No submission command was executed or authorization assumed by this executor. The receipt validator runs before any credential read or platform submission, including check-only. Verify the real platform label is `yes`; R4 alone does not establish it.

After the job is terminal, use the unchanged copy and terminal receipt tools through `/ds4-exec` in its established platform Python environment. These are the exact script interfaces; `python3` here means that environment's existing authenticated interpreter, not an instruction to install packages. Replace `ACTUAL_RJOB_NAME` with the actual launcher-reported job. Keep the copy launcher alive through FINAL_STATUS:

```bash
TAG=dsv4-h200-p2-64-formal-20261102-01
COPY=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261102_01
CLOUD_RUN=/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261102_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261102-01
# Through /ds4-exec:
python3 -B "$TASK/h200_mainline/cloud_copy.py" submit "$PKG/config.json" "$COPY" "$CLOUD_RUN"
# Through /ds4-exec after completion, writing the local receipt:
python3 -B "$TASK/h200_mainline/terminal_receipt.py" ACTUAL_RJOB_NAME "$PKG/case/runs/groundtruth_clean/$TAG/platform_terminal_receipt.json"
```

Then run the CPU post-run path locally. `--controller-run` supplies the byte-preserved local launcher/terminal receipts if the cloud tree lacks them. Each output directory must be new:

```bash
cd "$TOOLS"
PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 "$PY310" compare.py \
  --case "$PKG/case" --download "$COPY/$TAG" \
  --controller-run "$PKG/case/runs/groundtruth_clean/$TAG" \
  --r4-python "$PY312" --output "$TOOLS/postrun_01"
# Inspect result.json, coverage.json, openers.json, R4 analysis,
# inputs/manifest.json, gate/reading.json, warm_reading.json, empty_reading.json.
```

Standalone coverage is available when desired, with new output filenames:

```bash
timeout 60 "$PY310" "$TOOLS/coverage.py" --run "$COPY/$TAG/artifacts" \
  --output "$TOOLS/coverage_postrun_01.json" --openers "$TOOLS/openers_postrun_01.json"
```

The unchanged R4 result is written under `postrun_01/r4_case/analysis/run_check_status.groundtruth_clean.json`. The 12 persisted files are under `postrun_01/inputs/p2_64r/`; they intentionally retain the familiar layout name. The wrapper records the exact normalize/gate/warm command sequence. If opener derivation is unverified, it cannot fabricate a warm reading with empty exclusions. Soft-gate exit zero must never be treated as acceptance: inspect result fields and evidence. Final PASS requires ticket 19's qualified inputs, valid pre-launch receipt, correct resource class and all four warm errors within 10% in every admitted repetition; the original all-request reading stays beside it.

Review priorities: inspect warmup-only header handling and actual pair evidence; the multi-API routing-state finding; the builder's exact allowed transformations and receipt refusal before check-only; source-lock/new-commit binding; ResourceConfig forwarding and independent platform label verification; cohort matching and legacy NOT_COVERED interpretation. Rerun A, C, E and the retained control F/G if any relevant code changes. The new-run coverage and R4 checks necessarily await Claude's finished hash-identical copy. There is no unresolved CPU acceptance blocker for the assigned preparation. The remaining material review question is whether unchanged native formal routing, with potentially different warmup-induced scan indices, satisfies the intended routing-state comparison; preserving exact old API state would require an additional agreed design.
