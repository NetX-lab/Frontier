**1. Assignment and outcome**

Implemented Q5 preparation and CPU post-run tools. The harness changes remain uncommitted in `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`, initially clean at `b62c155abd981a71e12e62aa0c0fe3990741d2f8`. Claude must review and commit the harness, then run the real builder. No real package was built or submitted. Final native admission and ticket 19 acceptance remain with Claude.

The complete report is written at `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_64_native_20261102/report.md`. Its SHA-256 is `16a84c8e0ecabe1d5b34e5e66c2fa6426add12ad713e90ed39bbad64066dc2f2`.

The task root is `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`. Writes were confined to the prepared harness, `h200_mainline/build_package.py`, `h3o_operator_costs/formal_p2_64_native_20261102/`, and private executor scratch. No Git mutations, dependency installations, platform/cloud-volume actions, GPU operations, submissions, or commits were performed. Read-only Git inspection and the builder's existing Git reads were used. Unrelated `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories remain intact.

Requirements inspected: D-P2-64-FORMAL-B, D-WARM-TARGET, D-P2-INPUT-QUALIFICATION and the standing resource policy; ticket 19 promotion, final acceptance and Acceptance sections; mainline design; formal P2 assessment and warm-reading reports; applicable AGENTS.md. No applicable CLAUDE.md was found.

The new case is `p2_64_formal`, with run/job tag `dsv4-h200-p2-64-formal-20261102-01`. It retains P32+D32, TP8/DP4/EP32/PP1 per role, eight nodes of eight H200, `step_main`, and three repetitions of 16 warmups plus 128 formal requests, 6144/1024 tokens at QPS 1. Changes are confined to warmup pair targeting, wholly preemptible resources, receipt validation, and necessary tags, paths and provenance bindings.

The receipt is bound to its committed path and SHA-256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`. It was issued at `2026-11-02T07:38:49.523696+00:00`; the final dry build recorded `2026-11-02T08:07:59.788333+00:00`, strictly later. It does not retrospectively cover `exp-1025-155441-668927`.

**2. Changed files and behavior**

Harness-relative paths below are in `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`. Other paths are relative to the task root.

| File | Change and reason |
| --- | --- |
| `tests/comparison/calibration/pd_replay.py` | `warmup_pair_targets` at line 406 derives the configured P×D Cartesian product. Fewer warmups than pairs fails clearly. The first cycle is one-to-one; surplus warmups cycle the product. `replay_repetitions` applies targets only with the new opt-in flag and rejects formal targets. |
| `tests/comparison/dp_placement_pp/vllm_replay.py` | `request_headers` at line 496 sends warmup-only proxy headers with separate P/D ranks. Formal headers remain Content-Type and X-Request-Id. |
| `tests/comparison/calibration/pd_proxy.py` | `warmup_pair`, `upstream_headers` and `build_app` preserve intended pairs and forward distinct `X-data-parallel-rank` values to P and D, including streaming decode. Targets require an explicit warmup segment and native v0.28 adapter. |
| `tests/comparison/calibration/native_pd_worker.py` | `validate_promotion_receipt` at line 19 checks bytes, issuance/build chronology and the promotion predicate. Pre-launch manifest records binding/resource class. Worker passes `--pair-warmup`; serving and I/O failure handling remain unchanged. |
| New `tests/unit/test_native_pair_warmup.py` | Twelve CPU tests cover products, insufficient warmups, formal refusal, three-repeat HTTP forwarding, pinned native methods, multi-API routing state and receipt failures. |
| `h200_mainline/build_package.py` | Adds the formal case and exact allowed-byte reconstruction/comparison. Restages only four changed harness modules. Inserts receipt validation before check-only and forwards preemptible to ResourceConfig. Updates cost/policy fields. Offline imports now include the package root, where its worker is staged. |
| New `h3o_operator_costs/formal_p2_64_native_20261102/coverage.py` | Reads ownership, remote engines and compatibility cohorts, checks actual/intended pairs and idle boundaries, and emits compatible openers. Errors yield NOT_COVERED; unverified derivation never fabricates empty exclusions. |
| New `h3o_operator_costs/formal_p2_64_native_20261102/compare.py` | Verifies local cloud-copy hashes, prepares an isolated R4 case, checks formal receipt/preemptible bindings, persists native inputs, and runs unchanged comparison tools against the unchanged formal replay. CLI errors are isolated and exit zero. |
| New `h3o_operator_costs/formal_p2_64_native_20261102/verify_control.py` | Checks exact semantic readings and all 12 P2-64 assessment CSV rows, including counts, exclusions, tokens and endpoints. Only provenance fields may differ. |
| New `h3o_operator_costs/formal_p2_64_native_20261102/test_tools.py` | Twenty-five CPU tests cover scope rejection, actual staged check-only receipt refusal, ResourceConfig forwarding with a CPU stand-in, coverage and failure isolation. |
| New `h3o_operator_costs/formal_p2_64_native_20261102/report.md` | Complete English report and handoff. |

New scratch evidence, including arithmetic/schema probes and isolated control copies, is enumerated with hashes in `artifact_inventory.json`: 373 output files including the inventory. Earlier scratch files belong to this same executor session. No unrelated pre-existing source changes were encountered. The harness tracked diff is four files, 100 insertions and eight deletions, plus the new test file; builder diff is 148 insertions and eight deletions. Edited Python modules remain below 2,000 lines.

The dry candidate differs from its retained source in exactly 12 files: the two native configs, case manifest, readiness, launch script, staged worker, overlay, source lock, submission script, and staged proxy/replay/client modules. Overlay and source-lock changes bind relocated paths and reviewed harness hashes. Engine JSON, Frontier input, model, trace, bootstrap, C26/native implementation and existing observers retain source bytes. Mutation tests reject other changes.

**Native header support and routing-state finding**

The header is honored on both roles. Pinned `vllm/entrypoints/generate/base/serving.py:206` reads it; completion serving extracts it at line 155 and passes it into generation. `vllm/v1/engine/core_client.py:133` selects DPLBAsyncMPClient for internal DP; line 1473 honors an explicit rank. Role heads use internal DP4, local size one, without external/hybrid LB flags. Other lanes are headless. Retained resolved records confirm these modes. No alternate route is needed.

A material limitation was found: each role head starts **four API processes**, each with its own persistent scan index. Native `vllm/entrypoints/cli/serve.py:103–125` defaults internal-DP API count to full DP size even with local size one; retained head records confirm `_api_process_count=4`. Targeted requests skip the scan-index increment at `core_client.py:1516`. Sixteen total untargeted warmups do not imply modulo-four wraparound in every API process.

The source-method test demonstrates an illustrative per-client distribution `[5,5,3,3]`: after idle, untargeted indices are `[1,2,1,2]`, targeted indices `[0,1,2,3]`, and subsequent untargeted selections can differ. This distribution is a CPU example, not an observed retained distribution. Finished outputs release inflight ownership, but idle metrics do not reset scan indices or establish exact asynchronous coordinator snapshot delivery.

Therefore targeted warmups **can change persistent tie-break state seen by formal requests**. Formal requests still use the unchanged native algorithm and carry no target. No API-count change or state reset was introduced. Claude should explicitly review this interpretation. Exact retained formal lane assignments cannot be promised; preserving identical old API state would require a separately agreed design beyond the current changes.

NIXL remote-engine TTL defaults to 3,600 seconds at `base_worker.py:508–509`; eviction requires inactivity greater than TTL at line 2557. No package override is present. Replay limit remains 1,800 seconds and outer worker timeout 3,600 seconds. Connections opened after startup should survive the campaign. Source hashes and retained runtime evidence are in `scratch/native_routing_findings.json`.

**3. R3/R4 compatibility and post-run controls**

No predicate that must fail solely because of the approved changes was found. This conclusion combines source inspection and retained R4 control; it does not admit an unrun new campaign.

- Inherited R2 harness checks at line 830 onward require actual clean HEAD/branch and committed/staged hashes. They do not hard-code b62c155a. The changed harness must be committed and remain clean through admission.
- Replay/idle checks at line 982 onward retain the 432 identities, exact workload and three repetitions/seven boundaries. Extra warmup fields do not replace existing required fields. The new coverage checker reuses the unchanged idle predicate and dedicated-service proof.
- Raw client/worker checks validate required fields and ownership. The unchanged client reader emits formal rows; proxy intentions are additional evidence.
- R3 reconstructs using the declared harness revision and actual Frontier input/hashes. Its fixed balanced-routing hash remains `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`. Frontier source/input are unchanged. New-commit reconstruction is deferred to the real builder.
- R2 platform checks at line 1106 onward validate launcher binding, successful replicas, GPU counts, image, zero exits and no restarts. They do **not** inspect `preemptible_label`. The new wrapper additionally requires config/manifest preemptible true, platform label `yes`, and matching receipt binding/chronology.
- R4 hash-binds R3 and replaces only the routing-owner census. No checker was edited.

Coverage reuses the retained event-binding method and adds both-role ownership, remote-engine consistency, complete configured-node/domain checks, full eight-TP cohort matching and intended-versus-actual pair checks. Compatibility logs have one-second resolution; ambiguous warmup cohorts retain candidates and a complete matching rather than invented unique bindings. Later repetitions reach all pairs but need not handshake again. Any formal opener remains in the compatible exclusion map. Missing evidence reports NOT_COVERED; failed derivation produces `openers: null`. CLI exits zero.

Retained P2-64 repeat01 warmups opened 11 pairs: `(0,2),(0,3),(1,0),(1,1),(1,3),(2,1),(2,2),(2,3),(3,0),(3,1),(3,3)`. Missing pairs were `(0,0),(0,1),(1,2),(2,0),(3,2)`. Exactly five formal openers reproduce retained evidence: r006, r017, r027, r035 and r036 in repeat01. Counts are 128 compatibility lines, 16 cohorts, 11 warmup events and five formal events. Repeats02/03 have zero formal openers.

Retained P2-32 has 32 compatibility lines, four warmup events, zero formal events and all four pairs reached in every repetition. Both controls verify all seven idle boundaries. Both report NOT_COVERED under the full new contract because legacy intentions are absent; P2-64 also misses warmup coverage. Historical admission is not altered.

The comparison wrapper rehashes the inventory and artifact bytes, preserves copied mtimes, runs unchanged R4 in an isolated case, persists 12 native-input files in the familiar `inputs/p2_64r/` layout, and reuses A10/h4a4 case construction and unchanged normalize/gate/warm tools. The formal replay and command record remain read-only and hash-checked. TOOLS_FINISHED describes tool completion, not acceptance.

**4. Verification: commands and observed results**

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

Python 3.10.6 is the assignment environment. Missing aiohttp/torch were not installed. Existing Python 3.12.13 supplied aiohttp for HTTP tests and torch for unchanged R4. Pinned-source tests use CPU stand-ins, not GPU execution.

| Check | Expected and observed | Result |
| --- | --- | --- |
| A: harness and worker tests | Pair forwarding, receipt refusal, native state and existing worker behavior; 28 passed | PASS |
| B: Python 3.10 pair tests | Non-HTTP tests; 11 passed, one aiohttp skip; HTTP executes in A | PASS with skip |
| C: package/post-run tests | Exact scope rejection, actual staged launch refusal, resources and soft coverage; 25 passed | PASS |
| D: existing client/trace regressions | Frozen trace and metric contracts; 27 passed, one skip for Python <3.11 asyncio.timeout_at | PASS with skip |
| E: formal dry builder | Exact file comparison, eight-host plan, receipt checks; DRY_CHECK_PASS | PASS |
| F: full P2-64 control | Hash-identical download, R4, 12 persisted inputs and readings; TOOLS_FINISHED, R4 PASS, errors []; five subprocesses exit zero | PASS |
| G: exact control verifier | Both readings equal beyond source provenance; 12 CSV rows × 26 fields exactly equal | PASS_EXACT |
| H: P2-32 coverage | Four warmup pairs, zero formal openers, idle verified in all repeats | PASS control; expected legacy NOT_COVERED |
| P2-64 coverage and opener comparison | Five exact repeat01 openers, 11 initial pairs, idle verified | PASS control; expected NOT_COVERED |
| Cost clocks | Eight durations total 10,945 seconds; 24.322222222222223 GPUh equals receipt | PASS |
| Native source/runtime inspection | Both heads internal DP4 with four API processes; rank header support; TTL 3600 | PASS inspection |
| PY312 schema probe | Retained flat input reconstructs, writes mocked, world size 64 | PASS probe; not R3 admission |
| Real build/seal, manifest preflight, new-commit R3 and real check-only | Require Claude's committed harness and real-package write scope | NOT_RUN, deferred as assigned |
| New native run and its post-run admission | No completed new campaign exists | NOT_RUN, outside executor authorization |

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
# Arithmetic probe: cwd TASK
timeout 30 "$PY310" "$TOOLS/scratch/check_cost.py"
# Schema probe: cwd HARNESS; config writes mocked
PYTHONPATH="$HARNESS" timeout 30 "$PY312" "$TOOLS/scratch/check_config.py"
```

Exact child argv, interpreters, cwd, stdout/stderr and exits for F are in `scratch/control_verified/result.json`.

The retained all-request gate remains FAIL because repeat01 TTFT error is `0.12254940665008353`. This is expected reproduction, not tool failure. Warm repeat01 TTFT error is `0.045998114700243405`; all 12 warm checks PASS. Warm counts are 123/128/128, versus unchanged all-request 128/128/128. Warm and empty readings match the retained formal readings beyond provenance; CSV counts, exclusions, tokens and endpoints match exactly.

Resolved failures: offline dry import initially missed the root-staged worker; package-root insertion fixed it. Python 3.10 R4 failed on missing torch; existing Python 3.12 passed. An existing regression initially attempted `/data/ycfeng/tmp/canonical_fixed_*` and hit read-only filesystem; FRONTIER_TMP_ROOT rerun passed. A probe from the main repository imported its different schema and rejected `measured_cc_backend_config_cache_dir`; running from the harness reconstructed successfully. Runtime inspection initially expected API count four on headless nodes as well; corrected inspection confirmed four on heads and default one on headless nodes. Missing-cohort coverage was fixed and regression-tested. Earlier synthetic assertion/import errors were corrected before final runs.

Tests that create fixture Git commits were inspected but not executed. The new HTTP test creates no repositories or commits. No checker, retained package, trace, replay, receipt, Q1 ledger, Frontier or data source was edited.

**5. Cost and resources**

Retained platform receipt SHA-256: `634a6e7380909a0992876407a067da1fb96506dd88f6f7f17a12b3094040256a`. Container durations are 1311, 1452, 1314, 1304, 1452, 1473, 1316 and 1323 seconds, totaling 10,945 container-seconds. At eight GPUs each, `10945 × 8 / 3600 = 24.322222222222223 container GPU-hours`; mean container duration is 1368.125 seconds. This includes initialization, warmup, serving and shutdown, and does not claim queue/allocation cost or actual cost of the new run. Evidence is in `scratch/cost.json`.

Resource request: eight replicas × eight H200, P32+D32, `step_main`, wholly preemptible; existing per-replica 64 CPUs and configured 512 GB. Standing non-preemptible ceiling is 32, consistently recorded in policy/readiness. This is not a mixed resource request.

Timeouts remain outer worker 3600 seconds, startup 1800, replay 1800 and idle observation 180. Ceiling is 64 GPU-hours per clean generation. Readiness retains scheduled arrival spans totaling 426 seconds, minimum schedule cost 7.573333333333333 GPUh and maximum payload cost 64 GPUh, while recording estimated container cost 24.322222222222223 GPUh. Retained first warmup origin to final token duration was 644.8770877560601 seconds.

Targeting adds zero requests and zero scheduled arrival seconds. It moves lazy contacts into warmup if targets are honored. Handshake placement and changed warmup routing/batching can affect elapsed completion; no exact new-run duration delta was measured or promised. Interrupted runs retain evidence and restart as new clean generations; fragments cannot form a reference. Claude applies the three-job loop cap and accounts for every attempt.

**6. Remaining work and review handoff**

After review and commit, keep the harness clean and at its recorded revision through launch and admission. Real build requires an existing torch-capable interpreter; these commands are deferred, not executed:

```bash
export PYTHONDONTWRITEBYTECODE=1
cd "$TASK"
timeout 300 "$PY312" h200_mainline/build_package.py p2_64_formal --package package_01 --harness "$HARNESS"
PKG=$TASK/h200_mainline/p2_64_formal/package_01
timeout 60 "$PY312" "$PKG/launch.py" --check-only
# Expected: INPUT_HASHES_PASS_NO_SUBMISSION
```

The builder compares files, seals, preflights the manifest, reconstructs through R3, plans all hosts and invokes check-only. Claude reviews `package_01_build.json` and the seal, then uses `/ds4-exec` after the quota/resource read. Receipt validation occurs before credential reads or submission. Confirm platform preemptible label `yes`; R4 alone does not establish it.

After terminal status, use unchanged copy/receipt scripts through `/ds4-exec` in its established authenticated Python environment. Replace ACTUAL_RJOB_NAME with the actual launcher-reported name. Keep the copy launcher alive through FINAL_STATUS:

```bash
TAG=dsv4-h200-p2-64-formal-20261102-01
COPY=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261102_01
CLOUD_RUN=/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261102_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261102-01
# Through /ds4-exec:
python3 -B "$TASK/h200_mainline/cloud_copy.py" submit "$PKG/config.json" "$COPY" "$CLOUD_RUN"
python3 -B "$TASK/h200_mainline/terminal_receipt.py" ACTUAL_RJOB_NAME "$PKG/case/runs/groundtruth_clean/$TAG/platform_terminal_receipt.json"
```

Then run locally with a new output directory:

```bash
cd "$TOOLS"
PYTHONPATH="$REPO" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 "$PY310" compare.py \
  --case "$PKG/case" --download "$COPY/$TAG" \
  --controller-run "$PKG/case/runs/groundtruth_clean/$TAG" \
  --r4-python "$PY312" --output "$TOOLS/postrun_01"
```

Optional standalone coverage:

```bash
timeout 60 "$PY310" "$TOOLS/coverage.py" --run "$COPY/$TAG/artifacts" \
  --output "$TOOLS/coverage_postrun_01.json" --openers "$TOOLS/openers_postrun_01.json"
```

Inspect result and coverage fields, openers, `postrun_01/r4_case/analysis/run_check_status.groundtruth_clean.json`, persisted inputs, `gate/reading.json`, warm reading and empty reading. Soft-gate exit zero is not acceptance. Final PASS requires qualified inputs, pre-launch receipt, correct resource class and all four warm errors within 10% in every admitted repetition, retaining the original all-request result.

Review priorities: warmup-only headers; actual/intended pair binding; the four-API persistent-index finding; exact builder transformations and receipt refusal before check-only; new-commit source-lock binding; ResourceConfig and platform label checks; cohort matching and legacy NOT_COVERED interpretation. Rerun A, C, E and F/G after relevant changes.

No unresolved CPU blocker remains for the assigned preparation. Real build/submission and new-run checks await Claude as assigned. The material review question is whether unchanged native formal routing, with potentially different warmup-induced scan indices, satisfies the intended routing-state comparison. Identical old API state would require an additional agreed design.
