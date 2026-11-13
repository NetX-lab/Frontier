# Post-key frozen full-unit and exact retained79 status sets

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Close canonical8020/279/32/281 with exact79 sets,complete environment joins and one declared old parser-ID loss. |
| 2026-10-24 | Record exact active command, verified baseline and unchanged acceptance predicates. |
| 2026-10-24 | Preserve first8036/279/32/267 raw run and restore original no-binary namespace for canonical rerun. |

TaskID D29-POST-KEY-FROZEN-REGRESSIONS. Source46fe83ac55876c10f6dd8ff792705e87efd1905f,
treebbbca98802261662e02d32a59c1a4d6ab20a12e5,submodule
3dac53c544e546574cb6e8078cd14053d0916113. Cwd
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-07`.

## Exact command and runtime environment

```bash
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/run_suite.py unit
```

Payload,unchanged test selection and retained1800s cap:
```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent06 --junitxml=/data/ycfeng/tmp/d29_unit_parent06.xml
```

Full prefix/environment/versions captured in
`/data/ycfeng/tmp/d29_post_key_regressions_20261024/unit/launch.json` and,
after completion,execution.json. Tests use `sudo -n systemd-run --scope --quiet
-p MemoryMax=30G runuser -u brainpp -- env ... bwrap ...`. Actual preflight
UID10250 and v1 memory.limit_in_bytes32212254720 verified;systemctl MemoryMax
matches. Host uses a container-relative v1 memory mount plus a unified hierarchy;
initial inspector paths failed without indicating test/resource failure.

Python3.10.6,pytest9.1.1,numpy2.2.6,pandas2.3.3,scipy1.15.3,
sklearn1.7.2,plotly7.1.0,fasteners0.20,ddsketch3.0.1;Torch/aiohttp absent.
PYTHONPATH exactly repo:repo/tests/unit,PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp,OMP/OPENBLAS/MKL threads1;
FRONTIER_TEST_CPU_WRITER_FILE/SHA256 unset. Existing tracked source entries remain
read-only. Unique scratch root/cache/logs/JUnit only. The old3dac ignored htsim
binary is projected read-only from retained integrated06,not built/copied into
the frozen source;its hash is guarded. No source/test/config/helper edits.

## Verified retained evidence and exact gates

Retained79 raw XML `/data/ycfeng/tmp/d29_unit_parent05.xml` SHA256
`cb64bf1bfbba142b5ad7b3fbad90343d1f9134e6a627c35af5017bfc3654a4d8`:
8610 unique IDs,8018PASS/279FAIL/32ERROR/281SKIP,rawexit1. This is the required
current baseline,not an inferred expectation for new counts. Original267FAIL/
31ERROR locked IDs and old latest overlay are also checked by the unchanged
retained recipe. Baseline/source input hashes captured before fresh launch.

Retained XML parser/transitions are reused unchanged. The existing failure/
environment recipe is executed with only new source/path/label bindings;its raw
verdict remains visible. Additional79 comparison enumerates every retained ID,
every actual status change,case loss,new case,known FAIL/ERROR/SKIP and exact
failure/error additions/removals. Raw status-set gate fails on any missing79 ID
or changed retained status. No source-name alias or exception is inserted.

The approved source replaces one parser hot-update test name with a new run-fixed
input test. This is a documented source change,not a test executed under its old
ID. All new actual IDs must pass. The already-completed metadata selection has
confirmed1old-IDloss/1newPASS;full-unit must independently determine all new IDs
and counts. A planned replacement is never represented as zero raw case loss.

After automatic completion:
```bash
sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/compare_results.py --phase before_governance
```

If actual full-unit requires the existing governance supplement,run only its
approved original module with the same py312 CPU extras,then execute the same
comparison with `--phase final`. Preserve both phase outputs and raw exits.

## Status and limits

At initial entry,full-unit is still running;no actual total is prefilled from
8018. Raw exit1 will remain visible if historical failures persist. The required
same-source101 supplement is separately101PASS;it is not an all-green replacement
full-unit result. Counts are never added across overlapping suites. All claims
remain observed regressions only;full CLI/native/GPU/T2/platform admission is
outside this assignment. Any retained-ID decision remains explicit,not hidden by
gate weakening or fixture edits.

## First execution and environment diagnosis

The first run completed with **8036PASS/279FAIL/32ERROR/267SKIP**,rawexit1.
Its279failure and32error IDs exactly match79;no priorPASS becomesSKIP/FAIL/ERROR.
It is a **non-canonical environment observation**:the driver projected a
currently available integrated06 binary,but retained79 XML proves absence when
the baseline ran. The new source clone itself has no binary.

Observed environment-only deltas,not relabelled baseline passes:

- Thirteen retainedSKIPs under
  `tests.unit.test_step5_comm_cost_export::test_finite_exporter_executes_joint_htsim_without_blessing_fixture[...]`
  were actually executed asPASS. Old reasons identify the missing binary.
- Old collectionSKIP `::tests.unit.test_collective_sim_zero_payload` expanded
  into three passing existing-source cases:one keeps synchronization latency,
  two across-server cases use ep8_one_phase/ep16_two_phases. These source files
  did not change since79;the three IDs are not genuine source additions.
- One parserPASS ID is replaced by the approved run-fixed testID,two additional
  genuine source tests pass. These three source-level new IDs are separate.
- Raw retained-ID/status predicate flags2missing IDs and13SKIP-to-freshPASS;
  no status alias or exception is applied. Original skipped results stay skipped
  in the retained baseline,and no native/G3/T2 qualification is inferred.

Correct the verification environment only by restoring the exact original
namespace without binary projection. Same source/test selection/cap/gates,
new unique outputs,no exclusions/skip markers/implementation:

```bash
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/baseline_environment/run_suite.py unit
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent06_baseline_environment --junitxml=/data/ycfeng/tmp/d29_unit_parent06_baseline_environment.xml
```

First XML/logs/comparisons remain intact. Original tests' missing-binary conditions
determine skips,without injecting new ones. Canonical rerun is pending at this
entry;new counts are not inferred. Required short receipts are reused since none
of their files depends on the binary. Strict77 is not rerun. Needed existing
governance52 runs independently with the same no-binary namespace.

Diagnosis and exact first-run deltas:
`/data/ycfeng/tmp/d29_post_key_regressions_20261024/native_availability_diagnosis.json`
and `unit_final06_before_governance_comparison.json`. The first diagnosis inspector
assumed nonexistent row.label/detail fields;existing schema uses selection and
message/text. After reading it,the inspector passed;no test/source/gate changed.

## Final canonical observed result

The original no-binary namespace completed in1124.468s under the retained1800s
cap. Actual XML8612 unique IDs: **8020PASS/279FAIL/32ERROR/281SKIP**,rawexit1.
No counts are added across suites. Final comparison command:

```bash
sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/baseline_environment/compare_results.py --phase final
```

Every retained79FAIL/ERROR/SKIP ID exactly matches. Every retained present ID keeps
its old status;noPASS-to-SKIP/FAIL/ERROR and no failure/error additions/removals.
All old optional binary-missingSKIPs remain unchanged. Genuine new IDs allPASS:

1. tests.unit.test_collective_sim_cc_backend::test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published
2. tests.unit.test_moe_routing_runtime::test_collective_input_binds_both_role_constructors_and_refreshes_next_run
3. tests.unit.test_prediction_measurement_bindings::test_prepared_points_follow_actual_model_rows_identity_and_fresh_queries

Exactly one oldPASS ID is absent,the planned hot-update parser test already named
above. Retained79 presence and old parser-parent128 presence predicates still
fail for this same unique ID. Comparator exit1 is preserved;no alias or waiver.
The new run-fixed replacement passing does not mean the oldID was executed.
Its behavior/source change was approved in the root contract;primary owns
admission of that declared change,not this regression worker.

Original locked267FAIL/31ERROR retained. The12aiohttp failures and one Torch
collection error relative to that older locked baseline are mechanically covered
by fresh same-source101,including16 existing native-method cases;five unchanged
governance Torch failures each pass in fresh52. Original group parent81PASS/1Torch
SKIP remains81PASS/1SKIP;trainers2PASS and loader29PASS remain unchanged. Parser
parent has127retainedPASS/1plannedoldID missing;no new code failure is observed.

Actual same-source197/53 memberships allPASS in canonical unit. No extra affected
union,baseline,component124,wave208 orfullCLI run. Finalsource2629file hashes,
13retainedinputs and8script bindings unchanged;HEAD/tree/3dacclean. Source clone
contains no binary. All raw scopes and differences remain separately preserved.

Canonical XML/log SHA256 and file sizes in evidence/final_verification.json;
raw XML `9d5ab2bcd0d5e969c4281b24f7f6b0fcde84a86ea344fbeef1b3173d93511df7`;
exact comparison receipt `30451fa98211a682fe9f2fb537f2162402376b964dbf8d9623a1f848aab6385e`.
Observation verdict:no new code failure after explicit environments. Raw exact-ID
predicate verdict:FAIL,one declared planned replacement. No automatic
qualification/admission or all-green unit claim;verification itself is complete.
