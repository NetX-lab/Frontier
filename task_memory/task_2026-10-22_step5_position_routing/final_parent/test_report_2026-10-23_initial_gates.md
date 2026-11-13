# Initial integrated parent gates

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Append YC authorization and parent194PASS on the repaired frozen source; strict77/CLI reruns are in progress. |
| 2026-10-23 | Preserve full-unit and strict77 RED, fixture preparation, restored T2 verification and merge-duplicate correction. |

## Snapshots and outcome

Full unit/fidelity snapshot:9974acdb, clean with collective-sim3dac53c. T2/main
assembly:0f017dff; parent corrected its identical duplicate test in d0dfcdb5.
CLI fixture generator remains frozen at19a35630. These are intermediate snapshots;
the complete D29 task has not been accepted.

| Check | Observed result | Interpretation |
| --- | --- | --- |
| Full Python3.10 unit suite | 8005 PASS /285 FAIL /281 SKIP /32 ERROR,1185.59s | RED relative to historical baseline; detailed disposition below |
| Strict77 candidate execution | 77 successful cases | Execution succeeds; comparison still RED |
| Strict77 comparison | 0 identical /77 mismatched | Only reported artifact types: precision metadata and EP-wave ledger |
| Restored native-position module, Python3.12 | 16 PASS,4.566s | Includes T2 engine limits; no GPU claim |
| Deduplicated native-config parameter cases | 2 PASS,1.144s | Same AST and file bytes as existing accepted test |
| Canonical HTTP/source module, Python3.12 | 85 PASS,3.642s | Resolves all12 new aiohttp-bound failures from the full Python3.10 run |
| Fresh position CLI input generation | exit0;7 CSV files,76 pickle files;P/D bundles2/5 | CPU fixture input only; CLI execution is pending source repair |

Suite results are separate and must not be summed.

## Full-unit comparison

Historical full baseline `/data/ycfeng/tmp/d30_unit_baseline01.xml` is b8be287:
8093 tests,267 failures,31 errors. Candidate has18 new failure IDs and one new
collection-error ID; no historical failure/error disappeared. Twelve new cases
fail for missing aiohttp and the new native-position module cannot collect without
torch. Existing Python3.12.13 environment resolves these dependencies without
editing either environment: canonical85 and native-position16 both pass.

Six remaining existing tests have stale catalogue or mock-interface assumptions:
one V4 inventory expectation incorrectly includes the newly declared Step-only
operator, two assert the previous four-member MoE catalogue, and three mock only
profiling_ops instead of the real family interface that also provides operators.
PRICING is the proposed owner under repairs/legacy_output_scope_amendment.json;
the two additional test paths await YC approval. Existing four-path repairs continue.
Raw full-suite outcomes remain unchanged evidence; this is not a second full run.

## Strict77 diagnosis

Compare source9974 against713e6a8 using the original comparator. All77 cases gain
one `moe_input_quantize,FP16,profiling` row. Thirty-nine MoE cases also gain empty
collective_proofs on all five phases. There are zero execution failures, missing
evidence, definition/provenance differences or predictor-cache filename changes.
The real MetricsStore writer on configured Llama-2 and the existing EP-wave
reporting fixture reproduce both defects without a Step position configuration.
Repair allocation and RED details: repairs/legacy_output_repair.md.

## T2 assembly correction

Main already contained accepted892b6a1/T2 and the12-line size note. Merging it with
9974 restored the native graph/unit files (baseline totals306 and313 lines,
native total1350). Automatic merge also duplicated the same37-line test function
and decorators in test_step5_dispatch_probe.py. Both definitions have identical
AST SHA256a0ab621486f0c06005e171d810efaf1d4dc161acd89a63b1e3d6f3174d6676b8.
Root removed only the duplicate41-line block in d0dfcdb5; the entire resulting
file is byte-identical to9974, its top-level names are unique, and both parameter
cases pass. Native T2 remains present; no distinct assertion or test was removed.

Before pending repair changes, actual cumulative source totals return to native
1350 +CORE3833 +PRICING3805 =8988; gitlink2 and size note12 give9002 global.
The earlier8907 measured the pre-T2 snapshot;9029/9043 measured the intermediate
duplicate-containing merge. Neither is a final repair total. Per-path ledger
reconciliation remains pending actual repair delivery.

## Exact commands and artifacts

Common environment: `PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1
TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Full suite runs in verify-step5-d29-integrated-01 with Python3.10.6; native checks
run in verify-step5-d29-integrated-02 with Python3.12.13 and
`STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm`.

```bash
timeout 1800s env FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent02 --junitxml=/data/ycfeng/tmp/d29_unit_parent02.xml > /data/ycfeng/tmp/d29_unit_parent02.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root "$PWD" --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261023 --jobs 4 --clean-cache --continue-on-failure
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261023 --baseline-label /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate --candidate-label /data/ycfeng/tmp/d29_workflow_fidelity_20261023/candidate
timeout 180s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_t2_restored_parent01 --junitxml=/data/ycfeng/tmp/d29_t2_restored_parent01.xml
timeout 60s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_dispatch_probe.py -k native_config_binds_only_supplied_limits -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_t2_duplicate_parent01 --junitxml=/data/ycfeng/tmp/d29_t2_duplicate_parent01.xml
timeout 120s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_fullunit_env_parent01 --junitxml=/data/ycfeng/tmp/d29_fullunit_env_parent01.xml
timeout 2400s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 /data/ycfeng/envs/frontier-py310/bin/python -B -m tests.integration.step5_measured_compute_fixture --output-dir /data/ycfeng/tmp/d29-parent-position-fixture03 --device h800 --gpus-per-role 16 --position-cyclic
```

Fidelity cwd is d29-fidelity-9974acdb, with PYTHONPATH set to the repo only; fixture
cwd is verify-step5-core-closure-01. Native source-root/VLLM_TARGET_DEVICE=cpu apply
to native-position and canonical modules; the duplicate-config check uses its
existing stubs and does not set these variables.
Fixture manifest: `/data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json`.
It declares cpu_fixture/native_admitted=false,TP8/DP2/EP16,model bound7168,
position_cyclic and explicit synthetic collective input. It is not H800 execution.
Raw logs and comparison JSON remain under the named temporary paths; compact
hashes and outcome records are retained in final_parent/evidence.json.

## Pending acceptance

Finish/review/verify legacy-output and six fixture repairs; reconcile actual
effective path ceilings after YC decides the two-file proposal; run unchanged strict77 again and the actual offline/online CLI
on the repaired source using the already-generated inputs. Before native launch,
rebind the T2 package's complete source provenance to final integrated files.
GPU/native, graph, G3 and E2E numerical acceptance remain separate.

## Subsequent repair checkpoint

YC approved both extra paths and similar necessary repair-path adjustments:
`批准，类似问题都授权` (requirements D29-REPAIR-PATH-AUTHORITY). Earlier pending
approval statements above describe the initial checkpoint. Effective scope111
with unchanged owner/global budgets; the original owner resumed both tests.

Four-file fix b60dfddd is37 gross against9974,35 retained-line growth. Root merged
it with the restored/deduplicated T2 tree as60ba4c7569b7c9fd4c8f9f43a16f92249fa76181
and initialized the same local detached submodule3dac53c. No network/build.
Parent independently obtained **194 PASS,0 FAIL/ERROR/SKIP,6.348s**:

```bash
timeout 180s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_legacy_parent07 --junitxml=/data/ycfeng/tmp/d29_legacy_parent07.xml > /data/ycfeng/tmp/d29_legacy_parent07.log 2>&1
```

Workdir verify-step5-d29-integrated-02 stays frozen for active strict77 at
`/data/ycfeng/tmp/d29_workflow_fidelity_20261023_repair02` and actual CLI at
`/data/ycfeng/tmp/d29-parent-position-main04`. Exact launch:

```bash
timeout 5100s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json --output-root /data/ycfeng/tmp/d29-parent-position-main04 --mode all > /data/ycfeng/tmp/d29-parent-position-main04.log 2>&1
```

The runner retains2400s for each mode. Rerun output is pending, not a pass;
independent four-file review and T2 full-source resealing run separately.

## Strict result and approved fixture follow-up

The unchanged comparator subsequently reports77/77 identical for60ba4c75 against
713e6a8. Parent verifies451 artifact hashes and519 cache filenames on each side;
receipt:fidelity_repair02_audit.json. The independent reviewer nevertheless finds
a Step-specific metadata precision/source/factor inconsistency outside this legacy
matrix. Its P2 remains open in review_2026-10-23_legacy_output.md.

Two-file test repair047f7313 is16/60gross. Parent runs both whole files in the
separate verify-step5-d29-integrated-03 tree and obtains53PASS,zero
FAIL/ERROR/SKIP,3.026s;all six original full-suite regression IDs occur in this XML:

```bash
timeout 120s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_share_expert_operator_families.py tests/unit/test_dsv4_precision_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_catalogue_parent08 --junitxml=/data/ycfeng/tmp/d29_catalogue_parent08.xml > /data/ycfeng/tmp/d29_catalogue_parent08.log 2>&1
```

Receipt:catalogue_fixture_receipt.json. Production source remains unchanged from
the active60ba CLI;the two tests and task records alone differ. Do not replace
the original full-suite results or claim another full run. Numerical-source P2,
actual CLI outcomes and remaining native gates still require acceptance.
