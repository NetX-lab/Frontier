# CORE three-finding repair test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record R1 actual inheritance/delegation RED/GREEN with scoped quota. |
| 2026-10-23 | Record R2 automatic preflight/trainer missing-descriptor RED and36PASSGREEN. |
| 2026-10-23 | Complete R3 proof/source-owner RED/GREEN, exact artifact seam, affected regressions and scoped budget. |
| 2026-10-23 | Record authorized wrapper diagnostics RED/GREEN and real joint producer/current-repo CPU consumers. |

TaskID:D29-CORE-REVIEW-REPAIR. Isolated worktree `/data/ycfeng/Frontier/.worktrees/step5-core-review-fixes`,baseline d96b3279. No full CLI or GPU/native qualification. Root submodule checkout3dac preserved unstaged. Commands use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`;py312 addsVLLM_TARGET_DEVICE=cpu. Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`,Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`.

## R1 actual supported routing callable chain

Observed source: supplied published export `/data/ycfeng/tmp/d29_published_sources01/source/vllm/model_executor/layers/fused_moe/router/{fused_moe_router,base_router,routing_simulator_router}.py`. Reached chain: inherited select_experts -> inherited _select_experts -> bound router._compute_routing -> RoutingSimulator.simulate_routing -> selected position_cyclic strategy.route_tokens. Existing PRICING C1 probe independently verifies these same callables. Production appends these three actual callables to the existing full-source verifier tuple; previous _select_experts check stays. No recursion/verifier changes/native strategy changes.

Test reuses real exported class definitions in the existing sealed full-source CPU fixture, with only unrelated import/environment hardware seams isolated. Valid foreign replacement tensors first execute through the real inheritance chain; class/public selector/inherited selector/module filename/spec/hash remain unchanged. Wrong-source replacement is in callable's code filename, not misleading changed module metadata. Four cases cover previous selector and each reached implementation. Absent supplied source export is explicitly SKIP; this is CPU provenance fault injection, not published-runtime/kernel execution qualification.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-red01 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router' > /data/ycfeng/tmp/d29-core-review-r1-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-red02 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router' > /data/ycfeng/tmp/d29-core-review-r1-red02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-green01 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-review-r1-green01.log 2>&1
```

- RED01:3FAIL/1PASS2.96s;bad injected string results fail actual inherited dtype conversion before verifier. Not counted as genuine reproduction.
- RED02:3FAIL/1PASS3.10s;three `DID NOT RAISE ValueError` for actual compute/simulator/selected strategy. Previous selector negativePASS. This proves incomplete bound callable verification.
- GREEN01:23PASS/11deselected9.21s;all four actual-chain negatives plus timer/restore/event preservation/source/Gate/oracle seam tests. Two SWIG deprecation warnings only.
- R1repairgross44. Baseline1aee totals:moe_impl56/62,test303/350 after both authorized overlays. All six exclusive path caps remain authoritative; final accounting follows remaining repairs.

## Retained earlier CLI terminal, not a new run

Original shell notification already arrived before this repair task. `/data/ycfeng/tmp/d29-core-position-main-seam03/command.json`, `execution.json`, `run.log`:exit1,wall1555.208038s,timed_out=false,internal2400s unchanged. Initialization passes after2a862c25 and reaches actual PREFILL sync/EP wave then missing old `data/profiling/network/h800/collective_local.json`. This vector predates complete CPU payload/path forwarding, so proves neither current complete fixture/main nor native importer rejection. No polling, original worktree edit or CLI rerun in this repair branch.

## R2 automatic supplied source admission

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py -k 'automatic_position' > /data/ycfeng/tmp/d29-core-review-r2-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py -k 'not position_source_wrapper' > /data/ycfeng/tmp/d29-core-review-r2-green01.log 2>&1
```

- Existing actual producer creates canonical252-row FP8/BF16 source/gathered CSV; initial real manager preflight succeeds. Only routing_input removed (or replaced with empty descriptor), then actual automatic shared trainer with its preflight or independent trainer called. Estimator/metadata sentinel proves rejection order, not a shallow filter-only test.
- RED01:absent column shared/independent reach forbidden estimator/metadata (`Failed: Unvalidated source reached estimator or legacy helper`). Invalid descriptor already rejected;old test regex schema_version mismatched actualsource_frames and was corrected. No production relaxation.
- GREEN01:36PASS/3deselected68.81s;four new source negatives plus current automatic actual producercold/warmfit/cache,command/defaults/metadata contracts. Hardware wrapper excluded here andcoveredR1py312/currentintegrationseam separately.
- Production requires routing_input column and always calls existing validate_position_moe_source_row. No new source validator,estimator/default,unownedtrainer orregistryedit. R2delta3added/4deleted+40testadded=47;baseline runtime103/133,test125/135. R1+R2delta91/230.

## R3 complete actual-owner compute proof admission

Pre-agreed seam: `run_step5_measured_main.validate` called with complete CPU artifact fixtures; these are explicit fault injection, not actual completed CLI measurements. All original validator requirements run first:6144/1024,P6128,D16+1023single-tokenforwards,graphcapacity/topology,3.5msREADbytes,prefixreservation,source release/readiness,positivemetrics. Every damaged input preserves empty-expert proof while removing/changing actualdummy-owner source gate/router/reached prepare orpositivepairedkeys. Peer-only case retains a live-source-owner EP0 row which mentions dummyDP1 butmustnotcountasactualemptyowner.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/integration/test_step5_measured_main.py -k 'completed_position_validator' > /data/ycfeng/tmp/d29-core-review-r3-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/integration/test_step5_measured_main.py -k 'completed_position_validator' > /data/ycfeng/tmp/d29-core-review-r3-red02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/integration/test_step5_measured_main.py -k 'completed_position_validator' > /data/ycfeng/tmp/d29-core-review-r3-green01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/integration/test_step5_measured_main.py -k 'completed_position_validator' > /data/ycfeng/tmp/d29-core-review-r3-green02.log 2>&1
```

- RED01:25FAIL/2SKIP3.23s. Existing actual validator expects sliced expert counts but position-count registry returns dict;TypeErrorbeforeproofassertion,notmeaningfulmissing-proofRED.
- RED02: temporary correctdictindexing(ultimatelydeleted)exposes16FAIL/9PASS/2SKIP2.53s;16`DID NOT RAISE` foractualmissingproofs/operators/peer-owner. PreviousNaN/countzeroalreadyrejected.
- GREEN01:25PASS/2SKIP2.25s. Sourceemptymeansmappedactualowner.live_rows==0,notexpertassignment0oranypeerempty. Applicableoperatorinventorycomesonlyfromselectedregistrywithactualdispatchbuffercomputecontext. Completepositivefiniteexactpairscheckedagainstdurationinphaseoperators.
- GREEN02:36PASS/3SKIP2.49s. ExpandedpositiveBF16(noFP8prepare),FP8prepareanddeferredFP8;negativeinapplicableprepare,normproofdeletion,mispaireddurationandfractionalcount. AllCOMPUTEoperator/proofkeysnowexactlycorrespondviaexistingregistrybinding,notjustMoEsubset. Boolcountrejected. ThreeexplicitinapplicabilitySKIPs:BF16/deferredmissingprepare,andFP8nondeferredinapplicablepreparecase.
- No new normalizer/operator registry/source API,peer edits,measurement synthesis,CLI orchangedqualification. Fullcontextreadfromexistingbufferfield,neverconstructedfrompredictedcandidate.

## Final affected regression and environment limits

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py -k 'not position_source_wrapper' > /data/ycfeng/tmp/d29-core-review-union-green01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-r1-final02 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-review-r1-final02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_fixed_probe.py > /data/ycfeng/tmp/d29-core-review-legacy-union01.log 2>&1
git diff --check
```

- Affectedunion**72PASS/3SKIP/3deselected70.53s**;R1final**23PASS/11deselected9.50s**. Do not sumoverlappingGREENchecks. Actualnewautomaticproducertrainercold/warmcacheandpairedkeysincluded.
- Expandedoldfixtureunion**73PASS/1SKIP/3FAIL7.19s**. Threefailures: test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers;test_independent_selected_actual_fit_cache_binds_native_context;test_selected_output_inventory_uses_declared_cells_without_structural_zero_fill. Alluse`tests/unit/test_moe_grouped_gemm_padded_features.py::_position_source_rows`withoutmandatoryrouting_input;nowcorrectlyValueErrorbeforefit. Thisfileisnotinthesixexclusivepaths. Rootmustupdatefixturewithactualsource descriptor/census;donotweakenproductionadmissionorsuppresstests. ValidfreshactualproducercoveragealreadyPASS.
- Py312union`d29-core-review-r1-final01.log`failscollectiononmissingfasteners,exit2. AST-isolatedexistingwrapperprobe ast01/02 firstfailedonrunneronlyBaseModelConfig/jsonglobals(notproduction);ast03reachesactualgatheredwrapperthenalsomissingfasteners. Noinstall,peeredit,wrappermodificationorfallback. Noadditionalprobingafterconcreteenvironmentblocker. CPU torch warningsTriton0driver/libcudaabsenceexpected,separatefromprovenancechecks.
- Pythonversionsmeasuredwith `env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -c 'import sys; print(sys.version)'` ->3.12.13andcorrespondingfrontier-py310command ->3.10.6. Earlierpy312header3.12.3wasincorrectenvironmentlabel;actualcommandsusedexistingexecutablesandCPUreceiptusesitsownsys.version. Correctedhere,notrelabelingpublishedruntime3.12.3.

## Effective quota and exact diff

Applyrootregistry_scope_amendmentthen core_review_scope_amendmenttoscope.rowsgroupedpathceilings. `d29-core-review-budget01.log` verifiesallcaps,PythonASTandwhitespace. Exclusivechangedcode/testpaths6,totaldelta**+164/-18=182/230**,48remaining. EffectiveCORE46allocatedpaths(currentregistrytwoaddedpathsremainroot-owned),baselinegross**3450/4404**,954remaining;donotsimplyadd182tobaseline3304becausereplacedpriorhunkscancelinbaselinediff. Exactrepairdiffretainedlocallyascore_review_repair.diff.

| Path | Repair gross | Baseline gross/effective cap |
| --- | --- | --- |
| frontier/profiling/moe/moe_impl.py |3|56/62|
| frontier/moe_routing_runtime.py |7|103/133|
| tests/integration/run_step5_measured_main.py |23|46/111|
| tests/unit/test_step5_moe_profiling.py |41|303/350|
| tests/unit/test_moe_profiling_output_metadata.py |40|125/135|
| tests/integration/test_step5_measured_main.py |68|221/243|

No code outside sixpaths,sixper-filecapsPASS,rootgitlink3dacunchanged/unstaged,recordsseparate,rootcurrentWIPnotimported. CompleteCLI/native/GPU/G3/PW/NONE/FULLacceptanceoutofscopeandunverified.

## Authorized wrapper point-diagnostics persistence

Three P2 slices committed before immutable PRICINGa7e2a3ca development merge55a4df3c. No conflicts, peer edits or root acceptance assumed. User then permits immutable registryb6714757 only after this owned wrapper commit; registry integration is not included in the following evidence.

Existing source-local wrapper seam calls `_profile_position_cell` for actual dummy-source-owner width1/live0. A real full-source verifier runs before each row: second call includes one additional valid sealed native module. CPU expert object isolates numerical/hardware execution only. Stable native_context remains equal, constructor provenance object unchanged; complete point_diagnostics must retain ambient inventory and row native_facts union. Normal CSV serialization/deserialization must retain the independent diagnostic JSON. This verifies persistence, not actual native numerical work. PRICING's separate real prepared_context/observer/full-verifier join test runs below, so persistence is not claimed as a replacement for actual producer validation.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-point-red01 tests/unit/test_step5_moe_profiling.py -k 'position_row_preserves' > /data/ycfeng/tmp/d29-core-review-point-red01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-point-red02 tests/unit/test_step5_moe_profiling.py -k 'position_row_preserves' > /data/ycfeng/tmp/d29-core-review-point-red02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-point-red03 tests/unit/test_step5_moe_profiling.py -k 'position_row_preserves' > /data/ycfeng/tmp/d29-core-review-point-red03.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-point-green01 tests/unit/test_step5_moe_profiling.py -k 'position_row_preserves or only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-review-point-green01.log 2>&1
env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_collective_sim_cc_backend.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_step5_moe_pricing.py > /data/ycfeng/tmp/d29-core-review-joint-consumer01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-review-joint-producer01 tests/unit/test_step5_dispatch_probe.py -k 'two_native_point_producers_join or native_point_writer or observed_compute_point' > /data/ycfeng/tmp/d29-core-review-joint-producer01.log 2>&1
```

- RED01/02: one error each before row seam because fixture vllm lacks __version__. Model-config replacement did not address eager import, not a valid RED. Test-only CPU version label restores required import shape; sealed file/spec/hash unchanged, no production or peer changes.
- RED03: one actual row assertionFAIL2.91s; new ambient module absent from row imports while context stable. This is the genuine persistence reproduction.
- GREEN:24PASS/11deselected9.35s. Exactly3sourcegross: provenance replaces context.native_facts with full experts.point_diagnostics; independent row point_diagnostics canonical JSON added. Existing verifier/source/context semantics unchanged.
- Actual joint producers:7PASS/243deselected2.91s, includes real prepared_context/observer full verifier wrong-origin/hash/ambient/layout negatives and stable-context/complete diagnostics join.
- Existing current-repo CPUwriter/H04consumer:144PASS/1SKIP4.81s, both FRONTIER_TEST_CPU_WRITER reference variables unset. No external writer/fallback/dependency install. Not full main/native acceptance.
- Seven owned repair paths now, delta**+207/-19=226/230** (4remaining), baseline test344/350, wrapper171/194. Other six totals unchanged. Final combined effective CORE accounting follows immutable registry integration, counting registry owner's existing paths separately.
