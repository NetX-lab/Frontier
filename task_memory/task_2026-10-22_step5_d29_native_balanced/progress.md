# D29 native routing progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record YC policy approval and integrate native design; numerical implementation scope remains pending. |
|2026-10-22|Recorded read-only native path findings and proposal investigation.|

## Current status

- completed: YC approved `position_cyclic`, necessary native/Frontier/profiling
  changes and internal derived-image publication; exact authority is recorded
  in `requirements.md` and the parent D29 decision.
- completed: `native_design.md`, commit `651415a`, is integrated on the parent
  branch. Estimated native-facing scope: 16 files, 716–1143 added/deleted lines;
  existing device positions are bound at the uncompiled Step outer forward.
- in-progress: independent native design review and Frontier/profile design,
  followed by a single combined numerical scope proposal. D30 proceeds in parallel.
- pending: approved bounded implementation, source/CPU/GPU graph verification,
  derived image, new source pins/package, and actual native/G3/profile evidence.
  Capture03 is a historical sealed pressure candidate, not a position-policy run.

## Completed investigation (historical)

- completed: frozen capture03 config gives default auto backend, no mega/SP override, default allgather_reducescatter, no EPLB. FP8 dynamic128 layers3–87; BF16 routed88–90; swiglustep7, E352/K8, expert-localTP1/EP16,22 full experts per rank.
- completed: retained actual H800 S2 TP8/DP2/EP16 module run uses auto backend and observes Fp8MoEMethod/TritonOrDeepGemmExperts→DeepGemmExperts and BF16/UnquantizedFusedMoEMethod→TritonExperts, both NaiveDPEPModular and no SP. E8M0 OR shape predicate determines DeepGEMM; small M does not alone imply Triton (the retained M9 case used DeepGEMM). This is native module evidence, not full serving/graph qualification.
- completed: exact oracle source removes H800 monolithic FP8 TRTLLM by SM100-only device predicate and rejects preferred CUTLASS for swiglustep; DeepGEMM modular supports it. BF16 H800 priority starts Triton modular. Both call `MoERunner._apply_quant_method`→router.select_experts before modular prepare's DP all_gatherv. Existing model gate remains upstream.
- completed: normal factory creates biased-sigmoid router; nonempty existing simulation env creates RoutingSimulatorRouter instead. Only random-uniform/normal built-ins exist; round-robin is expert placement and other load_balance matches belong to vision/attention. No balanced setting found.
- completed: `contract_probe.py` actual frozen factory/runner/DP method seams and seven hypothetical global-prefix count cases PASS; exact Frontier Hamilton counts, distinctness and conservation. DP reset fails6/7 cases. All16 retained native module witnesses agree; exact source/witness hashes retained in `cpu_receipt.json`.
- completed: `semantic_probe.py` counterexamples PASS. Forward reset differs on16 tail rows for P6128+D16/full6144 and4096 rows for chunks[2048,2048,2032,16]; absolute-position IDs preserve every real row. q1 starts224/232/240/248, not0. Stable-position and current ascending-Hamilton cannot both hold for arbitrary waves. Actual native dummy method CPU execution yields positions0 and padding true.
- completed: `proposal.md` records actual reachability, native settings absence, MRv2 exact device positions vs optimistic CPU mirror, FULL graph capture implications, weight/mask contract and two materially different pending decisions. No native strategy/image/package change or backend flip performed. Capture03 remains not balanced.
- completed: preservation check verified all57 capture03 manifest entries and50 native exported-source pins unchanged;3 probe sources compile. Exact manifest digest unchanged and no attempt/submitted marker. `test_report_2026-10-22_native_routing_contract.md` and `check_progress_2026-10-22.md` complete; no route drift.
- completed: task-only records/probes/receipts prepared for commit after `git diff --cached --check` PASS; no files outside this task are staged. Clean branch/commit identity is returned in the final handoff.
- completed: the original proposal-only handoff preceded YC's policy approval;
  remaining implementation work now follows the current status above.

Commands (owned repo, Python3.12.13/Torch2.9.1+cu128, CPU-only):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/contract_probe.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/semantic_probe.py
```

Semantic probe first attempt failed only on the CPU seam object lacking `.device`; no receipt written. Inspected actual `make_dummy` dependencies, supplied CPU device and local deterministic UUID seam; one rerun passed. This did not change native behavior or suppress a native error. Receipts are exclusive writes; rerunning requires a new filename/directory.

Preservation probe first attempt misread image-installed absolute pin paths as local paths (`FileNotFoundError` for `/usr/local/lib/python3.12/dist-packages/vllm/config/kv_transfer.py`). Inspected sealed pin format and mapped its exact fixed prefix to the retained local export; one rerun passed50/50. Manifest/pins untouched. Task directory follows existing ignored-task convention; explicit force-add only this task's files, never production/native or another task's files.

Local calibration site GPU defaults are overridden by explicit task H800/read-only/provider rules. Source-only/module receipts are not formal timing/GPU route attestation. No CPU overhead measurements or suits.
