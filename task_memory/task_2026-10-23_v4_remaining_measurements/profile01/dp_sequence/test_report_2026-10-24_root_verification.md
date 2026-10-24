# Root diagnostic DP sequence verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded fixture-enabled tests, same-input baseline classification and independent94-check binding verification. |

## Environment and commands

Test interpreter `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13; no conda activation. Working directory `/data/ycfeng/Frontier/.worktrees/v4-dp-sequence-record`, HEAD178c1123. Exact argv, environment and start/end timestamps for each suite are retained in `root_test_receipt.json`; original XML/logs are in `root_logs/`.

Each suite ran independently through `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env ...`, with empty CUDA_VISIBLE_DEVICES, PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site. The native checkout is the pinned `glm-h5-clean-2cf0a69`; TMPDIR is `/data/ycfeng/tmp/glm53_dsv4/dp_sequence_root_w4peji6l`.

The existing C26 fixture path is a read-only alias to sealed profile02 for only overlay_model.py, overlay.json and source_lock.json; all three hashes match the source binding. This alias does not establish an earlier candidate run identity. NATIVE_CHECKER_LEGACY_BASELINE points to the retained `h200_glm_canonical_integration_20261021/check_groundtruth_run.before.py`.

| Suite | Expected before restored-fixture discovery | Observed |
| --- | --- | --- |
| test_native_v2_serving_inventory.py |46PASS|46PASS|
| test_native_serve_profile.py |36PASS|36PASS|
| test_native_serving_inventory.py |28PASS|28PASS|
| test_native_groundtruth_check.py |99PASS|98PASS/1FAIL|
| test_native_full_structure.py |6PASS|6PASS|

Total214PASS/1FAIL/0SKIP; the original aggregate receipt remains FAIL. The sole failure is `test_v2_native_auto_and_c26_actual_collector_cover_all32ranks[serving_counters0]`:32 setup KeyErrors for absent `vllm.v1.worker.gpu.dp_utils`, followed by an incomplete serving-inventory verdict.

## Baseline classification

The exact same single test, real fixture inputs, legacy baseline, workload and environment was run at44f2ddb9 and178c1123. Both returned0PASS/1FAIL/0SKIP and32 identical setup errors. Original commands/results are `baseline_control/{baseline,candidate}.json`; diagnosis is `baseline_diagnosis.json`. Root independently compared all32 counter and hook dictionaries and rehashed the bound source/log/input files. The baseline limitation is accepted as unchanged, not relabelled PASS.

## Root binding command

```bash
python3 -B /data/ycfeng/tmp/glm53_dsv4/dp_sequence_root_binding_20261024.py
```

Working directory is the primary `glm53-dsv4-vllm028` worktree. Archived script: `root_binding_script.py`; result `root_verification.json`:94checks PASS. It verifies source/Git bytes, owner record hashes, frozen R2/R3, three native sources, unchanged base inventory methods/hooks and the complete existing77-case comparison/domain identity. This is an existing comparison binding, not a fresh matrix capture.

An initial root binding check incorrectly addressed hooks through an observer object. Inspection confirmed the actual row schema has top-level hooks; only the verifier read was corrected. Native pairing, CUDA execution and serving/operator costs remain unverified.
