# Canonical position producer and consumer validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record real automatic fit/cache consumers, source census and partial runtime boundary. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Worktree step5-position-core, Python frontier-py3103.10.6. CPU evidence only; no native numerical/GPU qualification.

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-selected-production06 tests/integration/test_step5_measured_main.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py
```

Observed **83 PASS,4 SKIP** in78.64s. Log `/data/ycfeng/tmp/d29-core-selected-production06.log`. Expected: KERNEL_ONLY/CUDA_EVENT rows enumerate FP8/BF16 without old helpers, preserve physical/live0 positive pairs, actual source/native schema, both trainers cold/warm reload without estimators. All observed PASS. Synthetic fixture252 rows/seven descriptors with only decode-context range narrowed; full fixture retains6144/1024 semantics and is not admitted native evidence.

Initial source gate fit entered expert schema; classifier now expert-only. Missing EP/count fields produced non-reflexive NaN cache comparisons; retained missing fields use JSON null. One old probe lacked new model keys; fresh producer carries actual dimensions. Three legacy-shaped executor doubles failed strict enumeration; only test forwarding seam stubbed, real admission tested separately. Errors retained in `/data/ycfeng/tmp/d29-core-selected-production05.log`, `/data/ycfeng/tmp/d29_core_independent_auto01.log` and prior auto-fit logs.

Fresh independent automatic trainer probe `/data/ycfeng/tmp/d29_core_independent_auto02.log`: PASS4models/4warm. Persistent two-method tests supersede this probe. Synthetic durations/layout receipts and narrow CPU leaves are not image/runtime proof. Registry replacement and unowned CPU-plan entry remain actual-main blockers; full native/E2E/strict77 are root-owned.
