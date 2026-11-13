# D29 read-only native routing verification report

## Modification History

| Date | Change |
| --- | --- |
|2026-10-22|Recorded bounded CPU method/math probes, retained all-rank native module evidence, preservation and verification limits.|

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`.
Interpreter: `/data/ycfeng/envs/asyncopd-py312/bin/python`, CPython3.12.13, Torch2.9.1+cu128. All tensors used here are CPU; no CUDA initialization/call or native engine/backend instance. No package installation, no conda changes, no GPU/platform/auth/network/cloud/SSH/collective operations. No formal run, timing or CPU-overhead measurement.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/contract_probe.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/semantic_probe.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d29_native_balanced/preservation_check.py
```

Receipts use exclusive writes. They are task-scoped investigation artifacts, not an installed routing implementation or a broad persistent test suite.

## Passing conditions and actual observations

| Check | Expected passing condition | Actual observation | Result |
| --- | --- | --- | --- |
|Retained16-rank native module witnesses|All ranks agree on actual auto FP8/BF16 classes/prepare/no-SP|16/16: layers3 Fp8/DEEPGEMM/TritonOrDeepGemmExperts,88 BF16/TRITON/TritonExperts; NaiveDPEPModular/no-SP|PASS retained module evidence|
|Actual frozen router factory method seam|Nonempty existing simulation selector reaches simulation router|Marker constructed with top_k8/global352/eplbNone; model weight/scoring settings not forwarded|PASS CPU seam|
|Actual frozen MoERunner method seam|Both modular methods call select_experts then forward_modular|Exact four-call sequence FP8 select/modular then BF16 select/modular|PASS CPU seam|
|Actual native DPMetadata methods|Read CPU sizes/cumulative prefix without collective|Seven cases pass local sizes/cumulative and context cleanup|PASS CPU method|
|Hypothetical global-prefix IDs|Exact current Frontier counts, distinct top8, sumN*K,max-min<=1|7/7 exact counts; reset6/7 mismatched,16/128/16/64/128 mismatch examples|PASS policy math; reset disproved|
|Selected sigmoid weights hypothesis|Normalized selected scores scaled3 conserve3|All tested CPU rows sum3 within1e-6|PASS math only; not native-GPU numeric proof|
|P6128+D16/full6144|Expose reset counterexample; position proposal preserves IDs|16 tail rows differ under reset; position IDs exactly agree; first tail96–103 vs0–7|PASS semantic counterexample|
|Prefix chunking|Same real positions preserve IDs|Chunks[2048,2048,2032,16]: reset4096 mismatched rows; position IDs agree|PASS semantic counterexample|
|q1 continuation|Position-based starts vs forward reset|Position6144..6147 starts224/232/240/248; reset0/0/0/0|PASS semantic counterexample|
|Dummy/padding invariance|Real IDs unchanged when dummy/stale padded rows added; actual dummy positions0|Three padding variants preserve real IDs; actual make_dummy CPU method uses0 positions/mask true|PASS CPU seam/math|
|Instantaneous balance conflict|Demonstrate stable position does not equal ascending-Hamilton aggregate|realq1(6144)+dummy0 and two requests at6144 differ; repeated expert countmax2|PASS semantic conflict evidence|
|capture03 immutable|All sealed entry sizes/hashes equal D28 manifest|57/57 entries, manifest `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d` unchanged|PASS|
|Native source immutable|All50 normalized installed-source pins equal retained local native export|50/50 exact; no source edit|PASS local exported-source evidence|
|Probe syntax|Compile three probe sources without writing pyc|3/3 compile|PASS|
|No launch artifacts|No attempt/submitted/job receipt marker; no job operation|No marker found; no submit/backend/auth/platform operation occurred|PASS local scope|

No timing/simulation error metrics are calculated because this task does not predict or measure latency. CPU mathematical assignment checks do not qualify the GPU route or numerical reference.

## Quick iterations / original errors

1. Semantic probe initially failed with `AttributeError: 'types.SimpleNamespace' object has no attribute 'device'` in actual `InputBatch.make_dummy` at line136. Inspected dependencies and supplied the CPU seam `.device` and deterministic local UUID function. One rerun passed; native code unchanged; no receipt was written by failed attempt.
2. Preservation probe initially failed with `FileNotFoundError: /usr/local/lib/python3.12/dist-packages/vllm/config/kv_transfer.py` because sealed pins correctly use image-installed absolute paths. Mapped that exact fixed site-packages prefix to the retained local export, then50/50 hashes passed. No pin/manifest content was altered; no failed-attempt receipt written.

## Evidence limits / pending decisions

- Source selection plus historical module evidence settles both modular route reachability, but **current full-serving FP8/BF16 subkernel/E8M0/graph behavior remains UNVERIFIED**. In particular smallM does not imply Triton: E8M0 OR shape predicate may choose DeepGEMM, and retainedM9 observed it.
- Ordinary biased routing passes `is_padding` to native topk; random simulator does not. Exact masked live-routed count is distinct from padded physical EP rows and must be represented truthfully in an approved policy.
- The frozen image has no exact balanced setting. Request-position cyclic routing needs new native context/strategy behavior and changed Frontier counts. It preserves routing invariance, not a guarantee of numerical/state equality.
- Preserving independent G3 vs insisting on current ascending-Hamilton instantaneous balance is a material semantic decision. Neither native/image/package change nor a separate timing-only campaign has approval from this child task.
- Capture03 remains sealed/unsubmitted/not balanced. No attempt/submitted marker or GPU job was created. Next generation only after parent integrates final approved source/image/settings. Calibration order stays workflow -> GPU/operator gap -> CPU overhead last.
