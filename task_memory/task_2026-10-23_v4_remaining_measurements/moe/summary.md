# W128 local-MoE measurement package

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Archived root image/SDK gates, successful1-H800 run and independently accepted isolated kernel data with explicit limits. |
| 2026-10-23 | Archived the frozen CPU preparation and explicit root-only runtime gates. |

## Outcome

Root accepted `ACCEPTED_ISOLATED_KERNEL_DATA_WITH_LIMITS` in `results/primary_acceptance.json`. Source737b6311 and documentation correction941a4450 were merged asbe067bb5. Actual-image440/20source/binary pins and1003inputs, fresh worker/SDK entries and both four-point dry runs passed. The sealed single-H800/codesign/non-preemptible job `exp-1023-235250-540507` succeeded2026-10-23T15:54:35Z; container exit0/restart0,92seconds. `root_run_check.json` binds platform/resources/mount/image and input integrity.

All19 outputs/3,261,347bytes were downloaded hash-identically. Two tables have four rows each;12traces exactly reproduce112statistics. Independent review found no issues; root reran the full audit with exact non-clock equality. At M16/32/6152/12288, grouped-GEMM KERNEL_ONLY means are0.1017937/0.1022997/0.7557434/1.29869045ms, each from20calls with7kernels/call. CUDA_EVENT remains finite aggregate diagnostics because individual durations and grouped-count are not retained. See `results/{review.md,root_verification.json,test_report_2026-10-24_data.md}`.

The following records describe the original owner preparation before root execution.

Task W128-V4-MOE-PACKAGE prepared a new, unsealed/unsubmitted local-MoE package. Two maintained non-core files were frozen at737b631191542202c42b32c0c4a225058021ff92: `moe/build_package.py`153 gross lines and `moe/profile_plan.json`51, total204/300. Runtime snapshot56e9405dad76d30d4ada9d5bdab124f898bf5065 and all original sources/packages remain unchanged. No further delegation or core/native/C26/consumer/admission edit.

Package: `/data/ycfeng/Frontier/.worktrees/v4-moe-local-run/task_memory/task_2026-10-23_v4_remaining_measurements/moe/h800_moe_local_01`. Manifest SHA256 `2ce8333148675cbec721c7f0f9710f70c1b031032be971ffcf7b1e754a1377f4` binds1003 inputs, including512 committed Frontier/config files,474 native reference files, original launcher/worker/wheels, builder/dependency/profile plan and corrected root requirements/plan/grid. Native source440 and inherited binary20 pins are references pending root actual-image verification. Original527 inputs remain unchanged. Only the generated launcher's WORKSPACE locator differs; worker/script bytes are retained. Separately counted generated inputs+manifest:1004 files/281898 gross text lines/3 binaries,23680569 bytes; none are maintained runtime changes.

Planned artifact root: `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h800_v4_moe/56e9405d_ep16_rank0_balanced_20261023_01`.

- `EP16_RANK0_LOCAL_MOE_CUDA_EVENT/compute/h800/deepseek-ai/DeepSeek-V4-Flash/moe.csv`
- `EP16_RANK0_LOCAL_MOE_RECORD_FUNCTION/compute/h800/deepseek-ai/DeepSeek-V4-Flash/moe_kernel_only.csv`
- Existing record_function traces beneath the method phase's `profiler_traces/`; CUDA_EVENT itself generates no traces.

These destinations are planned, not observed or created. Existing launcher JSON: package `jobs/moe.json`; root must retain the package's absolute paths when launching.

## Validation

`moe/cpu_checks.json`: eight fresh entry/dry-run controls PASS;1003 immutable hashes, Git-equal runtime/native source, original527 hashes, trace-copy CPU fixture, SDK source/default semantics, and explicit existing77matrix binding PASS. `moe/routing_checks.json`: exact balanced routing/local16 experts and two selectors PASS; rank0 aligned rows5120/3072/128/128 from actual native block-selection AST, CUDA uninitialized. `moe/matrix_binding.json`: existing77 identical-case comparison bound explicitly to unchanged simulator domains; no new matrix capture.

Existing tests:172 PASS/1 baseline-environment FAIL/2 SKIP; the pinned56e9405d baseline reproduces the same native vLLM modular-API failure and skips. No fallback/skip/runtime patch added. Exact commands and limits: `moe/test_report_2026-10-23_cpu_preparation.md`, selected XML/test receipt, and `moe/checkpoint_2026-10-23.md`.

The checkpoint corrected repeated CPU assertion guesses and one faulty stop signal that triggered a permission-refused local /mnt mkdir attempt. No successful cloud write occurred; final CPU control uses BaseException and an audit prohibition. No GPU/platform/network/Docker/seal operation or actual image/backend gate was executed.

## Original root handoff (completed; retained for history)

1. Independently review204-line source and generated inputs; check actual locked-image native source440/binary20 pins, the MARLIN selector/backend, actual producer entry and runtime capabilities. CPU backend labels/AST are not GPU execution.
2. Verify actual SDK non-preemptible/cpu16/gpu1/mem_gb128 (source maps to128000 decimal MB), personal auth/quota/local mount/new cloud destination; perform existing external root-seal procedure and submit only afterward.
3. Collect only the two planned four-row invocations on1H800/codesign/non-preemptible,1800s/≤0.5GPUh; inspect actual columns, finite measured-scope statistics, traces and terminal status.

Scope remains physicalTP1/EP16/rank0/local16/global256, M12288/6152/32/16, balanced_topk. M6152 is conditional mixed-prefill/dummy evidence. Candidate cost scope is only existing grouped-GEMM; gating uses M rather than native L_d and standalone shuffle overlaps. No rank/time/weight/graph/serving equivalence, simulator consumption or admission follows.

Current next action: establish native tuple/rank population and GPU dependency comparability before consumption. Do not add standalone shuffle/gating timings to the grouped-GEMM envelope or infer CPU overhead from the difference between measurement methods.
