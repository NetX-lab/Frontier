# W128 local-MoE CPU preparation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded package, fresh entry, source/routing, existing test and immutable-input checks; distinguished root-only runtime gates. |

## Outcome and environment

Task W128-V4-MOE-PACKAGE: CPU package/entry/input planning verified; actual image/backend/GPU and root seal remain pending. Source commit737b631191542202c42b32c0c4a225058021ff92, runtime snapshot56e9405dad76d30d4ada9d5bdab124f898bf5065, native reference2cf0a6915ce544dc493a0990f2ea38d81601128a. No simulator/native/C26/collector/consumer/admission change.

Working directory: `/data/ycfeng/Frontier/.worktrees/v4-moe-local-run`. Existing virtual environments, no conda or installation: `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6; `/data/ycfeng/envs/distillm-py310/bin/python`3.10.20/Torch2.1.2+cu121/NumPy1.26.4. CPU Torch checks used CUDA_VISIBLE_DEVICES='' and confirmed CUDA uninitialized. Host temporary files were under `/data/ycfeng/tmp`.

## Exact commands and results

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B \
  task_memory/task_2026-10-23_v4_remaining_measurements/moe/build_package.py
```

PASS: generated `moe/h800_moe_local_01`,1003 pinned inputs, exact4 rows per method, manifest SHA256 `2ce8333148675cbec721c7f0f9710f70c1b031032be971ffcf7b1e754a1377f4`. Re-running the same command is a required negative control: exit1, `Fresh generation only; never alter an existing package`; all inputs unchanged.

```bash
env PYTHONPATH=.:/data/ycfeng/envs/distillm-py310/lib/python3.10/site-packages:/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages \
  PYTHONDONTWRITEBYTECODE=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
  CUDA_VISIBLE_DEVICES='' TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/distillm-py310/bin/python -B -m pytest \
  tests/unit/test_dsv4_moe_profiling.py tests/unit/test_balanced_routing.py \
  tests/unit/test_moe_load_distribution_contract.py tests/unit/test_moe_fused_event_contract.py \
  tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_moe_profiling_output_metadata.py \
  tests/unit/test_moe_routing_runtime.py tests/unit/test_profiling_timing_stats_contract.py \
  tests/unit/test_moe_ep_sampling_domain.py tests/unit/test_examples_profiling_contracts.py \
  tests/unit/test_moe_routing_input_contract.py -q -p no:cacheprovider --tb=short \
  --basetemp=/data/ycfeng/tmp/w128-v4-moe-cpu-20261023-04 \
  --junitxml=/data/ycfeng/tmp/w128-v4-moe-cpu-20261023-04.xml
```

Observed exit1: **172 PASS/1 FAIL/2 SKIP**,16.39s. The failure `test_checkpoint_mxfp4_runtime_requires_cuda` first tests online-AITER admission and requires the absent native vLLM modular API: `NotImplementedError: MXFP4 profiling requires current vLLM's modular fused MoE API.` Both skipped tests require installed native Marlin. These are not package regressions and are not declared PASS.

Pinned baseline replay used exactly the same environment/pytest options, cwd `moe/h800_moe_local_01/source/56e9405dad76d30d4ada9d5bdab124f898bf5065`, and absolute test path `/data/ycfeng/Frontier/.worktrees/v4-moe-local-run/tests/unit/test_dsv4_moe_profiling.py`, with basetemp `/data/ycfeng/tmp/w128-v4-moe-baseline-20261023-01` and XML `/data/ycfeng/tmp/w128-v4-moe-baseline-20261023-01.xml`. Observed exit1:5 PASS/the same1 FAIL/the same2 SKIP,3.35s. No runtime/test patch or skip was added. Selected XML and exact failure/skip records are retained in `test_receipt.json` and `cpu_test_results.xml`/`baseline_test_results.xml`.

`cpu_checks.json.commands` stores exact argv, inline Python code, cwd, selected environment, timestamps, stdout/stderr and expected exits for eight fresh-process controls. These execute generated launcher/worker paths, not imported stand-ins before entry. Two launcher controls use `python -I -B -c <recorded-code>` under Frontier Python and the existing StepMind interpreter, stopping before dependency imports/submission. A later explicit CPU ResourceConfig/spawn stand-in checks command/mount/resources, not actual SDK behavior. Worker help and pre-runtime entry PASS; the corrected entry uses BaseException and an audit prohibition on /mnt accesses. Both actual generated `python -m frontier.profiling.moe.main ... --dry_run` commands PASS under the committed snapshot without preloads. The existing `_copy_results` retained every byte of a local CPU fixture, including a `profiler_traces/profiler_trace_sample.json`; this is not a real GPU trace.

Routing control uses the existing `tests/unit/test_moe_routing_input_contract.py::_cpu_wrapper`, actual `_prepare_routing_inputs`, actual parser/backend-label functions, and the block-selection AST directly read from native marlin_moe.py:329/333-335. Its environment matches the CPU suite without VLLM_BS_SOURCE_ROOT; results are in `routing_checks.json`. No native backend launch occurred.

```bash
bash -n task_memory/task_2026-10-23_v4_remaining_measurements/moe/h800_moe_local_01/launch/launch_glm_producers.sh
git show --numstat --format=fuller 737b6311
git diff 56e9405d -- frontier data tests examples
git diff --cached --check
```

PASS: shell syntax only; two maintained files153+51=204/300 gross lines; no changes in runtime/test domains; source/records have no whitespace errors. Final staged-record check excludes byte-identical generated originals: `git diff --cached --check -- task_memory/task_2026-10-23_v4_remaining_measurements/moe ':!task_memory/task_2026-10-23_v4_remaining_measurements/moe/h800_moe_local_01'`. Generated files are checked by original/Git hashes, not cosmetically rewritten. The shell wrapper's proxy/auth/network body was not executed.

## Source/input expectations and observations

| Check | Expected | Observed |
| --- | --- | --- |
| Exact grid | M12288/6152/32/16; TP1/EP16/local16/global256/topk6, balanced/seed0,1 sample | PASS4 rows/method; no EP8 or extra points. |
| Methods | cuda_event→CUDA_EVENT; record_function→KERNEL_ONLY | PASS actual parser and dry-run; separate dirs and CSV paths. |
| Routing | balanced_topk/global_round_robin_balanced; no logits | PASS exact CPU IDs `(6r+j)%256`, rank0 expert_map0..15 then -1. |
| Native block/rank0 shape | 64/64/8/8;5120/3072/128/128 aligned rows | PASS actual native AST; assignments4608/2320/16/16. |
| Runtime source | All snapshot bytes from56e9405d | PASS512 committed files. |
| Native reference | Source/symbol pins2cf0a69 | PASS474 Git-equal files/440 installed-source hash pins;20 inherited binary hashes. Actual image not checked. |
| Original inputs | Base remains unchanged | PASS527 hashes and original manifest7929afae…; retained worker byte-identical, generated launcher only WORKSPACE locator differs. |
| Existing77matrix | Explicit source binding; simulator unchanged | PASS77 identical cases; comparison SHA7a8728ec…; no new capture. Details in matrix_binding.json. |
| SDK source units | mem_gb128/non-preemptible/codesign/H800 | Source says128000 decimal MB/default preemptible=False. CPU stand-in matches; actual SDK/platform root-owned. |

The existing record_function tracer exports Chrome traces from gating/shuffling/grouped-GEMM scopes (`frontier/profiling/common/record_function_tracer.py:143-160`); the existing Q1 worker archives them without pruning. CUDA_EVENT only measures statistics and does not itself generate traces. The historic W4 worker skips traces and was not used. Root must verify actual trace presence and finite statistics after the run.

## Preserved failures and limitations

Initial CPU setup failures: nonexistent tracer test filename (exit4/no tests), missing pandas/torch in individual environments, and NumPy2/Torch2.1 ABI mismatch (`_ARRAY_API not found`/`Numpy is not available`). Existing NumPy1.26.4 first on PYTHONPATH resolves ABI/import problems; no installation. Extra CPU assertions initially guessed a keyword-only signature, routing dict keys and `next()` instead of native loop fallthrough; inspected source and then executed its AST. Checkpoint records the retry correction; no persistent test/helper was added.

One faulty worker-entry CPU control raised Exception, which worker caught and then attempted a **local** /mnt/codesign-exp mkdir; PermissionError, no successful cloud-path write. Reported immediately. Corrected BaseException+audit control proves the boundary without entering the handler. No successful cloud read/write, GPU/platform/network/Docker/seal operation occurred.

Only the existing rank0 grouped-GEMM envelope is a candidate scope. Gate M is not native L_d; shuffle overlaps grouped-GEMM. M6152 is conditional mixed-prefill/dummy evidence. Random packed weights, physicalTP1/rank0, and independent CUDA_EVENT/KERNEL_ONLY evidence establish no rank/time/weight/graph/serving equivalence or admission.

Root still owes independent reviews; actual locked-image source440/binary20 pins and native selector/MARLIN capability gates; actual SDK non-preemptible/resource/mount/auth/quota/cloud-destination checks; external seal; then the bounded1H800/1800s collection and observed four rows/method, all original columns, retained traces and terminal status. Existing root balanced_decode/root_seal.json is the reference procedure; its receipts are not reused as MoE runtime qualification.
