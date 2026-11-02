# Test report: B12 and B12b, the G3h MoE producer (design entry 16, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Added the G3h run on 1 H200 and its kernel-identity check against the native P2 traces. |
| 2026-10-28 | Created: lead review and rerun before the B12 + B12b commit. |

## Scope

- B12 (Codex `68caa2f4`, [h3o_operator_costs/codex_assignment_B12_g3h_moe_rank_classes.md](h3o_operator_costs/codex_assignment_B12_g3h_moe_rank_classes.md)): the rank-class planner (`frontier/profiling/moe/moe_rank_plan.py`), the class algebra in `frontier/moe_ep_workload.py`, per-rank grouping in `main.py`, the gate on SP-local rows L, the row identity and the G3h package plan.
- B12b (Codex `15de688d`, [h3o_operator_costs/codex_assignment_B12b_g3h_native_gate_and_ids.md](h3o_operator_costs/codex_assignment_B12b_g3h_native_gate_and_ids.md); report `h3o_operator_costs/codex-implement-15de688d-ce25-43c9-8ec1-fe214a5ce2e1/implementation.md`): the native V4 `GateLinear(out_dtype=float32)` on the `balanced_topk` path, the routed id dtype per routing path (`balanced_topk` int64, the other paths int32), the restored `uniform_topk` body, the class columns on balanced rows only, single checks, and the G3h notes ([h3o_operator_costs/g3h/native_kernel_notes.md](h3o_operator_costs/g3h/native_kernel_notes.md)).

## Lead correction

B12b carried the routing path twice: in the wrapper (`self.routing_runtime_path`) and as an extra field of the prepared routing sample, read with `routing_inputs.get("routing_runtime_path", "standard_fused_topk")`. Production samples always come from `_prepare_routing_inputs`, so the default served only tests that build a wrapper without its constructor. The lead removed the sample field and the private-method parameter; `_profile_with_vllm_kernel` now passes `self.routing_runtime_path` to `profile_fused_moe_kernel`. Two constructor-bypassing fixtures now set the path of the model they represent (`test_glm_moe_profiling.py`: `grouped_topk_sigmoid`, the GLM MoE DSA profile; `test_moe_native_admission.py`: `standard_fused_topk`). The forwarding test now checks the kernel call.

## Environment

- Working root `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD `05ec172c` with the B12/B12b changes in the tree.
- `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6); CPU torch 2.1.0 from `/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages` on `PYTHONPATH` after `$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`; `ulimit -v 31457280`. Scratch `/data/ycfeng/tmp/b12b_review/`.

## Lead rerun (after the correction)

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Required suite: `pytest -q -ra -p no:cacheprovider` on `test_dsv4_moe_profiling.py test_balanced_routing.py test_moe_profiling_output_metadata.py test_moe_load_distribution_contract.py test_moe_routing_input_contract.py test_moe_routing_runtime.py test_moe_gating_constructor_boundary.py test_moe_ep_workload_materializer.py test_moe_rank_plan.py test_moe_rank_routing.py test_moe_native_v4_gate_and_ids.py` | no failure | 325 passed, 2 skipped (installed-vLLM checks) | PASS |
| 2 | Importing suite: the 59 modules listed in Codex's `importing_tests_final.json` | no failure | first run 2 failed (the two constructor-bypassing fixtures above); after the fixture fix 1,702 passed, 24 skipped (5 installed-vLLM, 19 retired PD-AF contracts) | PASS |
| 3 | Codex CPU verifier, copied to `/data/ycfeng/tmp/b12b_review/verify_cpu_final.py` with the two fixture files added to its file list | H800 grid dry runs equal to `moe/h800_moe_local_01/plans/{cuda_event,record_function}.json`; `g3h/prepare_package.py --check-plan` 13 EP16 and 20 EP32 tasks per method; cold imports without torch, vLLM or CUDA; no unused imports; `git diff --check`; `uniform_topk` and the gate constructor calls equal to HEAD; frozen `balanced_routing.py` SHA-256 `6562ec1c…f4ea` | all PASS; 4 plan steps per method, no changed plan field; 530 base-package inputs verified; 19 files without unused imports; largest changed module `moe_vllm_kernel.py` 1,279 lines | PASS |

Codex-reported checks not rerun by the lead: the source comparison of the native gate dispatch (`router/gate_linear.py:183-232`) and the trace of every routed-id consumer against `evidence/runtime_census/p2_32.txt` (alignment and sorting `<long>`, reduction `moe_sum_vec_kernel<c10::BFloat16,long,6,true>`, Marlin GEMMs read the int32 sorted ids).

## Limits

- CPU evidence only. GPU kernel names, cuteDSL availability on the worker image, and zero-local-class launches are not yet observed; G3h checks them against the native census.
- The standalone shuffling scope keeps its own block-size rule and `ignore_invalid_experts=False`, while native grouped Marlin aligns inside the grouped envelope with `ignore_invalid_experts=True`. The consumer must not charge the shuffling row as a native cost outside the grouped envelope.
- `tests/unit/test_profiling_governance_minimal_red.py` is 2,071 lines before and after; B12 replaced 2 lines in it and did not refactor it.

## G3h run on 1 H200 (after the commit)

Submitted through /ds4-exec (Pi `g3h-submit-1028g`, report `/data/ycfeng/tmp/ds4_runs/g3h-submit-1028g/output.md`) from the sealed package (manifest SHA-256 `711b70a0…8a8e`, 1,024 files, all hashes verified before submission).

| Item | Expected | Actual | Result |
| --- | --- | --- | --- |
| RJob | Succeeded | `exp-1029-010008-493772`, 1 H200, `step_main`, non-preemptible; both phases (`G3H_MOE_CUDA_EVENT`, `G3H_MOE_RECORD_FUNCTION`) succeeded; no traceback | PASS |
| Worker runtime | H200, vLLM 0.28.0 | NVIDIA H200, vLLM 0.28.0 | PASS |
| GPU time | within the G1h-G4h cap of 7 | phase runtime 132.8 s = 0.0369 GPU-hours | PASS |
| Copy | equal to the cloud tree | 112 files, 26,895,108 bytes, every size equal; `/data/ycfeng/tmp/glm53_dsv4/h200/g3h_01/download/` | PASS |
| Tables | 13 EP16 and 20 EP32 rows per method | CUDA_EVENT 13 / 20, RECORD_FUNCTION 13 / 20; 99 kernel traces | PASS |
| Backends | native V4 gate and grouped Marlin | `vllm_gate_linear`, `MARLIN`, `vllm_marlin_mxfp4` | PASS |

### Kernel identity against the native P2 traces (CPU, lead)

Scripts and outputs are kept in `h3o_operator_costs/evidence/g3h/`; scratch `/data/ycfeng/tmp/g3h_accept/`.

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `trace_kernels.py <G3h RECORD_FUNCTION profiler_traces>` → `trace_shapes.txt` | kernels per annotated scope and row | one multiset per row and scope | done |
| 2 | `native_moe_kernels.py <P2-32 torch_traces>` and `<P2-64 torch_traces>` → `p2_32_moe.txt`, `p2_64_moe.txt`, `p2_32_moe_fill.txt` (dispatch segment of the TP0 trace of each lane, the census segmentation) | native routed-MoE and gate kernels per forward | per forward 43 each of `moe_align_block_size_kernel<long>`, `count_and_sort_expert_tokens_kernel<long>`, `_swiglu_limit_pad_aware_kernel` and `moe_sum_vec_kernel<c10::BFloat16, long, 6, true>`, at least 43 `FillFunctor<int>` grid 1, and 86 Marlin GEMMs | done |
| 3 | `compare_shapes.py` → `cmp_ep16.txt` (P2-32 / EP16), `cmp_ep32.txt` (P2-64 / EP32) | every native (name, grid) appears in G3h | "native shapes not in G3h" empty for both; 13 and 20 shared shapes | PASS |

Details:
- Each G3h grouped-GEMM call launches the native per-layer set of 7 kernels: `FillFunctor<int>` grid 1, `moe_align_block_size_kernel<long>`, `count_and_sort_expert_tokens_kernel<long>`, 2 Marlin GEMMs (grid 528 on EP16, 264 on EP32), `_swiglu_limit_pad_aware_kernel` and `moe_sum_vec_kernel<c10::BFloat16, long, 6, true>`. The `<long>` kernels confirm the int64 routed ids on the `balanced_topk` path.
- Gate: rows with L ≤ 4 launch the cuteDSL `ll_bf16_dotprod` kernel at grid 256; L = 768 launches `nvjet_sm90_tss_64x32_64x16_2x4_h_bz_TNT` at grid (8, 12, 1). Native uses the same two kernels: cuteDSL in decode and dummy forwards, nvjet in the 6144-token prefill.
- Shapes only in G3h are rows that the native runs did not observe (other grids of the same kernels) and the balanced-routing top-k kernels; the 4 `CUDAFunctorOnSelf_add<float>` kernels per trace are harness work outside the scopes.
- Zero-local rows (EP16 rank 6 with 16 tokens and rank 12 with 32 tokens; EP32 rank 24 with 32 tokens) launch all 7 kernels; their grouped-GEMM median is 0.0118-0.0120 ms, against about 0.101 ms (EP16) and 0.062 ms (EP32) for non-zero rows of the same token count (`rows.txt`).
- The RECORD_FUNCTION gate rows hold one traced call each (no variance estimate); every CUDA_EVENT row holds 20 measurements for each of the four operators.

Limits of the run: `weight_source=dummy`, `gating_runtime_context=standalone_legacy`, `graph_mode=eager`. The standalone shuffling limit above still applies.
