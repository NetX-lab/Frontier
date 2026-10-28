# Test report: B12 and B12b, the G3h MoE producer (design entry 16, CPU)

## Modification History

| Date | Change |
| --- | --- |
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
