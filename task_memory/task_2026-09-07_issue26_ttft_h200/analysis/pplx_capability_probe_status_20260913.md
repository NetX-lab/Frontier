# PPLX capability probe status (2026-09-13)

- RJob `yc26-h200-pplx-capability-20260913-01` is terminal **Failed**; no new job submitted.
- `preflight.json`: `find_spec=true`; PPLX wheel import and custom op registration reached all ranks on H200.
- First failed operation: `AllToAll.intranode(...)` on ranks 0,2,3,5,6,7; rank 1 and 4 were terminated by torchrun.
- Error on every persisted rank: `RuntimeError: No backend type associated with device type cpu`.
- Traceback: `pplx_kernels/all_to_all.py:107` -> C++ `DistributedTorch::allToAllImpl`, which builds CPU tensors and calls `group->alltoall_base`.
- Cause: probe registered NCCL world group under `"default"`; NCCL cannot service CPU tensors. This is a probe setup error, not PPLX build/runtime unsupported.
- Evidence-only correction prepared in `pplx_capability_probe_20260913.py`: create `dist.new_group(backend="gloo")`; register that group as `"default"`; use it for metadata barrier; write rank-specific status files. Local Python syntax and `git diff --check` pass.
- Next action requires coordinator authorization/scheduling: rerun the same capability probe with corrected Gloo metadata group. Do not launch formal replay before this probe passes.
- Detailed report: `test_report_2026-09-13_pplx_capability_probe.md`.
