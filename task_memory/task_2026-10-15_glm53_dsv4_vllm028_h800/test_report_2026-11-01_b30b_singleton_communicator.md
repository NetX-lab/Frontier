# B30b singleton communicator metadata repair

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Archived exact-job RCA coverage: six successful reads and two timed-out all-operation audits. |
| 2026-11-01 | Preserved links/KV02 failure evidence, verified the cause-specific repair, and prepared collectives-only generation 03. |

Tickets 06/09/11/19; decision 46 and D-CODEX-DIRECT-RJOB authorize the diagnostic probes and their ordinary in-scope repairs.

## Observation and cause

RJob `exp-1101-004206-443314` reached Failed at 2026-10-31T16:49:51Z. Both containers exited 1 without restart. Both nodes passed 22/22 source/runtime checks and physical GPU UUID selection. A_intra, B_same_rail and B_ring_boundary each exited 1 on both nodes; C_nixl exited 0 on both nodes. Exact platform and selected cloud records are in `h3o_operator_costs/b30b/execution_20261101/`.

The first actionable collective traceback is `AttributeError: 'NoneType' object has no attribute 'pynccl_comm'` in `CollectiveRunner.communicators()`. Native `vllm/distributed/parallel_state.py` constructs a device communicator only when world_size > 1. A_intra has singleton DP; the two B phases have singleton TP. Reporting these legitimate groups must record disabled communication backends. Peer-close errors are consequences. Later worker cleanup does not establish an external initiating stop.

## Code change and verification

`frontier/profiling/collectives/vllm_collective_benchmark.py` checks for an absent device communicator while recording backend metadata. The existing communicator test now includes a singleton DP group and verifies every enabled field is false. Collective timing, routing, simulator code and formal tables are unchanged. Source edits began after the locked GPU run terminated.

Python: `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6. CWD: the current worktree. Environment: PYTHONPATH=$PWD, PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.

- RED: the extended existing communicator test on the old source: 2 failed, 16 deselected, 0.34 s; same AttributeError.
- GREEN: `python -m pytest tests/unit/test_b30b_ex_situ_probes.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_vllm_dsv4_forward_collectives.py tests/unit/test_vllm_forward_collectives.py tests/unit/test_collective_kernel_evidence.py task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/tests -q`: 167 PASS, 6.42 s. Expected: singleton metadata is valid and existing multi-rank metadata/probe contracts pass.
- Logs: `/data/ycfeng/tmp/b30b_singleton_{red,green}_20261101.log`; XML: `/data/ycfeng/tmp/b30b_singleton_green_20261101.xml`.

## Next verification

Generation `stage_b30b_links_03.json` retains the three unmeasured collective phases and excludes successful C_nixl. It uses two 8-GPU H200 nodes with a 3,900 s timeout. The unchanged planner supports configurations without NIXL; dry-run, current quota, immutable source lock and fresh cloud output checks precede submission. NIXL CSV/JSONL verification and fitting proceed from generation 02. F2 is not resubmitted. CPU success does not establish GPU measurement success.

## Generation03 submission

Planner: `/data/ycfeng/envs/frontier-py310/bin/python h3o_operator_costs/b30b/planner.py --config h3o_operator_costs/b30b/stage_b30b_links_03.json --output h3o_operator_costs/b30b/dry_run_03` (paths relative to this task; PYTHONPATH is the repo). PASS:548 case instances; estimated1,677.589s/7.455951GPUh, timeout3,900s/17.333333GPUh. The NIXL phase is absent as intended.

At2026-10-31T16:56:37Z, the complete personal RJob listing found zero non-preemptible reservations. The quota CRD read returned403; it is preserved as an unavailable query, not treated as zero capacity. The fresh personal cloud destination was absent. Codex directly launched the existing `g5h/launch/launch_g5h.sh` through Python RJobBackend at16:57:41Z as `exp-1101-005741-958298`, sourceac0f908a (contains repair662a427e). Initial creator,16H200,non-preemptible/step_main,image,RDMA,and local NFS100.96.128.195 were verified; Starting with two replicas. Launcher session67965/PID495129 remains alive. Receipt and exact commands: `h3o_operator_costs/b30b/execution_links03_20261101/`.

## Exact-job RCA coverage

The bundled RCA collector completed with exit 1 for `exp-1101-004206-443314`, replica `exp-1101-004206-443314-cfcd2084`, window 2026-10-31 16:45:00–16:50:00 UTC. Python 3.10.6 / brainpp 2.8.6, personal identity `i-fengyicheng`. Six read families succeeded: live and historical events for job and replica, incident logs, and existing DeepScan (zero existing flows, complete pagination). Both all-operation audit queries failed with `ReadTimeout`, 15 seconds. They are unavailable evidence, not empty audit results. Stop actor/reason remain UNKNOWN; the original application traceback and phase returncodes already establish the collective metadata failure.

Exact source paths, query timestamps, identities, errors and hashes are retained in `h3o_operator_costs/b30b/execution_20261101/rca_coverage.json`; raw collector output remains `/data/ycfeng/tmp/glm53_dsv4/b30b_links_kv02_rca_20261101/`. No additional platform repair or job restart follows from those audit timeouts.
