# Task: Pair-warmed formal P2-64 native package and its post-run tools (Q5)

## Goal

Prepare a new formal DSV4 P2-64 native reference run on H200, ready for the lead to submit.

- Topology: P32+D32, TP8/DP4/EP32/PP1 per role, 8 nodes × 8 H200 in `step_main`.
- What makes this run formal: ticket 19 needs a 30% promotion receipt that exists before a formal 64-GPU native submission. The existing P2-64 reference `exp-1025-155441-668927` was launched before any receipt existed. YC chose to run a new campaign (D-P2-64-FORMAL-B in `requirements.md`).
- The receipt is already issued and committed (`1a7950ed`): `h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json`, SHA-256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`.

The new run must differ from the retained P2-64 reference package (`h200_mainline/p2_64_reference/package_01`) in only three ways:

1. **Pair warmup.** In each repetition's warmup, every (P lane, D lane) pair opens its lazy NIXL connection. At DP4 × DP4 there are 16 pairs. D-WARM-TARGET says: "Future native runs warm every lazily connected communication pair before the formal window and prove coverage from lifecycle/server records; a fixed warmup count is not evidence of coverage."
   - The retained run's 16 untargeted warmups left 5 pairs unopened. Their first contacts (1.3–1.9 s handshakes) landed on formal repeat01 requests r006/r017/r027/r035/r036.
   - Formal requests keep vLLM's own DP routing.
   - The workload stays unchanged: 3 repetitions × (16 warmup + 128 formal requests), 6144 prompt / 1024 output tokens, QPS 1. Both roles are idle and request-owned KV is released before and after each formal window.
2. **Resource class.** Wholly preemptible: `ResourceConfig.preemptible=True`. Today `submit_native_diagnostic.py` never passes this field.
3. **Receipt binding.** The package binds the issued receipt by path and SHA-256, and refuses to launch (including `--check-only`) when the receipt is missing, its hash differs, or its `issued_utc` is not earlier than the build.

Everything else stays byte-identical to the source package: engine, image, NATIVE `2cf0a691`, C26 overlay, dummy weights, `balanced_topk`, model metadata, trace, timeouts and observers. A new run tag, job tag and output directory are needed, for example `dsv4-h200-p2-64-formal-20261102-01`.

## Deliverables

1. **Harness change, uncommitted.** Make it in the prepared worktree `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`: branch `p2-64-pair-warmup` at `b62c155a`, the source package's harness commit, clean at start.
   - Warmup requests are assigned one-to-one to (P lane, D lane) pairs derived from the configured DP sizes; no hard-coded 16. If a repetition has fewer warmups than pairs, it fails with a clear message.
   - Formal requests carry no target.
   - The client and proxy records keep each warmup's intended pair. The coverage check compares these intended pairs with the pairs actually observed.
   - A likely mechanism is vLLM's `X-data-parallel-rank` request header, sent with a different value to the P and D servers. The pinned native source reads it at `vllm/entrypoints/generate/base/serving.py:206-211`, and completions pass it to the engine at `vllm/entrypoints/openai/completion/serving.py:155`.
   - Confirm from the native source and the package's server commands that this deployment's DP mode honors the header on both roles. If it does not, propose another route that leaves formal routing native.
   - Also check whether targeted warmups change any routing state that formal requests see after the idle boundary, compared with untargeted warmups.
   - Check the handshake cache lifetime (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:508-509` defaults the remote-engine TTL to 3600 s) against the run length.
   - Add focused tests in that worktree. The lead reviews and commits it.
2. **Builder case.** Add a case, for example `p2_64_formal`, to `h200_mainline/build_package.py`.
   - Source: `p2_64_reference/package_01`. Harness: the new worktree. Changes: only the three above plus tags and paths.
   - The non-preemptible limit recorded for the case is the standing rule of 32. A wholly preemptible 64-GPU job is outside that limit.
   - Update `evidence_readiness.json` cost fields and the `manifest.yaml` policy (`preemptible: true`) consistently.
   - The builder's file comparison must still fail on any other difference.
   - Because the harness is not committed yet, verify with a dry build in scratch, in the way the `p2_64_wave` case's `--dry-check` does: package file comparison, the offline 8-host launch plan and the receipt checks. After review the lead commits the harness and runs the real build: seal, preflight run manifest, R3 reconstruction, `launch.py --check-only`.
3. **R4 compatibility.** The reference is admitted with the R4 checker `task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py` (R3 plus one replacement).
   - Establish whether any R3/R4 predicate depends on what the new package changes: warmup records, proxy records, the harness commit or file hashes, or the resource class.
   - If a predicate would fail only because of the change, report it with evidence and a proposal. Do not edit the checker.
4. **Coverage check (post-run, soft gate).** A CPU script that reads a local copy of a finished run and proves, per repetition:
   - every pair's first contact falls inside a warmup window;
   - no formal request opens a pair;
   - each warmup reached its intended pair;
   - the idle boundaries are verified.

   Reuse the event-binding method of `h3o_operator_costs/warm_target_20261101/identify_openers.py` (decode request ownership, the remote prefill engine from proxy records, TP-worker compatibility-log cohorts). Its output must be usable as the `--openers` input of the repository's `tests/analysis/e2e_warm_reading.py`.
   - It reports COVERED / NOT_COVERED with evidence and never raises into a caller that could stop a job.
   - **Control:** on the retained P2-64 reference it reproduces the five repeat01 formal openers of `warm_target_20261101/openers.json` and reports which pairs the warmups did open. On the retained P2-32 reference it reports zero formal openers.
5. **Comparison path (post-run).** Scripts that take a hash-identical local copy of the new run and produce, against the unchanged formal P2-64 Frontier replay (`/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64`, command record `h3o_operator_costs/formal_p2_20261102/p2_64_command.json`):
   - the R4 result;
   - the persisted native inputs, in the layout of `h200_mainline/diagnostic/inputs/p2_64r`;
   - the all-request gate reading;
   - the warm reading, using the coverage-check openers.

   The lead makes the copy with `h200_mainline/cloud_copy.py` and the terminal receipt with `terminal_receipt.py`, through `/ds4-exec`.
   - Reuse `h3o_operator_costs/a10/a10_gate.py`, `h4a_p0_diagnostic/h4a4_gate.py` (`native_layout`, case table), and, in the `glm53-dsv4-vllm028` repository, `tests/comparison/e2e_gate/run_gate.py` and `tests/analysis/e2e_warm_reading.py`. Add a new case entry or wrapper in the output directory; do not edit retained scripts.
   - **Control:** run the whole path on the retained P2-64 reference and reproduce `h3o_operator_costs/formal_p2_20261102/readings/p2_64_{warm,empty}_reading.json` and the gate values in `h3o_operator_costs/formal_p2_20261102/assessment/gate_readings.csv` exactly (provenance paths may differ).
6. **Cost and resources.** From the retained platform receipt of `exp-1025-155441-668927`, estimate container GPU-hours, then state:
   - the timeout ceiling;
   - the resource request (8 replicas × 8 H200, `step_main`, preemptible);
   - any change in run length caused by the targeted warmup.

## Scope and context

- Read first:
  - `requirements.md` sections D-P2-64-FORMAL-B, D-WARM-TARGET, D-P2-INPUT-QUALIFICATION and the 2026-10-21 resource policy (all H200 64-GPU jobs wholly preemptible; interrupted preemptible runs restart as a new clean generation; fragments never form a reference);
  - ticket `tickets/19-comparison-and-acceptance.md` (`promotion_32_to_64`, `final_acceptance`, Acceptance);
  - `h200_mainline/design.md`;
  - `test_report_2026-11-02_formal_p2_assessment.md`;
  - `warm_target_20261101/test_report_2026-11-01_warm_reading.md`.
- Soft-gate rule for runs of 32 or more GPUs: any agent-written checker or collector is non-blocking. It isolates its own errors, logs evidence, marks artifacts unverified, and never stops or fails serving. Native serving and I/O failure handling stay as they are.
- Do not edit:
  - `frontier/`, `data/`, the trace, thresholds, the receipt, retained packages and run copies, the R3/R4 checkers or the Q1 ledger;
  - the formal replay outputs or their command records.
- No RJob, platform, cloud-volume or GPU action. Do not commit.
- Paths in this brief are relative to the task directory `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/` unless absolute. Untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` in the `glm53-dsv4-vllm028` worktree are not yours.
- Writable:
  - `/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup/`;
  - `h200_mainline/build_package.py`;
  - `h3o_operator_costs/formal_p2_64_native_20261102/`, with scratch in its `scratch/` subdirectory or your private temporary directory.
- Environment:
  - Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6);
  - `PYTHONDONTWRITEBYTECODE=1`; run heavy reads under `timeout`.
  - The native vLLM checkout is `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Read its files directly: `git log` there fails with a missing-object error.

## Expected return

- `formal_p2_64_native_20261102/report.md` in English, plain words, actual values. It covers deliverables 1–6.
- For each check: the command, environment, expected and actual result, and PASS/FAIL.
- The harness diff summary and the files changed.
- The exact commands the lead runs after committing the harness: real build, `--check-only`, and the post-run sequence.
- Every open question or blocker, with evidence and a recommended action.
