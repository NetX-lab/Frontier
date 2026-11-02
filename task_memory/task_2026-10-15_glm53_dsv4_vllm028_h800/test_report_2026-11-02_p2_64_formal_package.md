# Test report: formal P2-64 native package (D-P2-64-FORMAL-B), lead review and build

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Created: review of Codex Q5, harness commit `4861e680`, builder import-cache fix, real build of `p2_64_formal/package_02`, independent checks before submission. |

Authority: requirements D-P2-64-FORMAL-B, D-WARM-TARGET, D-P2-INPUT-QUALIFICATION, the 2026-10-21 resource policy and D-NO-INTERRUPT; ticket 19 (`promotion_32_to_64`, Acceptance).

Assignment: [Q5](h3o_operator_costs/formal_p2_64_native_20261102/codex_assignment_Q5_p2_64_formal_package.md). Codex request `54cbb041-c825-4706-b25e-75530b995d23` (default model and effort, outcome `implemented`), report [implementation.md](h3o_operator_costs/codex-implement-54cbb041-c825-4706-b25e-75530b995d23/implementation.md) and [formal_p2_64_native_20261102/report.md](h3o_operator_costs/formal_p2_64_native_20261102/report.md).

## What changed

- **Harness** (`/data/ycfeng/Frontier/.worktrees/p2-64-pair-warmup`, branch `p2-64-pair-warmup`, commit `4861e680` on `b62c155a`):
  - `pd_replay.py`: `--pair-warmup` assigns the warmups of each repetition to the (P rank, D rank) pairs derived from the configured DP sizes. It fails if there are fewer warmups than pairs, and refuses a target on any formal request.
  - `vllm_replay.py`: sends proxy-only target headers for warmup requests; formal headers are unchanged.
  - `pd_proxy.py`: forwards a separate `X-data-parallel-rank` to the prefill and decode servers for targeted warmups, and records `intended_pair`.
  - `native_pd_worker.py`: `validate_promotion_receipt` (hash, issued before build, predicate true), called before the pre-launch manifest; the manifest records `preemptible`.
  - New `tests/unit/test_native_pair_warmup.py` (12 tests).
- **Builder** (`h200_mainline/build_package.py`): case `p2_64_formal` from `p2_64_reference/package_01`.
  - It restages the four harness modules, inserts receipt validation before `--check-only` in `launch.py`, passes `preemptible` to `ResourceConfig`, updates the readiness cost fields and the manifest policy, and fails on any other byte.
  - Lead fix: `build_formal` calls `importlib.invalidate_caches()` after writing the package. Without it, the real build failed (see below).
- **Post-run tools** (`h3o_operator_costs/formal_p2_64_native_20261102/`): `coverage.py` (pair coverage, soft gate), `compare.py` (hash-checked copy → R4 → persisted inputs → gate and warm readings), `verify_control.py`, `test_tools.py` (25 tests).

## Review findings and decisions

1. **The DP rank header is honored on both roles.** Pinned source: `serving.py:206` reads the header, completions pass it on (`completion/serving.py:155`), and `core_client.py:1473` uses an explicit rank before load balancing. Both role heads run internal DP4. Lead read of `core_client.py:1471-1521` confirms this.
2. **Routing state seen by formal requests.** Each role head runs four API processes (`cli/serve.py:103-125`: internal-LB default `api_server_count = data_parallel_size`). Each process has its own tie-break scan index (`eng_start_index`), which rotates only for untargeted requests.
   - Targeted warmups therefore leave the indices where they were. Before repeat01 that is the fresh-server value `client_index`; before repeats 02 and 03 it is wherever the previous formal window left it.
   - In the retained run the starting indices were also uncontrolled, because the kernel hands incoming connections to the four processes unpredictably.
   - The formal routing algorithm and its inputs are unchanged, and formal requests carry no target.
   - **Lead decision:** this satisfies D-P2-64-FORMAL-B ("formal requests keep vLLM's own DP routing"). Exact per-request lanes were never fixed and are not an acceptance criterion. The Frontier replay does not consume native lane assignments; the R4 "routing-owner census" concerns the MoE router modules (`check_groundtruth_run_successor_r4.py:2-52`). No further design is needed.
3. **NIXL remote-engine TTL** is 3,600 s (`base_worker.py:508-509`); eviction needs more than 3,600 s without use. The worker timeout is 3,600 s, so connections opened in the warmup outlive the run.
4. **R3/R4:** no predicate fails only because of the change (Codex source analysis; the control below passes R4 unchanged). R2 platform checks do not read the preemptible label. `compare.py` additionally requires preemptible true in config and manifest, the receipt binding, and the platform label `yes`.

## Checks

Environment: local host. `/usr/bin/python3` 3.10.6 with torch 2.1.0 for the build (as the earlier package builds), `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6 for tools, `/data/ycfeng/envs/openmopd-py312/bin/python` 3.12 for aiohttp/HTTP tests and R4 (torch). `PYTHONDONTWRITEBYTECODE=1`, `ulimit -v 31457280` for build/control. Lead scratch `/data/ycfeng/tmp/glm53_dsv4/h200/q5_lead_review/`.

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Harness tests | `PYTHONPATH=$HARNESS py312 -m pytest tests/unit/test_native_pair_warmup.py tests/unit/test_native_pd_worker.py -q` | all pass | 28 passed | PASS |
| Tool tests | `py310 -m pytest test_tools.py -q` (in the tools dir) | all pass | 25 passed | PASS |
| Dry build | `build_package.py p2_64_formal --dry-check <scratch>/dry` | DRY_CHECK_PASS; only the declared files change | DRY_CHECK_PASS, receipt checks PASS, no other harness difference, 12 changed files; JSON leaf diffs only tags, paths, `pair_warmup`, `preemptible`, `promotion_receipt`, hashes, cost and purpose fields; manifest diff only purpose, generation, overlay hash, `preemptible: true`, non-preemptible limit 32; `launch.py` +2 lines; submit +1 line | PASS |
| Comparison-path control | `compare.py --control --r4-python py312 --output <scratch>/control`, then `verify_control.py` | R4 PASS; readings and 12 gate rows equal to the retained formal values | TOOLS_FINISHED, R4 PASS, errors []; all-request FAIL reproduced (repeat01 TTFT 12.254941%); PASS_EXACT | PASS |
| Coverage control | openers from the control run | the five retained repeat01 formal openers | r006, r017, r027, r035, r036 in repeat01; none in repeats 02/03; status NOT_COVERED (expected: legacy run has no intended pairs and five unopened pairs) | PASS |
| Real build, first attempt | `build_package.py p2_64_formal --harness $HARNESS` | build passes | FAIL at preflight: `ModuleNotFoundError: No module named 'native_pd_worker'` | FAIL, fixed |
| Cause | probe: add a missing dir to `sys.path`, import any unloaded module, create the dir, import from it | — | import fails until `importlib.invalidate_caches()`; the builder loads the worker (which imports `socket`, `importlib.metadata`, ...) before the package root exists, so Python cached the root as having no importer. The dry check never reached preflight. | diagnosed |
| Real build | `build_package.py p2_64_formal --package package_02 --harness $HARNESS` | seal, preflight manifest, R3 reconstruction, launch plan, check-only all pass | exit 0; 43 sealed files; manifest sha256 `c22085987055d018efb8abc104081cfe56d001b22026bb3aa367564d46754cd7`; harness `4861e680`, NATIVE `2cf0a691`; Frontier roles TP8/DP4/EP32/PP1 world 32 on both roles; check-only `INPUT_HASHES_PASS_NO_SUBMISSION` | PASS |
| Preflight run manifest | read `p2_64_formal_preflight_yj7kid0b/.../run_manifest.json` | clean harness, preemptible, receipt bound | harness `4861e680`, no dirty paths; resources 8×8 H200 `step_main`, timeout 3,600 s, `preemptible: true`; receipt sha256 `d505eece...f7596`, build_utc 2026-11-02T08:42:19Z (receipt issued 07:38:49Z) | PASS |
| Independent check-only | `python3 launch.py --check-only` in `package_02` | `INPUT_HASHES_PASS_NO_SUBMISSION` | same | PASS |

`p2_64_formal/package_01` is the partial output of the failed first attempt: sealed, but its preflight never ran. It is not committed, will not be launched, and is listed for deletion with YC's approval.

## Package to submit

- `h200_mainline/p2_64_formal/package_02`, run tag `dsv4-h200-p2-64-formal-20261102-01`.
- Cloud output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261102_01/case/runs/groundtruth_clean/dsv4-h200-p2-64-formal-20261102-01`.
- Resources: 8 replicas × 8 H200, `step_main`, wholly preemptible, timeout 3,600 s (ceiling 64 GPU-hours). Expected about 24.3 container GPU-hours, from the retained reference's eight containers (10,945 container-seconds).

## Limits

- The pair warmup and its coverage proof are untested on GPUs until the run finishes. Coverage is a soft gate and cannot stop the job.
- The new run's formal lane assignments will differ from the retained run (finding 2); this is expected.
