# Locked-image CPU DP sequence control

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | W128-DP-SEQUENCE-CPU: recorded locked-image WS2 Gloo getter availability, symmetric counters, source binding and an unimplemented metadata proposal. |

## Outcome and scope

**PASS for this CPU control only.** The real ProcessGroup `_get_sequence_number_for_group()` is callable in the locked image. Both ranks returned identical strictly increasing post-collective sequences for six blocking all-reduces, including three entirely zero-valued tensors. No CUDA initialization occurred.

Only this report and temporary scripts/evidence were written. No production/test source or Git writes, native worker execution, GPU/platform/StepMind/cloud operations, image pulls, installations or external network requests were performed. Gloo used container loopback inside `--network=none`.

## Exact environment and execution

- Image, local ID and local manifest descriptor digest: `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`; Docker architecture `amd64`, OS `linux`.
- Python `/usr/bin/python3`: `3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`; Torch `2.13.0+cu130`, CUDA build `13.0`, Torch Git `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Image native build label: `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Actual installed source, not that label alone, supplies the source binding below.
- Evidence directory: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-cpu-I0HAIn` (called `E` below).
- Exact host command: `timeout 60s python3 /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-cpu-I0HAIn/run_probe.py`.
- `E/run_receipt.json` preserves the full Docker argument array, UTC times, script/artifact hashes and exit code. Docker used `--pull=never --network=none --runtime=runc --memory=6g --memory-swap=6g --cpus=2 --pids-limit=128`, no GPU device flags, and only `E:/evidence:rw` as its host mount. The container command was `/usr/bin/timeout 45s python3 /evidence/dp_sequence_probe.py`; host subprocess deadline 55s.
- Environment: `NVIDIA_VISIBLE_DEVICES=void`, empty `CUDA_VISIBLE_DEVICES`, `GLOO_SOCKET_IFNAME=lo`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/evidence`, `TORCH_DISTRIBUTED_DEBUG=OFF`.
- Observed UTC interval `2026-10-23T18:14:53.030512+00:00` to `18:15:02.626878+00:00`; Docker exit `0`, elapsed `9.596067756880075s`, child exit codes `[0,0]`. This is the receipt clock, not an inferred task date.

## Raw counts and acceptance

The one actual group was `torch.distributed.distributed_c10d.ProcessGroup`, name `0`, backend `gloo`, ranks `[0,1]`, world size `2`. FileStore rendezvous created the default WORLD group; every reduction explicitly passed that group and `async_op=False`, SUM, CPU int32 shape `[4,2]`. This was not a native DP subgroup or a serving worker.

| Rank | Immediately after group startup | Before each collective | After each collective | Terminal |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0,1,2,3,4,5 | 1,2,3,4,5,6 | 6 |
| 1 | 0 | 0,1,2,3,4,5 | 1,2,3,4,5,6 | 6 |

Startup offset was observed as `0` on both ranks; do not assume that offset for native startup. Zero-valued tensors occurred at zero-based collective ordinals `0,2,5`, with after-counters `1,3,6` on both ranks. At ordinal 3 only rank 1's token row was zero. Full local operands and reduced matrices are in `E/rank0.json`, `E/rank1.json` and `E/result.json`.

Expected/actual PASS conditions: getter callable on both ranks; six equal after-counter pairs; strict increase at every call; all 12 reduced matrices equal their explicit expected matrices; all 12 native return values were `None`. `torch.cuda.is_initialized()` was false before/after each rank's workload, after every collective, and before/after the parent workload. Instrumented `_lazy_init` had zero calls in all three processes; `/dev/nvidia*` was empty. CUDA libraries imported by Torch are not evidence of CUDA device initialization.

## Source locations and minimum proposal (not implemented)

- Actual image source: `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/dp_utils.py`, SHA256 `3c882f85109ba47e473953351d166c4377ceb995205e91cbf128ca5075775a5d`, byte-for-byte copy `E/locked_native_dp_utils.py`. The CPU group is selected at line 33, tensor constructed at 34–38, blocking `dist.all_reduce(tensor, group=group)` at **39**, all-zero-token return at 46–50 and padding mutation at 92–93. The earlier line-41 location does not describe these actual image bytes.
- Worktree source-location binding was recorded at HEAD `5f7d0950b78276a77a66b7330cf0041a9f380f12`; four inspected file hashes are in `E/source_locations_binding.json`.
- Existing seam: `tests/comparison/calibration/native_full_structure.py:516–586`, especially native call 570 and post-call snapshot 572–580. `observe` at 378–383 catches observation exceptions; native all-reduce exceptions occur outside it and remain original exceptions. `FullModelInventory:269–275` already records role-relevant worker/group membership.
- Smallest candidate: immediately after the existing CPU-vector snapshot, separately call `observer.observe('dp_post_all_reduce_sequence', ...)`, reading `kwargs['group']._get_sequence_number_for_group()` and recording the raw integer as `dp_post_all_reduce_sequence` in the existing active dispatch state. Do not add a collective, infer a default group, subtract startup offsets, reset the ProcessGroup, or call this a step index. Missing/throwing getter must log a soft observation error and leave this key absent; preserve the already-copied vector, native return and native exceptions. Duplicate snapshot errors remain soft.
- `tests/comparison/calibration/native_serve_profile.py:141–150,481–491` already carries/detaches dispatch metadata into per-step rows. Its local `step` remains a local collector index. A candidate join is scoped by run generation, role and ordered DP group members from existing inventory, then raw sequence; group name `0` alone is not a global identity.
- **Implementation decision still required:** frozen `task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py:142–147` requires exactly the three existing dispatch keys. An extra key is intentionally rejected. Do not relax this checker or drop its tests: a separately authorized diagnostic-envelope placement or schema successor is needed before implementing the candidate. This control does not grant that authority.

## Tests to extend from actual existing stand-ins (not executed or edited)

1. `tests/unit/test_native_v2_serving_inventory.py:533–578` already verifies vector capture before padding, return identity, untouched transport attributes and identical native exceptions. Its oracle currently uses `get_dp_group().cpu_group=None` (`check_groundtruth_run_successor_r3.py:125–138`). In this test's local oracle namespace, replace only that transport carrier with a fake group getter; advance its counter only when the reduce stand-in completes. Assert the exact same group is read after reduction and vector capture remains unpadded.
2. The simpler V2 fixture at `test_native_v2_serving_inventory.py:100–118` currently omits `group=` entirely. Make that CPU stand-in pass an explicit fake group as actual image line 39 does, rather than introducing production default-group handling. Use a nonzero startup counter, repeated/zero-valued vectors and successive reductions to assert raw increasing values, not tensor-derived or local-step-derived indices.
3. Add missing-method and throwing-getter controls to that existing stand-in: successful native call still returns its original object, dispatch vector remains available, the sequence is absent, and bounded observation errors identify the sequence read. Retain the existing native-failure identity and inactive-dispatch controls; no getter read is needed outside the active dispatch scope.
4. `tests/unit/test_native_serve_profile.py:344–363` already asserts profiler/inventory dispatch equality without another collective. Extend the chosen authorized metadata placement's corresponding copy assertion, plus detached-copy coverage at 232–243. Preserve the exact-schema frozen-checker regression rather than weakening it.

## Artifacts and inference limits

`E/run_probe.py` SHA256 `8b56fbd30d7ac2da25672fcabc0db2f45347b2b271aadede271e4976849ac19d`; `E/dp_sequence_probe.py` SHA256 `81fe442ddaec1d928f58d34be13d0c4d387b7c3d7328218cb5210f807d0a92a8`.

`E/result.json` SHA256 `75b6600f727751a2000299a2b18c1fda2c25bba29737bfc2f1253246d24ae53f`; `E/rank0.json` `388e08e299dfb27596b7bd46bb570722044b3f7d985019874ed308541c494022`; `E/rank1.json` `5b88afeb4d4ec507f076d8d97350ac314ead7c2cee81073f5ba073c46db7ebd4`; raw `E/docker.log` `f67fc280b84c47805dba32e251571c1bdf33b04acff29c90c52c695d283e95f9`. Metadata/image inspection hashes are retained in the run receipt.

Final artifact verification: `timeout 60s python3 /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-cpu-I0HAIn/verify_delivery.py`, host Python `3.10.6`/GCC `11.3.0`, **60/60 PASS**. `E/delivery_verification.json` records each expected/actual check and the final report hash/line count. The self-check in `E/check_progress.md` found no route drift and required only final artifact verification and delivery, not further runtime work.

The CPU control closes runtime getter availability and WS2 symmetry **for the measured group only**. It does not establish full native execution, native startup offsets, no-other-collective behavior, cross-host/TP8/DP2/EP16 qualification, dense native step numbering or historical profile02 recovery. Repeated vectors can identify different collectives, but native pairing still requires fresh same-seam metadata and scoped runtime verification. Proposed source tests and metadata/schema decisions remain unfinished and belong to the primary owner.
