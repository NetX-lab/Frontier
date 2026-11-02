# Profile03 profiler/RPC failure verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Closed root full-metadata and native-statistics reruns; retained coordinator/schema failures and selected only the existing optional-table correction. |
| 2026-10-24 | Recorded the failed run, independently reproduced native CPU timeout controls and complete selected raw-trace censuses; actual-image native statistics-skip control passed. |

## Outcome and scope

The native run `exp-1024-141341-669621` is **FAILED**. These checks verify retained diagnostic evidence and a native timeout mechanism; they do not admit successful serving, operator costs, GPU dependencies or the original failure's exact pending-reply schedule.

| Check | Expected / observed | Verdict |
| --- | --- | --- |
| Complete artifact import | Exactly190 files/8208668144 bytes; every size and SHA-256 matches the remote inventory. | PASS |
| Request census | 432 recorded rows:144 submitted,137 completed including16 warmups,7 failed,288 not submitted. | PASS |
| Actual placement | Four distinct hosts0108/0329/0422/0618 and32 distinct H800 UUIDs. | PASS |
| Native CPU RPC symptom | Autostop301 seconds/300-second budget/unready reply produces exact `RPC call to execute_model timed out.` | Expected RED |
| Native CPU controls | No pause,299 seconds, ready at300, all replies ready after deadline, non-output pause without aggregator and FIFO controls succeed. | PASS |
| Selected prefill raw traces | Complete JSON/gzip for localrank0 of P0 and P1; hashes match retained inventory. | PASS |
| Parser cross-check | P0 validated JSON batches equal the original ijson receipt after excluding reader/clock/RSS fields. | PASS |
| Formal collector publication | All32 requested `.profile.json` absent; P0/D1 published snapshots are stale, not formal evidence. | FAIL completeness |
| Initial review | P2 D0 host-label defect corrected to0422; previous timeline bindings unchanged. | Closed |
| Primary census inspection |64 torch-trace directory entries contain32 gzip traces and32 text tables; corrected full190-file/32-rank audit is byte-identical to root rerun. | PASS correction |
| Actual-image native statistics toggle | Both branches export one trace/eight payload annotations; true invokes statistics/lazy events once, false invokes neither. CUDA uninitialized. | PASS CPU |
| Statistics-toggle RPC control | Same export cost, optional table true RED/false GREEN; export-too-slow and CPU-only tables remain failures; root final run matches all non-clock case fields. | PASS CPU |

## Exact execution and environment

Import receipt: `/data/ycfeng/tmp/glm53_dsv4/profile03_results_7dsz5a22/receipt.json`. Original artifacts are read-only analysis inputs. Timeline and metadata commands are in [command.md](command.md) and [metadata_command.md](metadata_command.md).

Root native-loop receipt: `/data/ycfeng/tmp/glm53_dsv4/profile03_rpc_root_jgnthas4/receipt.json`; source script SHA-256 `5608b2ec83b0744fa756ca203f8d01963b33516a6018d7adab1f5c1c37db80d9`. Python3.10.6/GCC11.3.0; actual native AST method bodies, virtual elapsed work and mocked transport, not parallel GPU workers.

```bash
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_rpc_loop_native_20261024/loop.py
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_rpc_loop_native_20261024/loop.py --symptom
```

Observed exit codes were0 and1 respectively. The symptom command's failure is required; successful execution is not substituted for the exact exception. Original repeated owner runs and native source/package-pin limits are in [rpc_loop_command.md](rpc_loop_command.md).

Actual-image receipt: `/data/ycfeng/tmp/glm53_dsv4/profile03_statistics_image_g8m3pvpv/receipt.json`; source stdin is preserved there with SHA-256 `522bd2b3da6912347ac32e3580053b860a323ac8a57ab9545dc665927f4c3137`. The full Docker argv uses image2286e853, no network/pull/GPU,6GiB/2CPU/128PIDs and only the native checkout mounted read-only. Native wrapper SHA-256 matches the failed worker's pin. Python3.12.3/Torch2.13.0+cu130 executed the actual mixed-activity constructor/autostop using CPU events: each case exports once with eight payload annotations, but false makes zero `key_averages`/lazy-event calls. CUDA was never initialized; CUDA kernel timing and the original RPC schedule are not exercised.

Raw census receipt: `/data/ycfeng/tmp/glm53_dsv4/profile03_trace_batches_kpn5_6wu/receipt.json`, with full argv/input hashes. Script SHA-256 `0cdb69efe01ebba8f1553aace70a6a4cd32dcfa327f76c261b97d7767ee39967`. Python `/data/ycfeng/envs/oneshot-opd-py312/bin/python`; ijson/orjson versions are retained in the per-trace receipts. Each isolated rank uses the following prefix, at most two ranks concurrently:

```bash
sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_trace_census.py TRACE EXPECTED_SHA256 OUTPUT --batches
```

| Selected rank | Kernels | Raw bytes | Native annotations | Graph launches | Elapsed seconds / peak RSS KiB |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill DP0 local0 | 356946 | 574658585 | 7063 | 155 | 5.56 /65748 |
| Prefill DP1 local0 | 2434560 | 4278830304 | 104942 | 1012 | 41.57 /68700 |

The original ijson-only P1 attempt exceeded the60-second inspection limit and returned124; its receipt remains at `/data/ycfeng/tmp/glm53_dsv4/profile03_trace_census_fhf5tmfv/receipt.json`. This auxiliary parser timeout neither caused nor reproduced the native failure. Validated batches parse every selected JSON event and the footer, verify gzip completion and match the independent P0 census.

## Observations versus inferences

- **Observed:** the P1 autostop/report-success log interval is542seconds. All eight P1 gzip mtimes are07:01:09–13Z; optional table mtimes are07:06:26–29Z. Native source places stop/export/table work inside `execute_model`, whose default RPC budget is300seconds.
- **Source-proven:** native `torch_profiler_dump_cuda_time_total` defaults true. Exact-image Torch source exports through Kineto before `key_averages()` materializes Python event statistics. Setting the existing field false skips that optional native CUDA-table branch for mixed CPU/CUDA activities.
- **Inference:** the post-export statistics phase is the decisive avoidable contributor to this failure. File mtimes are not phase instrumentation; CPU virtual-duration controls do not reproduce Kineto's measured cost, actual host stacks or the original outstanding-reply schedule.
- **Preserved limits:** export remains synchronous and may independently exceed300seconds. Missing formal publication, actual NET/IB/NIXL transport, threeStopped container exit codes, kernel-bearing useful counts and native captured per-operator ownership remain unverified. CPU-only profiling has a separate automatic CPU-table branch.

Any fresh correction must retain original artifacts, runtime/native/C26 bytes, raw traces,512/64 caps, record limits and the300-second RPC budget. Optional text-table omission must be explicit; a new H800 attempt requires its own recorded package, purpose, ceiling and acceptance.

## Final root execution

- Complete metadata: `/usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/candidate/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_root_final_ipb_iyt3/metadata_audit.json`. Exit0; exact SHA-256 `ab476ca22619818e50b7b142f7d2f7dedd6141740ffa979b9081cea52c1c4376`; all190 sizes/hashes and32 rank summaries rechecked. Individual rank readers retain60seconds/30G and concurrency2. The initial120-second coordinator timeout is preserved in `profile03_metadata_root_final_p_xckr_e`; only that inappropriate aggregate cap was removed, not a native timeout or per-rank inspection limit.
- Native statistics: exact argv/source hash in `/data/ycfeng/tmp/glm53_dsv4/profile03_statistics_root_final_4sfx78oy/receipt.json`; exit0, source `ed17c139...d9c1a`, all non-clock case fields equal. Only three explicitly corrected metadata-record hashes differ from the old20-file snapshot; other bindings are identical. The first wrapper's `KeyError: 'adapter'` occurred before native execution and is retained, then corrected to the inspected receipt key.
- [Primary acceptance](primary_acceptance.json) accepts offline diagnostics and CPU correction controls only. Profile03 remains FAILED; the selected fresh plan omits optional CUDA statistics tables without changing raw export, runtime/native/C26 bytes,512/64 windows or the300-second RPC budget. Future GPU success and complete coverage/pairing remain separate requirements.
