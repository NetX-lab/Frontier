# W128-NATIVE-STREAM-PROBE-DATA verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Independently audited downloaded bytes, exact raw joins, lifecycle and graph evidence; retained UNKNOWN. |

Outcome: **CPU audit PASS, eager stream evidence PASS, captured per-op graph correspondence UNKNOWN**. Final162/162 CPU checks passed; fresh-process repeat is byte-identical. These are verification predicates, not162GPU samples. Commands/environment: [command.md](command.md). Raw inputs were not changed.

## Expected versus observed
| Criterion | Expected | Observed | Result |
| --- | --- | --- | --- |
| Download inventory | Exact5files/228175bytes, receipt hashes | Every byte count/hash matches both download receipts | PASS |
| Package/source binding | Seven pinned inputs, frozen source5f7d0950 | All7hashes;4source-lock/current/Git-commit byte comparisons; executed probe/import hashes match | PASS |
| Native profiler pins | Config/wrapper pinned bytes | Local native source hashes match worker telemetry and root CPU pins | PASS |
| Library binding | Mapped CUPTI13 matches pinned hash | All12reported library paths in proc_maps; CUPTI hash matches source_lock/root_cpu | PASS, telemetry limits below |
| Snapshots | Eight stages×default/aux, cache=fresh | All16rows identical across both cycles; directly queried IDs7/13 equal cached trace IDs | PASS |
| Eager targets | 2cycles×2streams×add/mul | Eight unique same-thread annotation/runtime/kernel joins; correct streams | PASS |
| Priming | Four per stream/cycle, retained separately | Sixteen unique same-thread annotation/runtime/kernel joins; correct streams | PASS |
| Profiler reuse | Same wrapper/outer profiler, two retained traces | Wrapper140509011878752; profiler140509156519344 throughout; startTrue/stopFalse each cycle | PASS, telemetry limits below |
| Capture/instantiate/replay completion | Returned success and actual native records | capture_returnedTrue; both synchronized replay_returnedTrue; begin/end/instantiate runtime records and2replay kernels/cycle | PASS, no numerical tensor check |
| Captured per-op identity | Exact captured graph/node fields, elseUNKNOWN | Each captured annotation has1runtime,0kernels,0graph/node fields; replay IDs alone cannot identify captured owners | UNKNOWN |
| Loss/error census | Preserve all expected labels/kernels/errors | 29CPU annotations,28kernels,50runtime rows; exact kernel partition; worker/collector errors empty | PASS for targeted census |

## Exact downloaded hashes
| File | Bytes | SHA256 |
| --- | --- | --- |
| native_profiler.log | 194 | a47c77ff4e0ab9a3c31cdd40064f0f9dd34278e1eab482bc4143c4c16ce9479d |
| native_traces/native_stream_probe.1792778788224797738.pt.trace.json.gz | 4259 | bf9be1a84dfa25c5d60c92715476fb7b9e8b7dc082feb46514a0f79e81efad18 |
| native_traces/native_stream_probe.1792778788371600646.pt.trace.json.gz | 3464 | b1648316902c3d12a1d91e76916711a8ad0c8ff201e983f94a5c0a66aea3c074 |
| probe_report.json | 91276 | f17a85b80a8c63c277f231208f7d218645ffb01326b9cd8b8bfa340e8de3812b |
| proc_maps.txt | 128982 | 23203705e961165f9229a2785ec00a305d144be00fcc0262ab083294861b947e |

Manifest SHA256: `0725941c3b433b2bd300c10a48c3d6f6d5316e1036e1e8efb5294b8c2845431d`; source revision `5f7d0950b78276a77a66b7330cf0041a9f380f12`. All seven exact hashes and receipt/native pins are in [data_audit.json](data_audit.json).
Audit JSON SHA256: `601cb737fd5f25d76feddd483225090f291dc6bb1f9b9acfb83647a5852f1310` (142589bytes).

## Raw identities and retained limits
- Independent reconstruction uses raw gzip JSON, exact pid/tid containment and runtime correlation IDs; it does not import the probe/parser or consume report verdicts. All retained annotation/runtime/kernel rows equal the independently reconstructed rows.
- Default: CUDA handle0, Torch stream_id0, cached/fresh CUPTI ID7. Auxiliary: CUDA handle1266856912, Torch stream_id3, cached/fresh CUPTI ID13. Torch IDs must not replace trace stream IDs. One observer cache survives all snapshots; no cache reset/new process/new stream is tested. Source: `native_serve_profile.py:224-250`, probe `:108-117`.
- Cycle1/2 raw events205/164; CPU annotations16/13; runtime rows33/17; kernels14/14. Capture labels each have1annotation/1cudaLaunchKernel/0kernels. Actual runtime args: add `{External id:29,cbid:211,correlation:228}`, mul `{External id:31,cbid:211,correlation:240}`. No captured graph/node IDs occur there.
- Capture envelope has16runtime rows, including cudaStreamBeginCapture, cudaStreamEndCapture and cudaGraphInstantiateWithFlags. Replay correlations257/473 each join2kernels on stream13. Exact replay pairs are `(graph id2,node id8589934592)` and `(2,8589934593)` in both cycles. Only these4kernel rows have positive graph/node fields. No name/order/numeric remapping supplies captured per-op identity. Probe `:167-186,41-77`.
- Cycle2 lacks `vllm_version` and `vllm_version_tuple` metadata present in cycle1. Native `vllm/profiler/wrapper.py:253,277-306` retains `_version_metadata_added=True` across starts. This source explains the loss; no source change was made. Target labels/kernels are complete, not proof of all profiler metadata completeness.
- Native log has one configuration line and one stop-success line, no warning/error. `wrapper.py:76-83` uses `logger.info_once`, and clears running state even if stop throws. Neither one success log nor stopFalse alone proves two successful stops; both distinct complete retained traces and source-bound lifecycle telemetry provide the additional evidence. Inner Kineto/CUPTI object IDs were not recorded.
- Worker Python3.12.3/Torch2.13.0+cu130/vLLM0.28.0; GPU NVIDIA H800 UUIDc459bcfe-2bea-e452-274c-beb947928449. Trace driver/runtime versions13000/13000, CUPTI130001; cuda.bindings calls report driver `(0,13000)` and runtime `(0,13030)`. The runtime-version discrepancy is retained, not normalized or claimed equal.
- All downloaded/local package/native source bytes were freshly hashed. Worker-only Torch/CUDA/library binaries were not supplied for local rehash: their exact hashes remain pinned worker telemetry, cross-checked against available receipts and proc_maps. CUPTI `libcupti.so.13` SHA256 `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`.
- Platform jobexp-1024-020317-012085 Succeeded/exit0, image/mount/resources/creator are root-provided `platform_terminal.json` evidence only, never requeried. Capture/replay telemetry establishes completion with recorded kernels, not output-value correctness. No native/reference admission, serving/operator cost, CUDA event dependency, cross-rank or cross-hardware qualification follows.

## 1. 规划完成情况
**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED0｜TODO0
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 下载和包绑定核对 | PASS | 5原始文件、7输入、4提交源文件独立哈希一致 |
| 原始关联和生命周期核对 | PASS | 8eager、16priming关联完整；graph保留UNKNOWN |
| 复跑和限定交付 | PASS | 162检查通过，复跑逐字节一致；只交付3指定文件 |
## 2. 偏离检测
**结论：** 未偏移，正常推进。**检测窗口：** 本次数据审计请求至交付｜**当前前沿：** root独立复跑和最终接收
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 临时脚本仅核对本次实际输入 |
| D2 细节反复重试 | CLEAR | 无重复修复失败；两次最终运行均通过 |
| D3 支线问题修补 | CLEAR | metadata差异仅记录，未改native源码 |
| D4 可并行任务停滞 | CLEAR | 用户限定单一审计任务、无delegation |
| D5 超范围镀金 | CLEAR | 仅3指定交付文件和临时脚本 |
| D6 分析空转 | CLEAR | 每项检查对应验收条件并形成交付 |
| D7 路线违背 | CLEAR | 无GPU/网络/平台/Git写入，无ID推断 |
**文档检查：** 当前审计请求与限定产物中，S1–S6均未发现明显缺陷；旧source-owner进度属于历史阶段，实际root收据已核对。
**归因：** 无。
## 3. 纠正建议
**分支：** A：spec/tickets合理，无需改动。**需停止的偏离行为：** 无。
**接下去的路线：** root复跑冻结临时脚本→完成标志：结果逐字节一致且接收UNKNOWN及证据限制；本owner停止写入。
## 4. 最值得关注的 3 点
1. 原始eager/priming与stream映射完整；162是CPU检查数。
2. captured per-op身份缺失，即使两轮replay节点一致也只能UNKNOWN。
3. cycle2 version metadata缺失，runtime API版本记录不一致；无扩大资格结论。
