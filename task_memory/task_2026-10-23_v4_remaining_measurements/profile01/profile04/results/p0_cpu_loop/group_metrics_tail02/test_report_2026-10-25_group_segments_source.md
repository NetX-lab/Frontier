# Group metrics segment source and tiny controls

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Preserved the failed original cgroup control and verified the separately allocated v1 sensor control. |

The new timing module observes the accepted group metrics without changing its computation. It wraps each original `sorted` call once, forwards all arguments, returns the original list, and adds sparse global sort/order and per-group count/sum/sort/union boundaries. Six group observations include the current `repr(key)` and `len(members)`. The source is 112/140 gross lines, SHA256 `9f456317208fa3646fd32cf24d6e92cfe7c9ca08da716b98aabf0b0aa619c426`. No accepted source, GC/loader policy, grouping, attribution, raw input or original result was edited.

The primary takeover supplement retained the original allocation pin and raised the unrun tiny cap to 115 lines rather than compressing the clear partial control. The first 112-line control added an incorrect cgroup-v2 `memory.max` assumption. Its sole invocation is FAILED and spent: child1, enclosing0.06200661929324269s, internal0.0007084459066390991s, zero checks. The actual memory limit was not read, so the requested30GiB is not recorded as observed. `owner/tiny_result.json` retains the FileNotFoundError and all original source/input/command/launch/log hashes remain unchanged. No AST or fixture PASS is inferred from this failure.

The measured repair is separately allocated in `primary/tiny_repair_allocation.json`. The fresh 114/115-line `owner/tiny_control02.py` copies the existing pinned observer's v1 memory-controller/mount-root mapping and reads `memory.limit_in_bytes`; it adds no future-controller fallback. Its SHA256 is `dabfdc8199ea3e6e17ea9c7ef96656801c5c8b973b29c02ca13a6096c0580650`. `owner/tiny_inputs02.json` is `df1c94280684d4c604644c4171b04316228868b4f5991f0d511c09c487f9a8a2` and binds the original failure and repair before execution. The original control was not imported or rerun.

Both commands ran from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, using `/usr/bin/python3` 3.10.6/GCC11.3.0 and no conda:

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -I -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/group_metrics_tail02/owner/tiny_control.py
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -I -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/group_metrics_tail02/owner/tiny_control02.py
```

The second command is its own newly allocated once-only repair, not an automatic retry. It returned0, enclosing0.21523058274760842s, internal0.13229615287855268s. Expected and observed actual memory are32,212,254,720bytes. All32 retained check rows have statusPASS and exact serialized expected/actual equality. Fixtures cover no groups, empty groups, stable equal-time ties, duplicate hierarchy membership and original-order floating-point sums. Each checks exact count/sum/union values with floathex, stable original/duplicate orders and sort count, sparse boundary count, and the exact six-row key/membership sequence. The original/reverse sum witness differs as required. Both wrappers preserve the original returned-list identity and exact positional/keyword forwarding.

The control also proves exact stripped composed-module, accepted source, unchanged union recurrence and restored pipeline CLI AST. Its recorder deep-copies expected/actual before clearing observation/order lists; primary reread every serialized row, not just the exit code. `primary/primary_source_inspection.json` binds this read-only inspection, all21 immutable pins, closed log hashes and the no-rerun limit. `owner/tiny_result02.json`, `tiny_launch_receipt02.json`, `tiny_stdout02.txt`, `tiny_stderr02.txt` retain the actual command and outputs. Standard-library AST/fixture execution imports no Torch/CUDA library; GPU ownership was not tested.

`primary/diagnostic_preflight.json` records a0.24852317478507757s streamed hash/size check of the unchanged input triplet, source pins, original55-line driver and download inventory. It did not parse a trace or invoke an auditor/control. Independent Spec and Standards reviews inspected all active source/control/input files and the retained32checks;both have zero open findings and preserve the original sensor failure/remediation. `primary/source_review.json` accepts this CPU/source gate with limits. The sole segment diagnostic remainsNOT_RUN0/1 and is frozen in `primary/diagnostic_execution_allocation.json` after those reviews and exact command/source bindings. Original tiny1/1 and fresh repair tiny1/1 are spent; this source/control evidence grants no affected-rank/complete-control invocation. R5 remains24/32 and native ownership, dependencies, timing/non-perturbation, whole-DP, transport and cost/reference admission remain open.
