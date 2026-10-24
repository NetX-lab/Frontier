# D29 Snapshot Copy Source Inspection Report

## Modification History

| Date (UTC) | Change |
| --- | --- |
| 2026-10-24 | Record source bindings and the supported static rejection; retain zero control usage. |

Disposition: **NO_ALLOC**. The candidate `1461573de388b72450189bc0c1b88b4ac32491f0` fails the required supported snapshot contract by static inference. No runtime control or performance test is represented as having passed or failed.

Component: ExecutionTime defensive MoE component/operator getters. Reviewer: this records owner; no delegate. Main source: `2e8d6168fd72d13955216d59320c9f68b306eca5`. Owner start: `886608ea574dd2107d0b30771d63d8057d23cac6`.

`MoEOperatorTimes.physical_frame` is `Mapping[str, object]` at `frontier/entities/time_components.py:394`. Its constructor validates the timing keys and values at lines 396-414 without narrowing the frame's object values. ExecutionTime copies the supplied MoE operator object at line 203, stores it in MoETime at line 389, and uses deepcopy for `moe_or_mlp_time_component` and `moe_operator_times` at lines 734 and 761. These are supported inputs to the existing public constructor/getters.

The inspected Python 3.10.6 `copy.py:151-153` invokes a custom `__deepcopy__` before considering `__reduce_ex__(4)`. The local pickle reference implementation uses serialization/reduction at `pickle.py:575-582` and exposes the optimized `_pickle` functions at lines 1777-1787. Protocol 4 aligns the unhooked reconstruction protocol; it does not implement the custom-copy hook.

The following is a concrete, importable custom frame-object definition used only to describe the static counterexamples. It is not executed and is not added to source or tests:

```python
class FrameSnapshotValue:
    def __init__(self, generation=0, reject_after_constructor=False):
        self.generation = generation
        self.reject_after_constructor = reject_after_constructor

    def __deepcopy__(self, memo):
        if self.reject_after_constructor and self.generation >= 1:
            raise ValueError("Frame snapshot rejected")
        result = FrameSnapshotValue(
            self.generation + 1, self.reject_after_constructor
        )
        memo[id(self)] = result
        return result
```

Place this object at a `physical_frame` value and supply valid MoE timings. The public ExecutionTime constructor produces generation 1 from generation 0. A defensive getter then returns generation 2 through the hook. The proposed `pickle.loads(pickle.dumps(value, protocol=4))` reconstructs generation 1 without that hook. With `reject_after_constructor=True`, construction still succeeds, the getter raises the specified ValueError, and ordinary pickle reconstruction bypasses the error. Both result and error divergences are **statically inferred, not observed runtime results**. No fallback or special-case wrapper is proposed.

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| Recovery | Reuse completed inspections and preserve prior usage | 47 completed reads, 3 rejected reads, 0 prior controls; existing receipt/history retained | PASS |
| Candidate/main/owner binding | Equal inspected source and exact current bytes | 25 paths; all three blobs agree; target bytes match candidate | PASS |
| Supported custom object domain | No unapproved narrowing of frame values | Existing Mapping[str, object] domain retained; timing-only validation | PASS (source inspection) |
| Candidate custom-copy preservation | Preserve hook results and errors | Two supported static divergences above | REJECTED (inferred) |
| Runtime parity and speed | No unapproved invocation | No pickle operation, benchmark, wave, or control executed | NOT RUN |
| Original control allowance | At most one, count preserved | 0 launched/executed out of 1; final instruction prohibits additional controls | SKIPPED |
| Primary execution/admission | Independent owner | Supervisor 73779 not queried/restarted; candidate remains unadmitted per primary | OUTSIDE SCOPE |

Exact executed version/semantic read commands:

```bash
/usr/bin/python3 --version
/usr/bin/systemd-run --version
/usr/bin/timeout --version
rg -n '__deepcopy__|__reduce_ex__|__reduce__|from _pickle import|class _Pickler' /usr/lib/python3.10/pickle.py /usr/lib/python3.10/copy.py
git -C /data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314 rev-parse HEAD
git -C /data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314 rev-parse '2e8d6168^{commit}'
```

Observed versions: `/usr/bin/python3` Python 3.10.6; systemd 249 (249.11-0ubuntu3.6); GNU coreutils timeout 8.32. Static Python readers use `-B`. No conda environment is used. Exact binding argument templates, all 25 path/blob bindings, local library SHA-256 values, primary document bindings, and other decisive read arguments are in `evidence.json`. Complete recovered session/tool commands remain at the existing recovery paths recorded there. No synthetic command exists to report: launch attempts, executions, configured control threads, and candidate pickle invocations are all zero.

The unused original envelope is 60 seconds/30 GiB, Python 3.10.6 -B with single math threads. It was not launched; memory enforcement and alias/runtime parity were not verified. Ordinary mutable graph behavior and exhaustive pickle compatibility remain untested because one supported divergence already prevents allocation. The C implementation itself was not executed or separately source-inspected; the rejection relies on the serialization/reduction contract represented by the local standard library.

Historical profile attribution is unchanged: source `7540414341d5b81a86b18b9562d1506c0ba74285`, 16 MoE getter calls with 15 ms cumulative time and 16 communication getter calls with 19 ms cumulative time in the retained profile. Their 34 ms sum is profiler attribution at the old source, not current wall time or a savings estimate. Current candidate speed is unmeasured. Inferred delivery contribution is zero newly implemented savings because the candidate is rejected. Prior loader, JSON, fields, and key savings are not counted again.

Record checks: all seven delivered files remain under the authorized directory, maintained Markdown files have Modification History, JSON is parseable, and staged changes are checked before the records-only commit. Exact verification outcome and the commit locator are in `receipt.json`.
