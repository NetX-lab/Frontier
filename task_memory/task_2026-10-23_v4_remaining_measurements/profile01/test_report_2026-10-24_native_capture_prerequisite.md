# Native capture prerequisite acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted limited capture-scope evidence after root census, exact-image source checks and independent review. |

Result: **ACCEPTED_CAPTURE_SCOPE_PREREQUISITE_WITH_LIMITS**; [report](native_capture_scope.md), [census](native_capture_scope_census.json), [limits](native_capture_scope_limits.json) and [acceptance](native_capture_scope_acceptance.json).
Root command: `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_20261024.py`; actual argv/stdin/script bytes are bound in the acceptance. Input is one105492191-byte profile02 decode rank0 startup artifact, SHA147c37ec…3fa41; Python3.10.6, no torch import or GPU execution.
Observed/PASS:159748capture rows/31scopes;72798stream-capturing/29scopes;49035breakable-active;43120both and29678stream-only, with the partition exact. All stream-capturing modes are None, layer maximum42, dropped0. This is not FULL-only or complete node ownership.
Expected/actual retention limits match:250000records/268435456bytes, independently read/hash-checked from the same complete input under60s/30G. Limits alone do not prove completeness.
Exact-image AST/docstring check binds `torch/cuda/graphs.py`15b53590…ed81 and confirms keep_graph defaultFalse/source getter requiresTrue/exec getter permitsFalse after capture_end. Python3.12.3 in image2286e853; no runtime handle lifetime or callback delivery observed.
Failures preserved: original31-vs29-scope assertion; original root/reviewer EACCES. A privileged read produced an exact4182-byte census copy, SHA0f2f6776…3678, readable in this task; original bytes/ACL were not changed.
Reviewer `ses_e9280cb35ffeiJS7PwP8sKRM1Z` read the complete corrected report and readable receipt: **No findings**, receipt-read gap CLOSED. Counts/hashes rely on supplied root evidence; no reviewer rerun/rehash is claimed.
Remaining UNKNOWN: callback native/Python thread identity and live scope freshness, source→clone→exec correspondence at native capture, unwrapped/internal/aux-stream completeness, runtime keep_graph=False lifetime and profile04's executed capture set. No cross-run joins, creation-order/numeric remap, probe implementation, extra GPU submission or serving/reference/operator-cost admission.
