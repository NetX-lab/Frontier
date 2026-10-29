# Balanced q1 isolated-data acceptance review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent data-phase review and subsequent root rerun. |

Reviewer: `W128-BALANCED-DECODE-DATA-REVIEW`, session `ses_e96d5b716ffeQ6CIFYA6vtBizn`. Verdict: **No findings**.

Inspected: parent and supplement requirements/plan, package plan, `data_audit.json`, `command.md`, the data report, `../root_run_check.json`, the complete temporary adapter, legacy raw/table auditor and package-pinned trace parser. The reviewer performed source/artifact inspection only; it did not rerun the trace or query the platform.

The evidence supports the fresh physical TP1/eager/six-layer/dummy-weight measured inputs: exact q1 contexts 6143–7166, 1024 steps and 6144 rows per method, 51200 CUDA_EVENT / 45056 KERNEL_ONLY calls, all 94 scope series, byte-exact CSV reproduction, constructed routing/source identities and unchanged eight output files. Pinned skip semantics and the measured window support the two unmeasured warmups.

The trace reproduces 45056 annotations. Of 352829 kernels, 124928 have measured-scope ownership and 227901 do not. Complete attention/request ownership and actual event-handle/stream dependencies remain unverified. Every context/layer/scope has one sample; repeated-sample stationarity and numerical expert-ID balance are not established.

The review does not qualify serving, graph, physical TP8, constructor-family admission, additive outer-scope sums, CPU gaps or another routing strategy. The final root rerun was not available to the reviewer. Root subsequently reproduced both non-clock receipts exactly and checked duplicate JSON fields and all eight output hashes; `root_verification.json` binds that separate evidence. No remediation was required.
