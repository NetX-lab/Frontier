# D27 effective-page source completion

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Close merged source/legacy acceptance; retain full CLI/native limits. |
| 2026-10-22 | Archive verified narrow source step; integrated acceptance pending. |

The Step payload resolver now intersects the native conservative page cap with
the actual retained window. Null slots no longer contribute KV transfer bytes.
The shared cap implementation is unchanged; only its documentation is clarified.

Changed code/tests: frontier/attention/step5_memory.py,
frontier/attention/memory.py, tests/unit/test_step5_structural_pd.py.
Parent4 expected RED failures become18 focused passes;191 affected checks pass.
Independent review reports No findings. At source6128,32 SWA pages and383
full-history/summary pages yield19826 descriptors and139887104 bytes per rank.

Deliverables in this task: requirements.md, plan.md, design.md, progress.md and
test_report_2026-10-22_swa_pages.md. Mergedfa5bae2 passes strict77/77 and
combined D26 checks; the pre-D274080-prefix expectation was corrected to
102203904 bytes. Latest merged per-case outcomes are258PASS/0FAIL/0SKIP across
three invocations; details in the parent merged-workflow report. No native
state, latency, final CLI or calibration claim.
