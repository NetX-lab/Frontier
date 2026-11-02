# CUPTI stream loader repair

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted source0d745706 after meaningful RED/GREEN controls, real-image method verification and independent review. |

## Result

`ACCEPTED_CPU_STREAM_LOADER_REPAIR`, source0d745706,34/80gross lines from `git diff --numstat`. Only the collector and its existing unit-test file changed. Torch's CUDA build supplies the major for its already-loaded CUPTI SONAME. No path/config flag, symlink, fallback or altered error suppression was introduced.

## Commands, environment and evidence

Exact argv/env/source hashes are in `red.json`, `green.json` and `image.json`. CPU suites used `/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest`, `PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site`, the pinned native checkout and approved TMPDIR. Each suite has its own XML, log and result; no skips/errors/failures in GREEN.

The CPU interpreter reports `3.12.13 (main, May 10 2026, 19:30:01) [Clang 22.1.3 ]`, checked with `/data/ycfeng/envs/asyncopd-py312/bin/python -B -c 'import sys; print(sys.version)'`. The locked-image dependency check used its Python3.12.3/GCC13.3.0 and Torch2.13.0+cu130, as retained in the earlier image inventory.

| Check | Expected | Actual |
| --- | --- | --- |
| Actual stream recording, only versioned CUPTI available | Baseline retains None and fails expected traceID7 | RED: `assert None == 7` |
| Collector suite | New successful mapping plus existing failure isolation |33PASS |
| Native P/D worker | Existing launch/lifecycle behavior retained |19PASS |
| Native serving inventory | Existing inventory behavior retained |28PASS |
| Native V2 inventory | Existing native V2 behavior retained |45PASS |
| Real locked-image method boundary | Old method loader fails; new method reaches CUDA-context check | PASS on Torch2.13.0+cu130, CUDA uninitialized |
| Independent source/test review | No unresolved actionable findings | No findings |
| Required default matrix | Explicit unchanged-domain binding, no new capture |77 identical; comparison SHA verified |

The unit control restores the actual `trace_stream` method and calls the actual `stream()` recorder; it distinguishes Torch streamID110 from CUPTI traceID7 while retaining raw handle10. The existing absent-mapping test continues to retain raw handles and an observation error without inventing synchronization edges.

The Docker test executes the exact AST-extracted method body from baseline and repaired files against real image libraries. This is a reduced-method test, not a full module/worker or GPU test. Its initial local driver chose the wrong argv index for the image and failed before Docker; the corrected unique-image selection passed and is recorded in `image.json`.

Reviewer `ses_e963f099dffeoPCo7YCv82W14r` inspected requirements, full two-file diff, source and RED/GREEN receipts; it did not rerun checks. Root separately verified source bytes against the committed source and the locked-image receipt. `matrix_binding.json` records byte-identical relevant domains to the prior56e9405d binding and the original77-case comparison SHA. No broader rerun was needed for this isolated native observer change.

## Remaining verification

The single-H800 probe must import the actual collector, map real streams and join their GPU kernel trace IDs across native profiler start/stop cycles. CPU success does not establish stream-ID stability, captured graph ownership, GPU event dependency, native-reference or operator-cost admission.
