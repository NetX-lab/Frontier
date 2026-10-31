# F3 H200 NIXL measurements and rejected fit

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Verified completed NIXL02 measurements; diagnosed and retained the invalid affine fit without selecting replacement parameters. |

Tickets06/07/11/19; decision46. The C_nixl phase of `exp-1101-004206-443314` completed on two distinct H200 nodes with all16ranks and both node returncodes0. The job's collective phases failed independently. Source/runtime checks22/22 per node and physical placement passed. Raw cloud input: `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_links_kv_02/C_nixl/`.

## Verification

Commands used Python3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`, CWD the current worktree, `PYTHONPATH=$PWD`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`:

1. `python /data/ycfeng/tmp/glm53_dsv4/verify_nixl02_20261101.py`. PASS:80cases x3passes=240raw records and240CSV rows; all keys match the planned matrix;32,400reader timing trials;214,272notifications received/expected; all recorded data checks pass; all CSV values exactly reconstruct from raw JSONL. Packed stride1,002,240B; bytes/reader13,029,120..865,935,360. The data check occurs on the first iteration of each case, not every trial. Both nodes use the recorded UCX RDMA devices and4UCX threads.
2. `python <task>/h3o_operator_costs/b30b/fit_tools.py nixl --input /data/ycfeng/tmp/glm53_dsv4/b30b_nixl02_20261101/nixl_summary.csv --output <task>/h3o_operator_costs/b30b/f3_nixl02_20261101/fit`. FAIL: `Unphysical unconstrained fit [-46.67691984020917, 0.0032710972078353654]`. No replacement fields were emitted.
3. `python /data/ycfeng/tmp/glm53_dsv4/analyze_nixl02_fit_20261101.py`. Diagnostic residual analysis completed. The120eight-reader rows imply unconstrained latency−46.676920us/bandwidth305.707821Gbit/s, p90 absolute relative error29.175907%. The same-case pass spread has median25.367988%, maximum100.589004%. Separate pass/inflight fits also have large residuals; selecting a favorable subset would hide this variation.

The first inline verification attempt incorrectly omitted requests_in_flight from the expected notification count and raised AssertionError. The producer explicitly emits active_pairs*requests_in_flight*(warmup+repeats) notifications. Correcting this auxiliary assertion gives the PASS above; measured data and production code were unchanged. This was a verifier error, not missing notifications or an incorrect reader count.

## Comparison and disposition

The retained H800126.2Gbit/s+64.7us pair overpredicts the120eight-reader measurements by46.6778%..228.7703% (p90absolute219.6517%). `fit_diagnosis.json` records absolute measured/predicted values and errors by prompt and every fitted stratum. Example T4096: median normalized actual794.567284us, H8002478.970681us, absolute1684.403397us, relative211.990027%.

Measurements have valid data/notification provenance, but the current affine timing fit is not qualified. The negative intercept and residual spread are observations; shared-fabric contention or UCX scheduling are unproven possible causes. Do not clamp the intercept, select one pass, resubmit successful NIXL02, or replace the formal H800 pair. F3 replay awaits a defensible physical fit or a separately decided model/collection correction. F1/F2 work proceeds independently. This standalone probe does not observe native serving request wire bytes; the packed/component accounting limitation remains.

Evidence: `h3o_operator_costs/b30b/f3_nixl02_20261101/{verification.json,nixl_summary.csv,fit_diagnosis.json}`. Raw local input remains `/data/ycfeng/tmp/glm53_dsv4/b30b_nixl02_20261101/`; download hashes and exact cloud paths are in verification.json.
