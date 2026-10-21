# Step HTTP capture verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded actual EOF/healthy/reuse evidence,17new+179existing PASS, preserved test/environment failures. |

Task D23-G3-HTTP-CAPTURE; worktree
`/data/ycfeng/Frontier/.worktrees/step5-http-capture`; branch
`feat/step5-http-capture`; base7fa73f91606841b069b5bd7cdb6ae410a93be8c8.
CPU-only local127.0.0.1 ephemeral servers; no native/GPU/platform/cloud action.

## Environment and source

- Python `/data/ycfeng/envs/asyncopd-py312/bin/python`:3.12.13,Clang22.1.3;
  aiohttp3.13.3,pytest9.0.2.
- Existing simulator fixture interpreter
  `/data/ycfeng/envs/frontier-py310/bin/python`:3.10.6,GCC11.3.0,plotly7.1.0.
  Direct MetricsStore import succeeds; no install or dependency edit.
- `PYTHONDONTWRITEBYTECODE=1`; no pytest cache; scratch
  `/data/ycfeng/tmp/d23_step5_http_capture01/`.
- Base/sealed proxy SHA256
  `b5662156371e44b888ce7afef2ef83b7698b4b4cf54fccc44b670a001aa9cb8f`.
- New proxy SHA256
  `1720b0e8145ab0fcd1ff19c716e59490f18591188cf50308a2a43e6ce61650ce`.
- New test SHA256
  `ec91a84d3cdaa8b4a06afbfc29389877f0fdf70cd37568e77892d62c2e8043c0`.
- Retained EOF driver SHA256
  `fa19dd6052e1118a2d9de21a647d05dc6c15e66cc7e4ba5ffc2bee307c9c3035`.

## Exact commands and results

All relative commands use the worktree above. Logs are at the scratch root.

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_http_capture.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d23_step5_http_capture01/red
```

Expected RED before implementation: exact EOF502 and healthy200 pass business
assertions, then fail missing capture. Actual2failed/1passed0.31s; both failures
`KeyError: 'http_capture'`, `red.log`.

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python /data/ycfeng/tmp/d23_g3_disconnect01/proxy_symptom.py --mode close-before-headers
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python /data/ycfeng/tmp/d23_step5_http_capture01/run_symptom.py --mode close-before-headers
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python /data/ycfeng/tmp/d23_step5_http_capture01/run_symptom.py --mode healthy
```

Retained driver unchanged; wrapper changes only SOURCE and SCRATCH paths, not
HTTP requests, connection policy or arrivals. EOF exit1 deliberately means RED,
not test failure. Original `original_eof.log`: exact502/ServerDisconnectedError,
1P/1D/1client,4.710ms, immutable sealed source. New `new_eof.log`: same502/error,
1P/1D/1client,4.872ms plus capture. `new_healthy.log`:200/GREEN, same counts,
4.880ms, no capture failure. No POST retry.

New EOF same-clock values (seconds):

| Boundary | Actual |
| --- | --- |
| arrival |3069362.466261648|
| P response |3069362.468332460|
| D submit |3069362.468379083|
| fresh connection end |3069362.468850870|
| request headers sent |3069362.468877933|
| public request exception |3069362.469228520|
| caught error |3069362.469234083|

`http_capture.failure`:decode/response_headers,ServerDisconnectedError,6symbolic
frames ending`aiohttp.client_reqrep.start:539`→`aiohttp.streams.read:703`;
message redacted; socket identity UNAVAILABLE_PUBLIC_TRACE. No decode_status.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-http-capture FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python /data/ycfeng/tmp/d23_step5_http_capture01/run_focused.py
```

Wrapper executes exactly:

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_http_capture.py tests/unit/test_calibration_pd_harness.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_canonical_native.py tests/unit/test_step5_client_metrics.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d23_step5_http_capture01/final02
```

Actual196PASS=17new+179existing,pytest77.89s,wrapper78.532592s,
returncode0,observed max childRSS535428KiB(~523MiB), `final02.log` and
`final02_metrics.json`. PASS against<120s/<2GiB proportional verification budget.
`git diff --check` exit0.

## Acceptance gates

| Gate | Result | Decisive evidence |
| --- | --- | --- |
| Actual P/D EOF failure phase and frames |PASS|new_eof.log;new test streaming/nonstreaming and P/handoff cases|
| Unchanged502/native exception,healthy200 |PASS|retained/new EOF and healthy results|
|1P/1D per completed handoff,no retry |PASS|actual server request counts,all exchange cases|
| Fresh vs reused attempts |PASS|same D connection receives local-0/local-1; trace fresh→reused; reused EOF remains502|
| Endpoint/header/token bodies unchanged |PASS|actual P/D paths,request ID,Content-Type,transformed P and exact D body assertions|
| Strict added-capture redaction/bounds |PASS|synthetic auth/token/env secret excluded; trace params ignored;4chain/12frames/16events limits|
| Callback/setup/init/exception recorder failures soft |PASS|injected failures keep200/502 and authoritative native error,UNVERIFIED capture|
| Existing cancel/canonical/client semantics |PASS|affected regression196PASS,deadline and broken HTTP/SSE controls retained|
| Non-Step receipts unchanged |PASS|actual legacy andv0.28 lack http_capture;existing tests|
| Native root cause/exact socket identity |UNVERIFIED|no native rerun,no exact socket ID/public capability|
| Deployment image public TraceConfig capability |UNVERIFIED|localaiohttp3.13.3 only;missing signal setup soft-disables trace|

## Preserved failures and corrections

- First focused command without fixture environment:179PASS/14setup errors80.22s
  (`focused.log`),`ModuleNotFoundError: No module named 'plotly'`. Use existing
  simulator interpreter, no code edit.
- Simulator-interpreter-only rerun:22PASS/1FAIL6.70s (`client_metrics.log`), CLI
  `ModuleNotFoundError: No module named 'tests'`. Explicit worktree PYTHONPATH
  closes this:23PASS9.42s (`client_metrics02.log`).
-17-test expansion had one misplaced assertion,16PASS/1FAIL0.49s
  (`capture17.log`); test-only assertion restored to EOF function; final PASS.
- Optional resource wrapper `/usr/bin/time` unavailable, command exit127
  (`final.log`). Retained simple Python subprocess/resource measurement then
  executes the exact focused suite once; no behavior or dependency workaround.

Observations prove local capture behavior, not the native cause or zero timing
overhead. Added metadata is diagnostic; no timing subtraction is introduced.
