# CPU proxy test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded actual red/green tests, commands, environment, and verification limits. |

## Environment

- Host: Linux; worktree `/data/ycfeng/Frontier/.worktrees/native-independent-connections`, base `e55c83c263f86f48972f137858723a07c6f5e691`.
- Interpreter: `/data/ycfeng/envs/oneshot-opd-py312/bin/python`; Python 3.12.13 (Clang 22.1.3), aiohttp 3.14.3, pytest 9.1.1. Existing virtual environment; no conda invocation or dependency installation.
- CPU-only real localhost aiohttp HTTP servers and actual `pd_proxy.build_app`; no GPU, native worker, platform, auth, SSH, cloud, Docker, or external checker.
- Command-local namespace bootstrap is necessary because installed editable `verl/tests/__init__.py` shadows the repository's namespace package. No repository or environment packaging file was edited.

## Exact commands and outcomes

Version inspection (exit 0):

```bash
/data/ycfeng/envs/oneshot-opd-py312/bin/python -c 'import sys, aiohttp, pytest; print(sys.executable); print(sys.version); print("aiohttp", aiohttp.__version__); print("pytest", pytest.__version__)'
```

Initial ordinary invocation (exit 2; collection FAIL, no tests executed):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/native-independent-connections TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/oneshot-opd-py312/bin/python -m pytest tests/unit/test_calibration_pd_harness.py -k 'independent_upstream_connections or does_not_retry_disconnected' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-red > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-red.log 2>&1
```

Observed `ModuleNotFoundError: No module named 'tests.comparison'`. An actual import located the shadow package at `/data/ycfeng/step-opd/upstreams/One-Shot-OPD/.worktrees/mopd-step1/verl/tests/__init__.py`.

Red baseline with new tests and unchanged production connector (exit 1; expected red signal PASS):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/native-independent-connections TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/oneshot-opd-py312/bin/python -c 'import sys, types, pytest; package = types.ModuleType("tests"); package.__path__ = ["/data/ycfeng/Frontier/.worktrees/native-independent-connections/tests"]; sys.modules["tests"] = package; raise SystemExit(pytest.main(sys.argv[1:]))' tests/unit/test_calibration_pd_harness.py -k 'independent_upstream_connections or does_not_retry_disconnected' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-red-namespace > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-red-namespace.log 2>&1
```

Observed **5 failed, 1 passed, 62 deselected in 0.97s**. Every failure was the distinct-connection assertion; all reached exact request-count/body/response assertions passed. This red baseline proves reuse exists in the original proxy, not that reuse caused the historical native disconnect.

Green verification after the single connector-parameter change (exit 0; PASS):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/native-independent-connections TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/oneshot-opd-py312/bin/python -c 'import sys, types, pytest; package = types.ModuleType("tests"); package.__path__ = ["/data/ycfeng/Frontier/.worktrees/native-independent-connections/tests"]; sys.modules["tests"] = package; raise SystemExit(pytest.main(sys.argv[1:]))' tests/unit/test_calibration_pd_harness.py tests/unit/test_native_streaming_replay.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-green > /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-native-connections-green.log 2>&1
```

Observed **96 passed in 28.82s**. This includes all six new connection tests and existing native streaming/token timestamp/failure-isolation/canonical-workload and legacy proxy tests.

```bash
git diff --check
git diff --stat
git diff -- tests/comparison/calibration/pd_proxy.py tests/unit/test_calibration_pd_harness.py
```

Observed whitespace PASS and code diff of one production-line replacement plus 169 test lines added in the existing test file.

The first full staged whitespace check flagged space-only blank context markers inside `code.diff`. The exact code artifact was regenerated using `git diff --cached --unified=0 -- tests/comparison/calibration/pd_proxy.py tests/unit/test_calibration_pd_harness.py`, preserving all edits and avoiding context-marker warnings. The final `git diff --cached --check` passes; no code change was needed.

## Per-case expectations and actual results

| CPU case | Expected received POSTs | Distinct TCP transports before | Distinct TCP transports after | Result after |
| --- | --- | --- | --- | --- |
| Legacy buffered, two clients | P=2, D=2; each identity once per role | 2 | 4 | PASS |
| Native buffered, two clients | P=2, D=2; each identity once per role | 2 | 4 | PASS |
| Native streaming, two clients | P=2, D=2; each identity once per role | 2 | 4 | PASS |
| P disconnect before headers, then healthy client | P=2, D=1; no D for failed P | 3 | 3 | PASS |
| D disconnect before headers, then healthy client | P=2, D=2; no retry | 3 | 4 | PASS |
| D disconnect after first SSE token, then healthy client | P=2, D=2; no retry | 3 | 4 | PASS |

- Server transport objects are retained to rule out object-ID reuse; `connections.json` files also retain received IDs and connection IDs, with peer socket addresses in success cases.
- Exact POST multiplicities hold before and after the change. aiohttp 3.14.3 `client.py:256,703-704,867-873` excludes POST from persistent-connection retry; the proxy also has no retry loop.
- Native decode bodies remain the original body plus unchanged native pull metadata. Legacy decode bodies remain unchanged and address-bearing IDs remain intact.
- The streaming server waits until the client receives the first token before it finishes, so buffering the full upstream stream would fail. The exact token-ID event and DONE bytes pass unchanged; existing replay tests verify token-event endpoint clocks, TTFT/E2E/TPOT contracts and fragmented SSE handling.
- Before-header faults remain HTTP 502 with `ServerDisconnectedError`. Mid-stream fault remains a visible client `ClientPayloadError`, with no successful decode-completion timestamp. Subsequent requests succeed.

## Evidence and limits

- Durable compact evidence: `evidence.json`, `code.diff`, `pytest_green.log`, and `pytest_red_connections.log` in this task directory. These are CPU artifacts, not native GPU qualification.
- Temporary full logs and per-case JSONL/JSON are under the exact basetemp/log paths shown above.
- Historical local run source: `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results/run`. The supplied diagnostic report path was not located; its 15s-client/5s-server mechanism and D non-receipt claim are parent-provided context only. The raw proxy/client logs independently establish P=200, no decode status, `ServerDisconnectedError`, and client=502.
- No claim about actual failed-run connection age, server-close timing, device readiness, native NIXL READ completion, GPU E2E recovery, or TCP setup magnitude. No timestamp code, workload counts, arrival schedules, or metric normalization was changed; setup therefore remains within the existing client-measured TTFT boundary.
- No simulator matrix is necessary or run under the explicit harness-only approval.
