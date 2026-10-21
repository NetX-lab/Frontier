# Native independent upstream connections

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded W128-NATIVE-INDEPENDENT-CONNECTIONS scope and explicit approval. |

## Original request

[Original Request] Task W128-NATIVE-INDEPENDENT-CONNECTIONS. Userjustapproved: nextgeneration P/D upstream POST independentconnections, noPOSTretry, nochange numbers/arrivals, connectioncost remainsinnative measuredTTFT, preservehistoricalV4failedrun. Newworktree /data/ycfeng/Frontier/.worktrees/native-independent-connections basede55c83c. Ownership tests/comparison/calibration/pd_proxy.py (currently TCPConnector(limit=0)) +existingproxytests, minimal code oneforce_close=True existingaiohttpAPI; noflag neededfixedapprovedpolicy. Realreport root task/warmup_disconnect_20261021/report.md andactual /data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results/run proxy/client logs: P200 Dnotreceivedwarmuprepeat2r000502 ServerDisconnectedError; mechanismCPUreproduced15sclientvs5server, exactactualconnectionagetimingunknown (don'tclaimstaleproved). Need CPU regression atrealproxy usinglocalHTTPservers andconnectionidentity/requestcounts demonstratingseparateupstreamTCPforconsecutiveP/Drequests inclstreaming/disconnect; eachPOSTissuedonce, nohiddenretry, originalstreamtimestamps/tokenIDprotocol preserved. Useasyncopd-py312/oneshot-opd-py312availableactualaiohttp3.14.3 preferredrecordversion. Runfocusedlegacyproxy/streamingsuites; noGPU/platform/auth/SSH/cloud/Docker,newpackaging,globalchecker/core changes,merge/push/delegation. Recordtask_memory/task_2026-10-21_native_independent_connections/ requirementsrawdecisionplanprogressreport. Smallscopedcommit+diff+evidence return; no77matrix neededharness-only. The cold-start andnativeobserverowners touchotherfiles; don'tmodifythem.

## Decisions and boundaries

- Use the existing `aiohttp.TCPConnector(limit=0, force_close=True)` API as fixed policy; no flag, retry, or additional production behavior.
- Keep request counts, arrival schedules, payload transformations, streaming bytes, token IDs, and timestamp definitions unchanged. Connection setup remains inside measured client TTFT; do not subtract it.
- Preserve all historical V4 failure artifacts. Do not claim that stale reuse was proved in the native failure: the actual connection age and close timing were not captured.
- Own only `tests/comparison/calibration/pd_proxy.py`, existing proxy test files, and this task's records. No GPU/platform/auth/SSH/cloud/Docker, packaging, external checker, simulator core, merge, push, or delegation.
- Deliver a small verified commit, exact diff, CPU evidence, and limitations. A simulator scenario matrix is explicitly not required for this harness-only policy change.

## Acceptance criteria

1. Consecutive P/D POSTs use distinct server-observed TCP transports on the actual proxy, including native buffered and streaming paths and legacy routing.
2. Pre-response P/D disconnects and mid-stream D disconnects are not retried; received POST counts are exact, failures remain visible, and the next request succeeds independently.
3. Native pull metadata, unchanged decode payload, token-ID SSE bytes, incremental streaming, and existing client/proxy timestamp contracts pass focused regression tests.
4. Historical logs are unchanged; reports distinguish inspected observations from prior diagnostic claims and CPU fixtures from native qualification.
