# D29 late copy/init agent failure recovery

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Recover complete failed-session histories and update the handoff without running either diagnosis again. |

## Result

PASS for evidence recovery only; both technical diagnoses remain unfinished.
Terminal notifications report `Agent not found: "analyzer"`. They are not
Frontier test failures and do not prove that the sessions never started.

| Session | Messages | Completed tools | Rejected tools | New controls/writes |
| --- | ---: | ---: | ---: | ---: |
| `ses_e92340744ffe58Yx5C7xx4qApF` snapshot copy | 20 | 47 | 3 | 0 |
| `ses_e92338e41ffe081mlmDNjpmxP5` initialization reuse | 24 | 33 | 1 | 0 |

All completed operations inspect source, existing small receipts or Git metadata.
Copy has23 reads,21 greps,5 shells and1 context reduction; init has3 reads,
30 shells and1 context reduction. Rejections are `Permission denied:
external_directory`, distinct from the final missing-agent notification.
The last rejected init shell contains a small scope-JSON read, not model loading.

## Commands and checks

Read-only API recovery uses the existing isolated v2 launcher:
`/home/brainpp/.opencode-v2/scripts/opencode-v2 api get /api/session/<sessionID>/message`.
Capture stdout to private regular files, not PIPE. Follow the returned next cursor
with `?cursor=<encoded cursor>`; both older pages contain zero messages.
Each subprocess is limited to60s. Local metadata checks use Python3.10.6.
Expect complete histories, preserved errors, no new technical run and unchanged
owned source trees. Actual: all checks PASS; both source trees are clean at
`886608ea574dd2107d0b30771d63d8057d23cac6`, and both owned record directories are absent.

Selected compressed histories, tool inputs/statuses, exact API read commands,
source checks and hashes are under `handoff_late_failures/`. Raw duplicate files
are removed only after archive read-back verification; the cleanup receipt owns
the exact deleted paths. Existing evidence and unrelated files remain unchanged.

## Limits and continuation

No safe snapshot seam or useful initialization seam is accepted from partial
inspection. The snapshot's single60s/30GiB control is unconsumed; init still
permits no new initialization/benchmark. Required shapes remain6/6 used.
Use an available agent for future continuation under the original scope; do not
automatically resume the unavailable analyzer, change OpenCode configuration or
increase invocation limits. No source edit, replacement agent, worker, GPU,
pricing wave or fullCLI run is performed. Why analyzer became unavailable is
not investigated; no service/configuration cause is claimed.
