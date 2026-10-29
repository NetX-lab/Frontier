# D29 timeout diagnosis progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Close the real-path feedback-loop diagnosis and record parent evidence boundaries. |

Task ID: **D29-CLI-TIMEOUT-REPRO**.

## completed — failure/source binding

Main07 offline/online failed by2400s timeout, not a traceback. Actual source is
e41/3dac, with fixed `TP,DP` placement. Streaming census found exactly seven
PREFILL scheduler rows per mode. Parent later independently supplied the full
input-integrity and missing-business-artifact receipt; use that receipt rather
than duplicate its checks. Initialization warnings began about21minutes after
launch and ended about26minutes after launch. These timestamps identify a slow
initialization interval but not a CPU stack.

## completed — real RED loop

`runtime_loop.py` reconstructs the actual main07 CLI replica configuration,
relocates output/cache paths, verifies the retained duration/count model bundle,
and uses the actual P6128+D dummy1 source at layer3/EP0. Production query and
pricing methods are called unchanged; full execution mode constructs the real
Cluster/collective backend and returns actual retained cost/proof/plan keys.

Final unprofiled repeated execution: **3.453060s,3.376364s**,1056 layer
descriptions total, deterministic **RED**, exit1. Profiling two production calls
found12 full routed-group resolutions,1056 layer descriptions,14.438s in the
group resolver out of15.239s cumulative pricing time.

## completed — hypotheses and one-variable probes

The ranked falsifiable hypotheses were reported before the corresponding probes:
1. Repeated resolution of the same fixed model's routed groups dominates; reuse
   of the actual resolved tuple should reduce time without changing query keys.
2. Retained-row/full-plan scan dominates; if1 is removed, scan should remain slow.
3. Repeated CSV reads/row validation dominate initialization; a bounded original
   loader profile should stop there even with retained fitted models available.
4. Collective pricing dominates; after1, its time should dominate residual cost.

Only diagnostic in-process reuse of the actual fixed model's group result:
**0.404157s,0.351173s**,zero repeated layer descriptions, **GREEN**, exit0.
Costs, count proofs and plan-key payload SHA256 are exactly identical to RED.
This isolates hypothesis1. Query-only residue was approximately0.085–0.102s;
hypothesis2 is not the primary hotspot in this sampled point. After group reuse,
actual backend repeatedly reads/parses the same collective-local JSON artifact
and accounts for0.542s of0.781s cumulative pricing time over two profiled calls.
That is a secondary hotspot, not an authorized combined fix.

Full original2.1149GB MoE CSV loader sampled for15s:9.302s CSV parse, then5.696s
per-row typed validation;3021 repeated model-family/config validations within
the sample. A separate repeated actual first-row metadata microprobe measured
5.764ms/row under profiling. Its213.52s projection to37,044 rows is explicitly an
extrapolation, not a full-map or main07 measurement.

## completed — diagnostic corrections and limits

- User/system systemd scopes were unavailable: user scope failed and system
  scope required interactive authentication. Probes used `timeout60s`,
  `RLIMIT_AS=30GiB`, streaming/scoped reads and peak RSS at most about1.14GiB.
- Initial pstats counter export read consumed stats after formatting; early
  counter fields are not relied on. Final RED uses an explicit call counter;
  profiles retain the actual1056/12 call counts.
- No production or tests were edited. All diagnostic substitutions occur only
  inside the scratch probe process. Original source/inputs/timeouts remain.
- No new full CLI, GPU, platform, network or dependency operation occurred.

## pending — parent-owned next work

Parent ranks the findings, assigns any repair paths/line budget, and validates
the original complete workflow after a reviewed fix. This diagnosis does not
prove that group reuse alone would pass2400s, reproduce all EP/layer costs, or
establish native/accuracy acceptance. No further diagnostic probe is needed to
return this phase result.
