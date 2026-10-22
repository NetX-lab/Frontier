# Isolated prefill plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Defined bounded source-first implementation and CPU acceptance. |

`a -> b -> {c, d} -> e`; c and d are independent local checks, no subagents.

1. a: Read primary approval/ticket/override and actual accepted raw, plan, manifests and readers. Record independent identity sources and actual enumeration.
2. b: Implement one production layout builder reusing producer model/metadata readers; extend only explicit accepted prefill provenance and PD rejection. Integrate an opt-in accepted receipt/package path in the existing checker. Stay within three maintained files/200 changed lines and150 core lines.
3. c: Run the affected unit file, preserving decode and error behavior; test actual accepted prefill positives and typed negative mutations without changing accepted files.
4. d: Run both explicit accepted-input checker and legacy checker on original tables. Require exact94 real scope comparisons and legacy102/72 unchanged. Inspect source diff and choose affected regression from reachability; do not automatically run strict77.
5. e: Run check-progress self-check, reread requirements, retain exact reports/source diff/hashes, commit and return to root for independent acceptance. Constructor, serving/PD/graph and complete bundle remain unqualified.

Environment: `/data/ycfeng/envs/frontier-py310/bin/python`, CPU-only, `PYTHONPATH=.`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`. Raw logs/caches stay under that temporary root; selected comparison evidence and durable reports stay here.
