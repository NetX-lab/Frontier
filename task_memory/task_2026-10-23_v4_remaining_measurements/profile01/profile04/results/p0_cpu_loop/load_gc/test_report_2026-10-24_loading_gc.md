# Loading-GC pilot results

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded exact decode control and incomplete sole P0pilot; no more invocation or acceptance. |

Scratch root: `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/`.
`load_gc_control_compare.json` binds60s/30GiB exact decode payload/index equality,
142567kernel rows and61useful steps;34-line source SHA2566b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02.
`p0_load_gc_01.receipt.json` records full argv, Python3.10.6, source/input paths,
initial actual30GiB controller, closed output hashes and exit124 at60.314844seconds.
The first bad decode filename/exit1 happened before loading and remains preserved.

P0 loading31.384seconds versus47.129seconds for the retained freeze baseline.
All1215919kernel records are processed by52.788seconds; stream processing58.187,
sync58.201 and512step predicates/39useful steps58.218seconds. No final audit JSON
was produced: the60second cap interrupts final sorted-interval metrics.
This directly measures loading improvement but not complete P0/full32 coverage.
GC re-enable, exact controls, raw sources and unchanged parser/predicates are explicit.
Complete kernel prefix and diagnosis artifacts remain at cited scratch paths;
no kernel ownership, dependency, cost/reference admission, limit increase or GPU operation.

Independent gzip parsing reaches clean EOF and exactly1215919rows, but final audit JSON
is absent. This settles only kernel-index completion, not all metrics or P0 acceptance.
