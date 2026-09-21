## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Indexed completed ten-warmup H800 validation evidence. |

# H800 ten-warmup test report

The complete test script paths, reproducible commands, environments, criteria, exact rank values and limitations are recorded in [the run report](analysis/h800-ar-176000-normal-10warm-01/report.md).

Bounded identity/drain PASS: 10 warmups, one request per replay, one formal request, 11 completed client rows. Formal DP0 batch5120 CUDA span median522.214080811ms, rank P90 522.508959961ms, maximum522.551147461ms, spread2.752014160ms. Normal-scale reproduction FAIL against70-110ms. Full standard-suite gate remains unfulfilled. Code commit d9fdf61a; both actual bounded analyses and shell syntax passed.
