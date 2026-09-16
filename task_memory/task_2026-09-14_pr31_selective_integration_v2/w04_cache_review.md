## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Replaced persistent-cache fixtures with constructor-valid stage-local cache tests and preserved typed EP workload contracts. |

## Scope and observed contract

The attention numerical cache is a fresh dictionary owned by one synchronous `predict_stage_execution_time` invocation. Its key is the real model-owned `(family_id, variant_id)` pair. Batch phase, context, padding, state initialization, estimator artifacts, quantization, and topology remain fixed within that invocation. Direct private-helper calls with `cache=None` bypass reuse. A separate constructor-initialized bounded LRU retains immutable exact per-layer EP workload records.

`tests/unit/predictor_cache_fixtures.py` constructs real `BaseModelConfig`, `ReplicaConfig`, and predictors through their complete production constructors. `CacheFixturePredictor`, `CacheFixtureDisaggregationPredictor`, and `predictor_fixture_config` are shared by fixture migrations. Dummy initialization avoids external profile training; tests explicitly replace numerical hooks where deterministic values are required. These checks therefore establish ownership/cache arithmetic, not trained numerical fidelity.

## Focused evidence

Environment: `/usr/bin/python`, Python 3.12.3; no active conda environment. Repository working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Exact command:

```bash
python -m pytest tests/unit/test_attention_query_cache.py tests/unit/test_typed_ep_predictor_contract.py tests/unit/test_moe_routing_conservation.py -q --tb=short
```

Final result: **52 passed in 3.79s**. Raw log: `/data/ycfeng/tmp/pr33-w04-cache-ep-final.log`.

- Actual model-owned GGGAGGGA topology executes eight distinct layers and eight layer-specific MoE lookups, with exactly two attention numerical misses and six hits.
- Explicit attention oracle is `[48, 48, 48, 72, 48, 48, 48, 72]` ms, with MoE layer times `[1, 2, 3, 4, 5, 6, 7, 8]` ms. The stage total is 468 ms. Cache bypass computes every physical layer and agrees exactly; bypass does not change cache counters.
- Separate calls with changed phase/context/padding/state/artifact produce respectively 648/486/612/828/900 ms, each with two fresh misses. These are deterministic numerical-fixture values, not hardware latency claims.
- Mutating the returned attention scalar or operator map cannot alter a cached or sibling result.
- Thirty-two public calls allocate thirty-two distinct dictionaries, each with two entries; aggregate counts are 64 misses and 192 hits. The test observer retains references to compare identity; production has no retained `_attention_query_cache` attribute. The public signature accepts no cache injection.
- Normal constructors own EP LRU initialization. Existing eviction, exact typed-lane validation, conservation, and wrong-routing assertions remain. Monolithic and disaggregated conservation still produce `{0: 2, 1: 2, 2: 3, 3: 1}` for eight routed tokens.

The first attention-only iteration had one test arithmetic typo: context increment by one adds 18 ms across six GDN and two dense layers; the expected stage value was corrected from 468 to 486 ms. Production was unchanged. The first standalone constructor probe omitted `cluster_num_replicas`; the shared fixture now explicitly declares one replica. An EP2 config probe exposed the documented shared-domain check; its valid fixture now uses attention TP2 and MoE TP1/EP2.

## Independent review limits

Reviewer: `/root/w02_acceptance_tests`. Inspected production methods: MoE predictor constructor, public stage prediction, `_predict_attention_layer_time_with_query_cache`, `_clone_attention_time`, and `_materialize_layer_ep_workload`; inspected disaggregation construction and role routing materialization. No production edits were made by this reviewer.

Routing maps are established before workload queries; these tests do not authorize arbitrary live mutation of routing/configuration after cached queries. No trained estimator, GPU measurement, wall-time improvement, or realistic attention artifact reload is established by this focused suite. Artifact freshness is a deterministic replacement of the numerical fixture coefficient between synchronous calls. Real performance ablations are a separate pending evidence item.
