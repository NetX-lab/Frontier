# Source references inspected (frozen `75404143`)

Source root: `/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`.
All line numbers are from that revision; `git status` was clean and no file was edited.

## Required getter snapshots (unchanged, kept)

`frontier/entities/execution_time.py`

```
731    @property
732    def moe_or_mlp_time_component(self) -> Union[MLPTime, MoETime]:
733        """Get the MoE or MLP time component."""
734        return deepcopy(self._moe_or_mlp_time)
...
778    @property
779    def communication_time_component(self) -> CommunicationTime:
780        """Get the communication time component."""
781        return deepcopy(self._communication_time)
```

`frontier/metrics/ep_wave_metrics.py:29-31` calls each getter exactly once per lane
record and reads `moe.physical_frame` as a plain dict field (line 30), so a lane
performs 16 + 16 getter calls per wave, not two getter calls per event.

## Where the detached snapshots are read again

`frontier/metrics/ep_wave_metrics.py`

```
87                     collective_plans=communication.plans if communication is not None else None,
88                     collective_costs=communication.costs if communication is not None else None,
```

`frontier/metrics/op_trace_utils.py` (meta path, once per collective event)

```
302            evidence=__import__("json").loads(cost.count_receipt_json)["evidence_kind"]
303            return dict(collective_plan=plan.to_dict(),collective_cost={field.name:getattr(cost,field.name) for field in fields(cost)},
```

`frontier/metrics/ep_wave_metrics.py` (ledger path, once per phase and lane)

```
121                collective_proofs={name:dict(plan=communication.plans[name].to_dict(),
```

Lines 87 and 121 select the same `CollectivePlan` object: `ctx.collective_plans`
is `communication.plans`, and both look the name up in that same dict, inside one
`record_ep_wave` call. The two `to_dict()` calls therefore derive identical
dictionaries from one already-detached plan object.

`frontier/cc_backend/collective_plan.py`

```
86     def to_dict(self):
87         return {**asdict(self), "context": json.loads(self.context_json),
88                 "buffers": json.loads(self.buffer_json), "stages": self.stages,
89                 "native_grouping": "outer_tensor_group/nested_root_group"}
```

`asdict` walks the frozen plan (7 fields, tuple fields converted to lists) and
`json.loads` parses the two receipt strings on every call. `cache_key()` (line
91) calls `to_dict()` too; that third derivation is the assigned key-only
candidate (`root 087db1bf`) and is not touched here.

## Per-lane common data (not repeated per operator)

`frontier/metrics/ep_wave_metrics.py:52-64` builds `common` once per lane record,
outside the phase and operator loops; line 103 merges it per event with
`**common` (shallow, no copy). `parallel_context.copy()` (line 108) copies six
integers. `store._get_step5_layer_metadata` (line 71) is warm-cached, which is
the owner's measured "0 warm constructors" result.

## Call sites of the meta builder (interface risk of a prepared-dict parameter)

`frontier/metrics/ep_wave_metrics.py:103`, `frontier/metrics/metrics_store.py:707`,
`frontier/metrics/metrics_store.py:756`, plus unit tests
(`tests/unit/test_ep_wave_trace_context.py:108`,
`tests/unit/test_glm53_attention_cost_contract.py:213`,
`tests/performance/step5_mrv2_cpu.py:301`).

## Tested alias boundaries (from the retained probe body, unchanged)

`/data/ycfeng/tmp/d29_immutable_cost_fields_repair/probe_body.py:83-92` asserts
`proof['plan'] is not event.meta['collective_plan']`,
`proof['cost'] is not event.meta['collective_cost']`, and that mutating the trace
meta dict leaves the ledger proof unchanged. `diagnose.py:198-207` records that
`native_context` is the same object for events of one lane, that different lanes
are independent, and that two communication events own independent
`context`/`buffers`/`tensors` dictionaries.
