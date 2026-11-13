# D29 emission immutable work report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record bounded H1/H2/H3 and one combination, output/ownership evidence, public conversion rejection and Git-derived candidate estimates. |

TaskID `D29-EMISSION-IMMUTABLE-WORK`; session
`ses_e96a5c1c7ffeoPBK7V08CMas7v`. Frozen source
`7540414341d5b81a86b18b9562d1506c0ba74285` in
`/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`. Parent owns design and
allocation. No production/test/peer edit and no trace encoder change.

## Outcome

The existing key-only diagnostic has a separate **9.65 ms/wave emission gain**:
49.60 → 39.96 ms. It retains all 64 consistency checks, existing getter
snapshots and fresh public proof dictionaries; only derived key strings are
memoized on the actually deeply immutable plans. This is the strongest narrow
safe diagnosis from this task, not a new allocation or a general public-API proof
for arbitrary constructor inputs.

H1's checked immutable-object deepcopy protocol saves 2.76 ms. It is not a plain
`frozen=True` shortcut and does not borrow mutable components. The extra protocol
and type-check mechanism need a parent design; no H1 production patch is proposed
for this small gain.

H3 saves 9.11 ms on canonical real-wave values; H2+H3 saves 18.52 ms, reaching
31.08 ms. **Reject both unguarded H3 and the combined source edit**: a separate
public-value negative proves a constructor-accepted bytearray receipt is returned
as a source alias by the fields conversion, unlike original `to_dict`. No second
H3 control, fallback or restriction is invented to force a proposal.

Even the diagnostic combination's emission alone exceeds the 12–17 ms
whole-output target. The trace escape candidate remains unallocated. No
full-CLI 2400 s admission claim follows from these measurements.

## 1. Ranked hypotheses and the measured objects

The ranking was shown before controls: H1 immutable getter children, H2 repeated
derived key serialization, H3 recursive immutable-field conversion. Each got one
control; H2+H3 was combined once only after both isolated gains and the current
wave assertions passed.

The accepted probe body was executed unchanged, SHA256
`1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`, with the
hash-checked existing setup (`b6c869b2...`). Baseline's real wave is 208 events.

Actual getter counts from the emission-only profile:

| Getter in `frontier/entities/execution_time.py` | Calls/wave |
| --- | --- |
| `moe_or_mlp_time_component`, lines 731–734 | 16 |
| `communication_time_component`, lines 778–781 | 16 |
| attention / MLP / MoE / communication operator-only getters | 0 |

`ep_wave_metrics.py:29` and `:31` use those getters. `:30` reads the plain
`physical_frame` field; it is not two getter calls.

The getter object graphs contain 16 mutable MoETime/MoEOperatorTimes pairs and 16
mutable CommunicationTime/CommunicationOperatorTimes pairs. `physical_frame`,
native-context/geometry lists, operator-cost dictionaries and plans/costs maps
are mutable and remain copied. Their deeply immutable descendants are:

| Reached value | Occurrences in source component graphs | Verified stored contents |
| --- | --- | --- |
| WaveRoutingSource | 16 | str, int, tuple of SourceLaneRouting |
| SourceLaneRouting | 32 | str/int, tuple of TokenInterval |
| TokenInterval | 16 | exact ints |
| CollectivePlan | 64 | str, int tuples, tuple of CollectiveTensor, tuple of int tuples |
| CollectiveTensor | 112 | exact str/int fields |
| CollectivePlanCost | 64 | str/float and tuples of exact int/float |

All reached instances' stored attributes, not just dataclass fields or
`frozen=True`, pass the recursive immutable-domain check. OperatorCost objects
are not reached here: the MoE frame already holds mutable proof dictionaries.

The corrected untimed `sys.setprofile` inventory observes exactly **21,472**
baseline deepcopy calls, including 8,592 str, 8,224 int, 832 dict, 688 tuple,
160 list and 368 type visits, plus the component/frozen-value visits. The first
baseline inventory's monkeypatch missed default-captured recursive calls; use
`evidence/observations.json`, not that initial short counter, for traversal types.

The parent owner's eight retained prepared plans and zero warm constructors
were not re-investigated. After ExecutionTime snapshots this fixture contains
64 distinct plan objects in its source components. H2 primes their derived
strings once **outside emission timing**; getter copies then carry the strings.
This count is not a claim of 64 warm plan constructors or a replacement for the
owner's preparation evidence.

## 2. Phase-isolated measurements

Each label has 15 unprofiled original-body emission measurements in one process.
Only `ep.record_ep_wave` is timed. Setup, key priming, JSONL flush, ledger
encoding/equality, ownership negatives and cProfile are excluded. Original
baseline is uninstrumented; counters are collected separately.

| Control | Median ms | Minimum ms | Delta vs baseline | deepcopy | asdict | Key reads / encodes | Plan conversions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 49.601 | 49.285 | — | 21,472 | 208 | 64 / 64 | 192 |
| H1 checked immutable `__deepcopy__` | 46.844 | 45.671 | -2.757 (-5.6%) | 13,408 | 208 | 64 / 64 | 192 |
| H2 exact derived-string memo | 39.956 | 38.174 | -9.645 (-19.4%) | 19,632 | 144 | 64 / 0 | 128 |
| H3 fresh fields-based conversion | 40.493 | 40.200 | -9.108 (-18.4%) | 15,568 | 16 | 64 / 64 | 192 |
| H2+H3, diagnostic only | 31.080 | 30.486 | -18.521 (-37.3%) | 15,696 | 16 | 64 / 0 | 128 |

The original-name counter is not the total after a control: H2's original
`cache_key` count is zero but `serialized_key_once` is 64; H3's original
`to_dict` count is zero but `plan_fields` is 192. H1 invokes its protocol 144
times. `compute_op_trace_meta` is 208 and both getters remain 16 in every case.
The extra 128 deepcopy calls in H2+H3 versus H3 copy the derived-key attribute
and string through the unchanged 64 plan snapshots.

Supplementary cProfile cumulative times are not speed estimates and overlap
(`cache_key` contains `to_dict`). No instrumented timing is used to claim gains.
First samples are slow (baseline 159.7 ms; H2 312.8 ms), with later raw samples
near the listed minima; all 15 samples are retained and medians are not trimmed.
These are sequential single-process controls, not an interleaved paired estimate.

## 3. Semantics and RED-capable negatives

Every control and the combination passed all unchanged probe assertions:

- 208 events and exact 2,889,655 B body, SHA256
  `18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`.
- Exact fixed-header whole JSONL, SHA256
  `1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11`.
- Exact 2,309,545 B ledger, SHA256
  `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.
- Getter mutation cannot change source or pending events; source mutation cannot
  change pending output; pending event mutation cannot change source.
- Fresh trace/ledger proof maps, costs and mutable contexts; 64 invalid plan-key
  replacements retain the exact `Trace collective plan and cost identities disagree` error.
- Late JSON TypeError preserves the 3902 B first-row prefix and original buffer
  order. No early JSON serialization or encoder change is introduced.

Additional untimed checks:

- Separate getter returns have independent mutable frame/plans/costs maps.
  Separate communication events have fresh tensor/context/buffer dictionaries.
  Two public `to_dict` returns are fresh, with exact recursive dict order, tuple
  types and values compared to the original conversion for canonical input.
- Baseline already shares native-context maps between some events of one lane;
  different lane maps are independent. Every control retains this same alias
  boundary. This is not a claim that all original event maps are independent,
  and no new aliases are introduced in the measured wave.
- Changing an actual buffered collective cost makes the 208-event byte comparison
  RED; restoring it restores exact bytes (`observations.json`).
- H1 uses a recursive exact-type immutable check before returning `self`.
  A bytearray receipt accepted by the original constructor and a frozen tensor
  with a mutable nested extra are copied instead. Deliberately installing plain
  frozen `return self` makes the mutable-receipt snapshot test RED.
- H2's public dictionary mutations do not affect the derived string;
  `dataclasses.replace` creates a distinct exact key and does not inherit the memo.

### H3 rejection is source-backed, not speculative

`CollectivePlan.__post_init__` (`collective_plan.py:72–74`) uses `json.loads`,
which accepts bytearray; it does not normalize context/buffer receipts to str.
The fields conversion returns that raw mutable `buffer_json` field by reference,
although its decoded `buffers` dictionary is still fresh. Mutating the returned
bytearray changes the source; original `asdict` detaches it. The same H3 function
was audited untimed in `retain.py`, with a RED result. This prevents claiming a
semantics-preserving general public-API replacement from canonical wave equality
alone. Do not allocate the unguarded H3/combined artifacts. A parent-owned design
must settle this without silently restricting accepted inputs.

## 4. Minimal candidate artifacts and limits

Artifacts are **scratch renders**, never applied to source. Git-derived counts
include blank additions. Rendered names/formatting differ from the monkeypatch
functions; full module execution was not another control and is not claimed.

| Artifact | Exact path | Git numstat | Gross / net | Decision |
| --- | --- | --- | --- | --- |
| `evidence/h2.diff` | `frontier/cc_backend/collective_plan.py:91–92` | 4+ / 1- | **5 / +3** | Narrow key-only candidate for actually deeply immutable plans; parent design/allocation pending. Original public `to_dict` untouched. |
| `evidence/h3.diff` | same file, import and `to_dict:86–89` | 4+ / 2- | 6 / +2 | Rejected general public conversion; retained diagnostic artifact only. |
| `evidence/combined.diff` | same file | 8+ / 3- | 11 / +5 | Rejected because it contains H3. |

H1's measured protocol would affect immutable values under
`CollectivePlan`/`CollectivePlanCost` in `frontier/cc_backend/collective_plan.py`
and `WaveRoutingSource` in `frontier/moe_routing_policy.py`, with a shared
deep-immutability check and fallback snapshot. No minimal production placement
is justified here; no invented gross estimate or new utility path is allocated.

H2 is proven only for the actual deeply immutable reached plans, not for
arbitrary accepted dataclass subclasses with mutable extensions. It must not be
sold as universally safe merely because the plan dataclass is frozen. The
parent owns that interface/domain decision and the final source implementation.

## 5. Host safety and deferred work

All new Python commands use both `-B` and `PYTHONDONTWRITEBYTECODE=1`, bounded by
60 s / systemd MemoryMax 30 GiB. No dependency install, GPU/platform/network,
full CLI, pricing rerun or unrelated investigation.

Tracked source remains at `75404143` with empty status. The exact earlier
root-bytecode manifest was rechecked: all 284 paths, owners, sizes, mtimes and
file SHA values are unchanged; there are still 33 parent directories. This
separately verifies those known filesystem writes, not an exhaustive filesystem
inventory. No deletion/chown. Large original-body outputs stay only in own
scratch; durable records contain compact results/scripts/diffs rather than a
full workload trace.

Open: parent decision on key-only memo and the small H1 protocol; a safe public
fields-conversion design if desired; larger emission work needed for the budget.
Trace encoding, full CLI and proof acceptance contracts stay unchanged.
