# Run-fixed input as the native key component

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record the existing lifetime and the single-expression replacement. |

## Inspected boundary

`collective_sim_cc_backend.py:69` creates an instance-local `OrderedDict`.
The getter/setter at598-609 only operate on that dictionary; there is no disk,
class-level or cross-run prediction cache. Lines809-813 return the same validated
bytes after the first read. The parsed artifact at824-826 is private: repository
search finds no production writer after loading. `Scenario.from_dict` copies
the collective outer dictionary, and the predictor/local-cost functions read,
rather than modify, profiling parameters. CORE parses its own copy of the bytes.

The native branch at841-878 checks qualification, frozen scenario, execution
policy, unique selected model, full actual plan/count context and evidence kind
before it computes the cache key. All these statements remain unchanged.

## Change and gain

Replace only `json.dumps(artifact, sort_keys=True)` at880 with
`self.collective_profile_bytes()`. Keep the `("plan", plan.cache_key(), input)`
tuple, native cost construction and LRU unchanged. Existing input ownership
already prevents a byte change in the run; re-encoding it does not add fidelity.
Using those bytes removes repeated whole-input encoding and duplicated artifact
strings from retained native keys. This supports the required native workload's
CPU/memory budget without weakening a current-point predicate.

## Identity and limits

Raw-byte identity is formatting-sensitive, unlike canonical JSON identity.
That cannot create a new observable formatting-only miss within a supported run:
the bytes are fixed, and a new backend/run starts with an empty LRU. No previous
key migration is needed. Plan keys and emitted receipts retain their canonical
format. Arbitrary mutation of private parsed profiling dictionaries is not a
supported input-update API; the selected run-fixed source remains the authority.

This is simulator execution cost, not a modeled host-time table or a change to
simulated durations. It does not close any native, clean-E2E or whole-CLI gate.
