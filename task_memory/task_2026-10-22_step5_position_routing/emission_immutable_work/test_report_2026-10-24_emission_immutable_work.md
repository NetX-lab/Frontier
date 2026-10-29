# Emission immutable work verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record exact commands, evidence and semantic rejection alongside measured improvements. |

## Environment and common command wrapper

Source cwd `/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`, frozen
`7540414341d5b81a86b18b9562d1506c0ba74285`. Python
`/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6, GCC 11.3.0.
No new dependency or py312 campaign.

Every script invocation uses:

```bash
sudo -n systemd-run --scope -q -p MemoryMax=30G timeout 60s \
  env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B \
  /data/ycfeng/tmp/d29_emission_immutable_work/diagnose.py LABEL
```

The exact labels are `baseline`, `h1`, `h2`, `h3`, `combined`. Each was run once;
stdout/stderr go to own scratch `LABEL.log`. Each runs the accepted body
unchanged for 15 emission-only measurements and assertions, then collects
untimed inventory. No timeout or memory-limit failure occurred.

## Input and control binding

- Existing setup `/data/ycfeng/tmp/d29_cost_fields_parent01/current_setup.py`,
  SHA `b6c869b2b01bd118b02aaf0a9755ee6b87e25a7f00c9667cbd52d50a2cca2b80`.
- Existing unchanged body
  `/data/ycfeng/tmp/d29_immutable_cost_fields_repair/probe_body.py`, SHA
  `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`.
- The body binds its original 208-event reference; `retain.py` also reads the
  requested retained `candidate/iteration-0/op_traces.jsonl` from the same owner.
  Both have exact body SHA `18d32043...` and 2,889,655 bytes.
- Baseline driver SHA at run `9ed0d963c6c7f82fc68dca982165cf28a90ccc8ffd1c86923385089ad1012fa5`;
  final control driver SHA
  `8c3297ec19b2f238d28bc8a81f970ae8ef4d73f6b4442e96594eeb46c48aad39`.
  The final driver adds the three scoped monkeypatches and untimed ownership
  checks, not changes to the preserved timing/assertion body.
- Nine source hashes, including collective_plan, trace_store, execution_time,
  time_components and moe_routing_policy, are in every inventory.
- H2 narrowly copies the function in owner's `compare.py:27–31`, renamed only in
  artifact render. It primes only derived strings outside the measurement; it
  does not rerun, inspect or change pricing.

## Results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| 208 actual events, five labels | body and whole JSONL match original | All five body SHA `18d32043...`, whole SHA `1850a9b6...`; exact body bytes 2,889,655 | PASS |
| Full ledger | exact values/order/proofs/bytes | All five 2,309,545 B, SHA `006dc7e8...` | PASS |
| Original getter/pending/source mutations | no source→event alias; getter mutable copies detached | All unchanged body assertions pass | PASS |
| Original trace→ledger maps | mutable plan/cost dictionaries independent | All unchanged body assertions pass | PASS |
| Invalid collective keys | 64 exact ValueErrors/label | 64/label, exact original error retained | PASS |
| Late JSON failure | same 3902 B prefix and buffer restored in order | 3902 B/label; actual TraceStore unchanged | PASS |
| Getter counts | no bypass | 16 MoE + 16 communication/label; all operator-only getters 0 | PASS |
| Separate communication events / getter outputs | no new mutable aliases | fresh tensor/context/buffer dictionaries and getter maps | PASS |
| Baseline same-lane native-context graph | report actual alias boundary | Original shares same-lane maps; controls preserve it; different lanes independent | PASS relative to baseline, not universal event independence |
| H1 constructor-accepted mutable bytearray / nested mutable extra | deepcopy does not return self | Both detached; deliberate plain frozen return-self negative goes RED | PASS |
| H2 public mutations / replace | immutable key unchanged or recomputed for replacement | Fresh to_dict unaffected; replacement gets distinct exact key | PASS in actual immutable domain |
| H3 recursive types/tuples/dict order | identical for canonical data | Exact original structure and values | PASS for canonical values |
| H3 public mutable receipt | must detach all accepted mutable content | Same tested fields conversion aliases bytearray, mutation changes source | **FAIL; reject unguarded H3/combined source edit** |
| Actual buffered-output mutation | differential must catch changed value | Last collective cost mutation turns byte comparison RED; restoration exact | PASS |
| Performance | measure emission only, no instrumentation around timed baseline | Baseline 49.60; H1 46.84; H2 39.96; H3 40.49; combined 31.08 ms medians | Observed, not CLI admission |

Original named asdict/deepcopy/key/conversion counters and substituted function
counts are separate in `evidence/evidence.json`. All raw 15 timings are retained.
Do not interpret original `cache_key=0` as removing the 64 identity checks.

## Other exact commands and artifacts

Using the same wrapper and cwd:

1. `observations.py`: one untimed original-wave call, recursive copy profile,
   actual-output RED negative, original alias boundary and exact structural
   comparison. Initial run failed before setup on a missing scratch parent;
   fixed once, then successful. No benchmark or control was rerun.
2. `retain.py`: reuse the same H3 function for the constructor-accepted bytearray
   public negative; render scratch candidate diffs from the actual frozen file;
   collect result artifacts. No new timing/control.
3. `git apply --numstat /data/ycfeng/tmp/d29_emission_immutable_work/candidate_artifacts/h2.diff`:
   4+/1-, **5 gross/net +3**. Same directory's `h3.diff`: 4+/2-, 6 gross/net +2.
   `combined.diff`: 8+/3-, 11 gross/net +5.
   H3/combined are retained diagnostic artifacts, not approved changes.
4. Final filesystem receipt reruns `pyc_manifest.py` read-only, compares the
   exact previous manifest and checks Git status/HEAD. All 284 prior paths,
   owners, sizes, mtimes and SHA remain equal, with 33 directories; tracked
   source stays `75404143` and clean. No deletion/chown.

Final script hashes are exact in `evidence/filesystem_receipt.json`; source-bound
scripts are retained as `.py.txt` in evidence and executable originals only in
own scratch. No source/test/peer file was edited.

## Limits

Single actual layer-60 decode wave on a shared CPU host, not full workload or
cross-layer evidence. Setup/pricing/flush/ledger encoding and key priming are
excluded from performance numbers. Medians include all 15 samples, including
slow first samples; processes are serial, not paired interleaving. H1's protocol
placement and H2's final interface/domain guards are parent-owned designs.
No persistent regression test or full CLI/GPU/native verification was run.
