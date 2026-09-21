## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently reviewed the bounded first-swap causality, observed-state policy equivalence, route-prefix scope, and scheduled-token interpretation. |

# Independent review of DP workflow RCA

**Verdict: ACCEPT the scoped RCA.** The supplied evidence establishes the first reproduced close-pair owner inversion and demonstrates that changing the replay observation boundary removes that inversion without changing operator predictions. It does not establish that every full-case owner or admission mismatch has the same cause, and the RCA expressly preserves that limit. No production or evidence file was changed by this review.

Reviewer: `/root/route_instrumentation`, independent from the `/root/dp_workflow_rca` author. The reviewer previously implemented the host route logger and independently inspected formal-start boundary semantics; this review does not claim independence from that diagnostic instrumentation's design.

Inspected artifacts, all under this directory:

- `rca.md`
- `fresh_boundary_controls.json`
- `fresh_same_run_evidence.json`
- `same-run-01/same_run_join.json`

The review was limited to the requested causal claims. It did not repeat GPU runs, simulation controls, all helper tests, or the operator RCA.

## Findings

| Target claim | Verdict | Independent reconciliation and boundary |
| --- | --- | --- |
| First close-pair inversion is caused by using engine-enqueue order as routing order | ACCEPT for clients 5/6 in the reproduced fresh prefix | The actual route order is 5 then 6, with counts `[[0,3],[0,2]]` then `[[0,3],[1,2]]`; selected owners are DP1 then DP0. Enqueue order reverses the IDs. Both DES controls reach the same count sequence, but enqueue-input associates it with clients 6 then 5. Route-input restores the observed owners with unchanged predictions. This is a concrete counterfactual for this swap. It does not establish that operator timing never affects subsequent counts or other swaps. |
| Selector equivalence for 400 observed route states | ACCEPT as the analyzer's reported external-state equivalence result | The saved analysis reports 400 routes and 399 successive snapshot-state checks. The detailed join contains 100 formal requests; the reviewer independently recomputed all 100 selected owners from their actual recorded counts using `4*waiting+running` and the recorded tie-start order: 100/100 matched. The 300 warmup choices are represented by the analyzer's aggregate validation, not independently rerun by this bounded review. No result here means that an autonomous Frontier simulation reproduced all 400 states. |
| Route-input 8/8 demonstrates a complete workflow correction | Correctly rejected by the RCA | The control observes eight routes and seven prefill admissions; the eighth route is its stop boundary. The owner and member/token multiset results are 8/8 and 7/7. They remain prefix results, and the second mixed admission has a different prior token history. The RCA explicitly states this control is neither a sufficient production repair nor E2E acceptance. |
| `4096 / 4097 / 4098` describe the differing decode histories | ACCEPT as cumulative prior scheduled model-token counts | For the mixed batch admitting request 1, request 0 has respectively 4,096, 4,097, and 4,098 prior scheduled tokens in vLLM, enqueue-input Frontier, and route-input Frontier. Subtracting its completed 4,096-token prefill leaves 0, 1, and 2 preceding one-token decode steps. These values exclude request 0's new one-token work in the current mixed batch. They are not generated-token counts, allocated KV blocks, or directly measured attention metadata. |
| All old full-case DP/admission differences have been explained | Correctly rejected by the RCA | It distinguishes the first reproduced pair from the old 22-owner differences, records the three new route/enqueue inversions separately, and does not equate many propagated member differences with independent admission bugs. The fresh traces and old clean/diagnostic artifacts are not treated as one execution. |

## Required interpretation when communicating the result

The opening causal statement about admission/KV differences should be read with the later qualification: the control exposes missing route/engine visibility and step-boundary semantics, while operator timing can still contribute to the remaining progress differences. The current evidence does not independently decompose every source of that progress mismatch. It is appropriate to report a proven workflow boundary problem and an unresolved execution-progress gap; it would overstate the result to say the entire KV gap is exclusively workflow-caused.

The two controls independently normalize their first input event to time zero. Therefore request 1 appearing at `118.210053071 ms` in route-input versus `84.983801935 ms` in enqueue-input does not mean its real route occurred later than its real enqueue. The spacing difference is exactly explained by the unequal route-to-enqueue delays:

```text
request 0 route-to-enqueue = 39.147422183 ms
request 1 route-to-enqueue =  5.921171047 ms
difference                = 33.226251137 ms
route spacing             =118.210053071 ms
enqueue spacing           = 84.983801935 ms
```

The present one-arrival control starts Frontier model work immediately at either normalized first boundary. This changes available model progress before the next arrival as well as routing order. The pair-owner causality remains supported because the compared pair's decision-state sequence matches explicitly; the timing/progress effect is not a second isolated proof that every admission error has one cause.

For operator comparisons, `prior_scheduled_tokens=4096` with one current decode token is a logical prior-work count of 4,096 followed by one new model token. The corresponding post-current-step count would be 4,097. The same arithmetic yields 4,098 and 4,099 for the two Frontier controls. Actual KV tensor shapes and occupancy still require their own metadata; the RCA correctly avoids presenting cumulative ledger counts as a direct KV sensor.

## Verification record

Environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13, executable `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`.

Read-only JSON inspection extracted the two controls' match counts, close-pair decision states, and request 1 prefill membership; reconciled them with the actual join; then recomputed the saved 100 formal weighted-score choices and the timing-spacing identity above. Observed results:

```text
enqueue control: DP owners 6/8; prefill member/token multisets 5/7
route control:   DP owners 8/8; prefill member/token multisets 7/7
saved formal-policy recomputation: 100/100
prior request-0 scheduled tokens in request-1 prefill batch: 4096 / 4097 / 4098
route spacing minus enqueue spacing: 33.226251137 ms
```

No blocking issue was identified in the four requested claims. The interpretation qualifications above must remain attached to any final RCA summary. No source, helper, or author-owned artifact was modified.
