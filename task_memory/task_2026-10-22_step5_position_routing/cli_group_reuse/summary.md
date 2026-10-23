# D29-MODEL-GROUP-REUSE summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive the completed three-path repair and committed-source verification. |
| 2026-10-23 | Separate shared-suite and parent dual-trainer evidence; records P2 corrected, reviewer confirmation pending. |

## Outcome

Code **3763e5b87a845eac186cfd481f0b56cffb2dd045**, e41-based isolated branch
feat/step5-d29-group-reuse. The existing position-query and FFN-binding consumers
share `_step5_routed_layer_group(layer_id)`, which resolves an immutable full
routed-group tuple once per fixed-model predictor. Current-layer selection and
all source/family/native-context/estimator/exact duration/count/plan checks remain
live. No global cache, invalidation, registry/profiler API or numerical change.

## Evidence

- Real dual-consumer RED24resolutions instead of1, followed by GREEN.
- Six-file shared consumer/trainer selection **81PASS,1SKIP,0FAIL**,166.49s;
  it does not include the independent trainer. Parent repeats81PASS/1SKIP164.431s.
- Separate parent frozen88fe6c65 trainer/preflight and independent fit/cache
  selection **2PASS,0FAIL/ERROR/SKIP**,10.327s; not added to81. Both XML hashes
  were inspected and match; exact command/artifacts are in the test report.
- Original main07 P6128+idle1 pricing without external reuse: committed candidate
  **0.945/0.369s**,88 descriptions; explicit cold/warm counter **88/0** at
  **0.919/0.369s**. Costs/proofs/full-plan keys match baseline SHA256d2c141f….
- Profile: **1 resolver/88 descriptions** instead of12/1056; cold validation
  remains. Backend JSON decoding is the largest residual warm-point cost.
- Fixed lifetime and no supported mutation references: plan.md. Regression uses
  same-model independent predictors and a distinct formal model, layers3/88,
  actual bindings/points/proofs and source/family/count negatives.

## Exact scope and interfaces

Only three edit paths, **+65/-6=71/105 gross**:
- workload:13step,143/175retained.
- predictor:4step,111/128retained.
- test:54step,254/260retained.

Source17/45,test54/60 PASS. CORE retained**3932/4404**,global9175/10000;
111allocated paths unchanged. AST/whitespace PASS; isolated3dac submodule pin
clean and unchanged. Primary donor allocations were not edited.

Consumer-local interface: `_step5_routed_layer_group(layer_id)` returns the same
actual frozen group for the requested layer from this predictor's own tuple.
No cross-predictor sharing; no selected estimator/query/cost/proof cache added.

Deliverables in this task directory:
- `group_reuse.diff`, `budget.json`.
- `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`.
- `test_report_2026-10-23_model_group_reuse.md`, raw selected logs/results/profile.
- `probe_source_pin.diff` (initial source-only changes), `probe_complete.diff`
  (additional observation-only per-iteration counter).

Full raw logs and source-bound scratch probe:
`/data/ycfeng/tmp/d29_group_reuse_runtime_loop.py` and
`/data/ycfeng/tmp/d29_group_reuse_{pricing01,pricing02,pricing_count01,profile01}/`.
The probe enforces code revision3763e5b8; later records commits do not change code.

## Handoff boundary

No unresolved owned source issue. Standards No findings; Spec found no source
defect but one records P2 coverage overclaim. Wording is corrected and separate
parent evidence recorded; confirmation remains pending with reviewer
`ses_e97df4d82ffetkqpyEbPP2iCA1`. This records-only correction changed no
code/test/script bytes or3dac pin and ran no new tests. Parent review/freeze and
final-source acceptance remain pending. This short-loop success does not pass
the original2400s offline/online
full-workload gate or native/GPU/accuracy qualification. No full CLI, strict77,
full-unit, platform, network, dependency install or extra agent was invoked.
Secondary backend/typed-initialization observations stay diagnostic and require
separate primary ownership/authorization.
