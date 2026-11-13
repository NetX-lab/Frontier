# Affected fixture parent acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Review two-file repair and independently verify all28 affected tests on integrated9974acdb. |

Source test commit f03a1157, owner receipt fe88a3a8; parent integration
`9974acdb88f5c47166fee28dec564b7cc782ca1a` in
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-01`.
Root-reviewed delta: only the two authorized tests,+45/-27=72/80 gross.
Production Python is byte-identical to source-reviewed bfbc87ad; the other code
tree change is the explicitly approved gitlink pin to already verified3dac53c.

## Review

Parent inspected the complete two-file diff and the final rejection assertions.
Source fixtures derive canonical descriptors/census from real Batch/dummy,
forward agreement and source materialization. Runtime fixtures use the existing
bound real model/replica/registry, actual layer88 group and CPU plan writer;
hard-coded model/feature lambdas are removed. Exact buffer/context and retained
model markers are checked. Foreign context and absent physical-row rejection,
both trainer/preflight/fit/cache assertions and source0/4 cases remain.
No production relaxation, test deletion or added skip. Targeted review: PASS.

## Parent execution

```bash
timeout 120s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
  tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_grouped_gemm_padded_features.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_fixtures_parent02 \
  --junitxml=/data/ycfeng/tmp/d29_fixtures_parent02.xml \
  > /data/ycfeng/tmp/d29_fixtures_parent02.log 2>&1
```

Python3.10.6; actual28PASS,0FAIL/ERROR/SKIP,33.33s. All original five failure IDs
now pass. AST, two-file whitespace and72-gross delta checks PASS. Complete full-unit
comparison and strict77 are separate in-flight checks, not inferred from this run.

## Ongoing integration checks

- Full unit suite on9974acdb uses the same P10/environment/1800s limit and
  `--continue-on-collection-errors` as the retained D30 unit baseline. Output
  `/data/ycfeng/tmp/d29_unit_parent02.{log,xml}`; verdict pending notification.
- Strict77 uses a separate clean checkout `d29-fidelity-9974acdb`, preventing
  unit-test cache interference. Output `/data/ycfeng/tmp/d29_workflow_fidelity_20261023`.
  Original D30 candidate baseline713e6a8 was re-audited:77 definitions/cases,
  clean source/cache,451 artifact hashes and no evidence errors. New candidate
  run is pending, with original four jobs/clean-cache/continue-on-failure options.
- Fresh full position fixture generation runs on unchanged19a35630 in
  `verify-step5-core-closure-01`, output `/data/ycfeng/tmp/d29-parent-position-fixture03`.
  New generation is justified by closure of its former registry/backend/metadata
  blockers. Generator auxiliary limit2400s; the existing eventual CLI timeout
  remains2400s per offline/online mode. Input generation is not CLI acceptance.

These source/CPU checks do not establish native/GPU/graph/G3/numerical qualification.
