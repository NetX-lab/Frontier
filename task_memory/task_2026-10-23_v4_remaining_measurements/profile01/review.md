# Native profile package review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent initial Standards/Spec reviews and root reproduction of the staged-entry failure. |
| 2026-10-23 | Both delta reviews closed the entry repair at d87f4c74 with no new findings; merged source at cc0227e9 for final package checks. |

## Initial review scope

Component: profile01 package preparation, baseline `1dc93c6f` through source `69f92c55` and records `62092da3`. Diff command: `git diff 1dc93c6f...62092da3`. The supplied `source.diff` covers the existing builder and the explicit profile plan, 139/300 changed lines. Reviewers inspected the task requirements/plan/proposal, repository/global instructions, source diff, generated package, delivery and CPU records. Neither reviewer ran tests or rehashed artifacts.

## Standards

Reviewer: `W128-PROFILE01-STANDARDS`, session `ses_e97ebd6b4ffexzGKcNI4N7jQzM`.

**No findings.** The builder reuses the existing launch/seal path, preserves accepted-runtime checks at preflight and actual staging, and binds configuration/source/workload/tool provenance. The fixed profile01 parameters have a recorded purpose. No actionable documented-standard breach or baseline smell was found in `build_candidate.py:44–70,107–126,152–166,202–215,247–282,297–322` or `profile_plan.json:1–44`. Source/config scope is two files. The existing 77/77 result is an explicit binding, not a new capture. This source review does not grant GPU/reference admission.

## Spec

Reviewer: `W128-PROFILE01-SPEC`, session `ses_e97ece46effeX9qSjHRv2ITgd3`.

**One P1: generated entry imports fail.** `build_candidate.py:157–164` flattens `native_pd_worker.py` to the package root, while that worker's top-level `vllm_replay` import still assumes repository-relative placement. The actual module is staged under `tests/comparison/dp_placement_pp`. `launch.py:30` imports the worker without that path, and `native_dummy_bootstrap.py:72–74` supplies only the package root to the child. Earlier CPU probes preloaded `pd_replay`, which inserted the missing path and masked the failure. This violates the working-entry requirement in `plan.md:24` and staged-extension verification in `proposal.md`. No other actionable omission or scope creep was found.

## Root reproduction and remediation

- Direct fresh import: `/data/ycfeng/tmp/glm53_dsv4/profile01_entry_red_acbrv5ib`; exact `ModuleNotFoundError: No module named 'vllm_replay'`.
- Actual copied launcher: `/data/ycfeng/tmp/glm53_dsv4/profile_entry_naaqrp6l`; fails at launcher line30/worker line19. No preloaded dependency, real run-manifest write or platform operation. The disposable copy's test manifest does not seal the actual candidate.
- Actual locked-image bootstrap environment with a fresh child: `/data/ycfeng/tmp/glm53_dsv4/profile01_bootstrap_entry_red02_7mv1c4cq`; same import error. The earlier callback-arity failure is preserved separately and is not entry evidence.
- Original delivering owner is correcting generated launcher/bootstrap paths through the builder, within the same two-file/300-line scope. Profile01 remains unchanged, unsealed and unsubmitted. Fresh profile02 requires independent recheck, actual entry/image verification and root acceptance before submission.

Initial summary: **Standards 0 findings; Spec 1 finding, worst P1.** Repair verification and runtime/GPU qualification remain pending.

## Repair re-review

Reviewed source `d87f4c74b9d5c0eec0958ba2b7351e76b04acf14`, records `a90c9dbbee53e2cabbfac73ba8490ffb958ce8eb`, same reviewer sessions as above. Delta `62092da3..d87f4c74` is 21 changed lines; cumulative maintained source/config scope is 152/300 lines in two files.

### Standards

**No findings.** The profile-only builder changes supply the actual staged dependency path in both entry environments (`build_candidate.py:177–181,275–279`; generated bootstrap72–74/launcher30–32). Source/staged hashes remain separately recorded (`source_lock.json:122–133`). Fresh02 rejects directory reuse and preserves profile01. No accepted-runtime edit, second launcher mechanism, scope breach or actionable smell was found. Exact byte preservation, owner CPU checks and matrix binding were supplied evidence; the reviewer did not re-execute them.

### Spec

**Original P1 source defect closed; no new actionable findings.** Both generated paths now resolve the staged dependency before importing the worker. The changes stay in the profile branch and preserve accepted runtime source. The reviewed owner controls use no preloaded `pd_replay` or external dependency-path supplement. Their bootstrap source/binary stand-ins remain explicitly limited. This closes the source issue, not full launcher/GPU qualification.

Root independently observed actual StepMind-interpreter launcher entry PASS (`profile_entry_1od7foss`) and real locked-image bootstrap-child PASS with 87 source/23 tool/20 binary pins (`profile02_image_entry02_vqtcre50`). An earlier root image assertion still required the old one-directory PYTHONPATH; its failure is retained and the corrected assertion requires exactly the generated two-directory path. Root merged at `cc0227e9` and proceeds with final frozen package checks. Final review summary: **Standards 0 findings; Spec 0 open findings (P1 closed).**
