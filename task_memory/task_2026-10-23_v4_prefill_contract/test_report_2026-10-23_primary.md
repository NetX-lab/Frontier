# Primary isolated-prefill acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Added merged-source verification and retained the outer-reporting error. |
| 2026-10-23 | Recorded root reruns, independent review and complete fixed-source matrix acceptance. |

Root source:2f87828; matrix source/harness576d278156ea21b209da33791337bd195047ae3e, clean for every case. Python3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`. Workdir `/data/ycfeng/Frontier/.worktrees/v4-prefill-contract`; `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH` set to this worktree, `TMPDIR=/data/ycfeng/tmp` (affected tests used its opencode-v2-tmp/opencode subdirectory).

## Checks

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Affected unit file | No failures/errors/skips |144passed,0failure/error/skip | PASS |
| Accepted isolated scopes | Exact CSV median, admissionfalse |94exact (50event44kernel),0absolute/relative error | PASS |
| Legacy case | Original outcomes |102positive/72typed negatives | PASS |
| Comprehensive default matrix | >=50settings, baseline equality |77/77identical,0missing/failed/provenance cases | PASS |
| Retained matrix evidence | Actual bytes equal indexed hashes |451files; allcase source/harness576d278clean | PASS |
| Cache identity | Same filenames |519filenames,0differences | PASS |

Source remains3maintained files/197changed lines/core55. Dual-axis review: no code findings; one Spec documentation pointerP3 fixed using the stable decision heading. The initial owner regression decision is historical: repository AGENTS.md:796-801 required the matrix, so root ran it after the source/affected checks stabilized.

## Commands and evidence

Common command prefix: `env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-prefill-contract TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B`.

```text
-m pytest tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_prefill_contract_2f87828.xml
tests/integration/check_dsv4_measured_scopes.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --output /data/ycfeng/tmp/glm53_dsv4/root_prefill_contract_2f87828_accepted.json
tests/integration/check_dsv4_measured_scopes.py --attention-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144 --compute-root /data/ycfeng/tmp/glm53_dsv4/h200_dsv4_compute_results_20261021/9f58dbe --output /data/ycfeng/tmp/glm53_dsv4/root_prefill_contract_2f87828_legacy.json
tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/v4-prefill-contract --label candidate --output-root /data/ycfeng/tmp/glm53_dsv4/prefill_contract_576d278_20261023 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure
tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/glm53_dsv4/prefill_contract_576d278_20261023 --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --candidate-label candidate
```

All commands exit0 and their outputs were inspected. Exact selected output/source SHA256 values are in `primary_acceptance.json`; logs are adjacent to the listed output roots. An initial root receipt inspection used results.json instead of the actual results.jsonl; directory inspection corrected this read path, without rerunning the capture or changing any data.

Result: ACCEPTED_ISOLATED_PREREQUISITE. No full predictor/physicalTP8/PD/graph/serving/E2E admission. Native06 and its frozen source are independent.

## Primary integration verification

Merge e89b45da0346d2d3de7c735d47a00c4a580193ad carries the exact three accepted source-file hashes. `git diff --name-only 576d278156ea21b209da33791337bd195047ae3e e89b45da0346d2d3de7c735d47a00c4a580193ad -- frontier tests data .real-engine` is empty, retaining the complete fixed-source default-matrix evidence.

Repeated the affected pytest and both checker invocations from the primary worktree with the same Python/environment/input arguments above and output prefix `/data/ycfeng/tmp/glm53_dsv4/root_prefill_contract_merged_e89b45d`. Outcomes:144unit tests, zero failures/errors/skips;94accepted scopes;102legacy positives/72typed negatives. Source and output hashes are in `integration_receipt.json`.

The outer two-check orchestration printed `r['typed_negative_count']` after the accepted checker had passed94, but that field exists only in the legacy report. It raised `KeyError: 'typed_negative_count'` and did not run the second command. Inspection confirmed accepted keys/status/94outputs and missing legacy output; the source and accepted report were preserved, and only the unfinished legacy command was subsequently executed successfully. This reporting error is not a consumer failure.
