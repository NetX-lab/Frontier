# D20 actual main initial execution report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Preserve deterministic offline/online initialization failure. |
| 2026-10-21 | Verify the authorized policy correction and preserve post-fix runtime failures. |

## Current result after policy correction

The approved graph-policy fix is verified, but the whole D20-ACTUAL-MAIN task remains **BLOCKED**. Existing scheduler semantics are used: full_decode_only selects physical full for a pure-decode capture hit, with physical none for prompt/tail or graph miss. `step5_cost.py` performs the supported-policy conversion and preserves both measured modalities for D. No profile enum, flags, guard, fallback or other production module changed.

| Verification | Actual | Verdict |
| --- | --- | --- |
| Pre-fix policy unit regression |5 failed /7 passed | RED on public policy and eager-D ownership |
| Focused manager tests |14 passed in9.88s | PASS |
| Final affected manager/public predictor/input union/model bundle tests |100 passed in57.60s | PASS |
| Actual CLI policy initialization regression |1 passed in64.11s; completion count explicitly0 | PASS boundary only |
| Full offline6144/1024 attempt03 | exit1 in23.585099s, request-free coverage error | FAIL runtime |
| Full online6144/1024 attempt04 | exit1 in18.785726s, same error | FAIL runtime |
| First-D tail/context, completed READ/release, final request metrics | Not reached | BLOCKED |

Final focused commands (same interpreter/environment as below):

```bash
env PYTHONPATH=.:tests/unit PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
  tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_public_predictor.py \
  tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_model_bundle.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d20_actual_main_affected03
env PYTHONPATH=.:tests/unit PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
  tests/integration/test_step5_measured_main.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d20_actual_main_cli_regression03
```

The CLI regression has trace arrival10s and existing integer time_limit1s, testing manager construction through the real subprocess without reaching the unrelated missing request-free runtime capability. It deliberately asserts zero completed requests. No simulator method is patched in the child. Profile production remains the explicitly labelled synthetic helper. This regression is not a replacement for the full attempts below.

The complete runner commands remain those in the original section, changing only output root to `/data/ycfeng/tmp/d20_actual_main_h80032_03 --mode offline` and `/data/ycfeng/tmp/d20_actual_main_h80032_04 --mode online`. Exact child commands/logs/status are retained in `evidence/policy_fix/{offline,online}/`; parsed configs remain at their recorded scratch paths with size/hash receipts. P16/D16 and8192 capacity are unchanged.

New exact defect: `dp_dummy_forward.py:78` -> normal predictor -> `step5_cost.py:64` raises `ValueError: Step measured request-free attention coverage is unavailable`. The workload is not a request-owned one-token decode: `DummyForwardBatch` has no attention metadata, skips attention kernels/cache writes and still executes projections/norms/MoE. Source scopes exist at `attention/step5.py:497`; the supplied request-owned synthetic profile rows cannot establish their measured cost. Parent must assign the missing producer/profile/runtime binding, not remove the rejection or substitute zero/regular-decode timing.

Intermediate verification limits: a broad225-case attempt exceeded120s after partial progress and has no complete result; no matching pytest process remained. The scoped final100-case result above is the affected-suite evidence. Initial CLI regression failures were test-only fractional time_limit (CLI accepts int) and hard-coded enum key3 (DECODE is5); both were corrected without changing workload semantics. Initial affected result98PASS/2FAIL identified obsolete global full and obsolete exclusion of eager-D input; tests were updated to require D coverage and preserve modality/role isolation. No matrix was launched. Parent's prior274PASS is historical, not relabelled as a post-fix aggregate.

Final artifact verification returned `PASS:14 retained hashes;2 parsed-config hashes/topologies;both runtime blockers;changed Python AST`. Authored changes pass `git diff --check`; original raw log bytes are retained. The receipt is `evidence/policy_fix/receipt.json`.

## Original pre-fix report

## Result

Task D20-ACTUAL-MAIN is **BLOCKED**, not PASS. Offline and online real `frontier.main` both exit1 with `ValueError: Unsupported Step graph_mode='full_decode_only'`. No request, output token, completed transfer or final metric evidence exists yet.

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-d20-integration`.
Interpreter `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6/GCC11.3.0; no conda activation, installation or environment changes. numpy2.2.6, pandas2.3.3, scikit-learn1.7.2, scipy1.15.3, plotly7.1.0, fasteners0.20, ddsketch3.0.1, pytest9.1.1.

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py \
  --fixture-manifest /data/ycfeng/tmp/d20_measured_compute_fixture_h80032_parent01/fixture_manifest.json \
  --output-root /data/ycfeng/tmp/d20_actual_main_h80032_01 --mode offline
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py \
  --fixture-manifest /data/ycfeng/tmp/d20_measured_compute_fixture_h80032_parent01/fixture_manifest.json \
  --output-root /data/ycfeng/tmp/d20_actual_main_h80032_02 --mode online
```

Both tools had60000ms timeout. Children use OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, disabled W&B and repository PYTHONPATH. Every complete child argument is in `evidence/{offline,online}/command.json` and `command.bash`; each effective parsed config is in `config.json`. READ input is in `evidence/synthetic_read_cost.json`.

Observed child wall times: offline2.040627269s, online2.684166906s. Source HEAD21f29728e2bfedf5713d602557ba96766e75e327. A separate AST/hash/config receipt check returned `PASS: runner AST;14 retained hashes;both exact topology/policy/error receipts`; it checked all14 files listed in `evidence/receipt.json`, all five TP/DP/EP/world-size fields per role, placement policy, capacity and exact error string. Initial `git diff --check` passed before staging untracked files. The staged check flagged only the raw logs' original final blank lines; those remain unchanged for artifact integrity. The authored-code/document check excludes the two raw logs. These checks do not validate the still-unreached success path.

## Acceptance observations

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| CLI topology | P16/D16; TP8/DP2/EP16/PP1; MoETP1 | Both logs print exact expected32-GPU topology | PASS configuration only |
| Graph/scheduling | full_decode_only; vllm_v1; vllm_load_balancing/v0.28.0 | Command/effective config retain these inputs | PASS configuration only |
| Data |6144/1024; offline arrival0, online arrival0.25s | Exact CSVs retained | PASS input only |
| Actual execution | Both child processes finish0 | Both finish1 during shared-manager init | FAIL |
| Request/token completion |1 complete request and1024 emitted outputs each | Not reached | BLOCKED |
| First D step |16 query /6128 cached | Input contract supplied; execution not reached | BLOCKED |
| NIXL READ | One complete3.5ms READ with exact per-rank bytes | Input generated from runtime cache geometry; transfer not reached | BLOCKED |
| Final metrics | Complete system/request/batch/transfer records | Only config exists | BLOCKED |

The production diagnosis is direct from the unmodified failing call chain and map definition, not a timing theory. No broad hypothesis campaign, fallback, source edit or workaround is warranted in this delegated scope. Parent owns the fix. Reproduction already takes about2s.

## Limits

CPU synthetic tables and the3.5ms READ cost are not native evidence or accuracy acceptance. CPU overhead is skipped. Fixture max_model_len8192 matches the actual run and profile table capacity but differs from canonical7168. One request per mode is not the formal128 requests/QPS1/three repetitions. The runner's success-validation path is not verified while initialization is blocked; first-tail and final-token checks must be completed after the parent fix.
