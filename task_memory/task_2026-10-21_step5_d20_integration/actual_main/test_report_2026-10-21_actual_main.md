# D20 actual main initial execution report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Preserve deterministic offline/online initialization failure. |

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
