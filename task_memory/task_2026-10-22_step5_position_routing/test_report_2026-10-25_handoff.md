# D29 Code and Documentation Handoff Validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Complete parent39PASS, source/cap/artifact checks and final committed UNADMITTED handoff binding; retain checker diagnostics. |

## Scope and outcome

Task: D29-EP-LANE-LEDGER-IMPLEMENTATION handoff. Primary repository:
`/data/ycfeng/Frontier/.worktrees/step5-preview`. Candidate repository:
`/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream`, branch
`fix/d29-ledger-stream`.

**PASS_COMMITTED_HANDOFF_BOUNDARY_UNADMITTED.** Initial source6c2f179c and final
caller1f6b431478e2351e3ca6819f707cccbc3e4336e2 are committed; owner records
dd3b6a69a17365b994a51ea7641197e768c8a237 are clean and source-equivalent.
Allocation and user decisions are committed in886945f3/imported924427c0. No final
ledger production admission or whole-CLI/native/G3/E2E acceptance is claimed.

## Completed primary checks

Small metadata/document commands use `python3 -B` and a60000ms tool timeout;
they do not import Frontier or inspect model/CSV/trace payloads. Source-focused
owner commands use the existing Python3.10.6 environment and configured30GiB
systemd/60s/single-thread envelope recorded in the implementation report.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Reallocation sum | Four step caps total135; new caps36/20/65/14 |135 | PASS |
| Zero-sum transfer | New-path allowance plus adjustments sums0 |0 | PASS |
| Authored allocation diff | No whitespace errors |git diff --check exit0 | PASS |
| Quota decision | USER_CONFIRMED, not fabricated measurement |Sufficient quota recorded from YC; measurement flagfalse | PASS |
| Document maintenance | Eight edited documents have date/history |8/8 | PASS |
| Current prompt precedence |2026-10-25 before2026-10-23 history |Confirmed | PASS |
| Required limits |6/6 shape ceiling and2400s gate retained |Confirmed | PASS |

Exact initial allocation verification:

```bash
python3 -B - <<'PY'
import json
import subprocess
from pathlib import Path
base = Path('task_memory/task_2026-10-22_step5_position_routing')
p = base / 'repairs/ep_lane_ledger_scope_amendment.json'
a = json.loads(p.read_text())
assert sum(a['path_step_ceilings'].values()) == a['repair_delta_ceiling'] == 135
assert sum(x['ceiling'] for x in a['new_paths']) + sum(x['delta'] for x in a['path_ceiling_adjustments']) == 0
assert a['path_step_ceilings']['frontier/metrics/metrics_store.py'] == 36
assert a['path_step_ceilings']['tests/performance/step5_actual06_iteration.py'] == 14
paths = [str(base / n) for n in ('requirements.md', 'progress.md', 'ledger_memory/requirements.md', 'ledger_memory/plan.md', 'repairs/ep_lane_ledger_scope_amendment.json')]
subprocess.run(['git', 'diff', '--check', '--', *paths], check=True)
print(json.dumps({'allocation': 'PASS', 'step_cap': 135, 'transfer': 'ZERO_SUM', 'authored_diff_check': 'PASS', 'paths': paths}, indent=2))
PY
```

The later quota/document command parses
`storage_readiness/quota_confirmation.json`, checks USER_CONFIRMED_SUFFICIENT,
no independent measurement, the root requirements pointer, dates/history in
eight maintained documents, current/historical prompt ordering and required
limits. It exits0 with `quota_record:PASS`, `maintained_documents:8`,
`handoff_precedence:PASS` and `budget_and_runtime_constraints_present:PASS`.
The final source checker is summarized below;the source and complete parent test
command are bound in `handoff_parent.json`.

Ordinary explicit-path `git add` reports the task directory as ignored (exit1).
Explicit `git add -f` on only the five owned allocation/decision documents
succeeds, followed by commit886945f3. No ignore/configuration setting changes,
unrelated stage, push, reset, worktree removal or broad cleanup occurs.

## Preserved failure evidence and verification limits

The initial owner report retains48PASS/3SKIP synthetic controls,30PASS existing
reporting checks, complete integration62PASS/4FAIL/6SKIP/rawexit1 and the actual
save_state PicklingError. Historical raw logs/IDs are not edited or relabelled.
The final corrected caller and checkout rerun are a separate result.

## Final primary observations

Parent command, working directory the candidate repository:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream:/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream/tests/unit \
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_ledger_handoff_parent_20261025 \
  XDG_CACHE_HOME=/data/ycfeng/tmp/d29_ledger_handoff_parent_20261025/cache \
  MPLCONFIGDIR=/data/ycfeng/tmp/d29_ledger_handoff_parent_20261025/mpl \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest \
  tests/integration/test_step5_measured_main.py::test_real_ep_lane_ledger_releases_rows_and_preserves_bytes \
  tests/integration/test_step5_measured_main.py::test_completed_position_ep_ledger_stream_failures \
  tests/unit/test_stage_reporting_contract.py tests/unit/test_ep_wave_trace_context.py \
  tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py \
  -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d29_ledger_handoff_parent_20261025/pytest \
  --junitxml=/data/ycfeng/tmp/d29_ledger_handoff_parent_20261025/pytest.xml
```

Expected39PASS/noFAIL-ERROR-SKIP. Actual39PASS,2.91s,exit0;Python3.10.6,
configured30GiB systemd scope starts. Effective cgroup peak is not measured.
Start/end UTC and exact argv are in `handoff_parent.json.primary_rerun`.

Metadata checker: `python3 -B`3.10.6 with30GiBRLIMIT_AS and60000ms tool timeout.
It compares committed/disk source SHA256, runs
`git diff --no-renames --numstat <0ca54025|1aee0af> 1f6b4314`, applies the
amendment to the accepted114-row TSV, checks116 caps/owner sums, source-equivalent
records head/clean candidate/pinned3dac checkout, hashes27 owner artifacts and
reparses raw JUnit. Actual124gross/net96/increment116;CORE4377/4404,
PRICING3955/3956,native1350/1582,source9682/9942,global9696/10000,zero violations.
Unchanged non-superproject usage is carried from the accepted receipt;this is
not the required independent composed audit.

Owner integration raw66PASS/6SKIP and EP30PASS are independently parsed and
hash-verified. The93 old statuses/9newPASS and actual save/load0/3,7124B ordered
outputs/unchanged old outputs are verified from the owner's committed receipts.
Primary does not rerun the entire integration or caller control. The39 direct
passes overlap owner results and are not added to them.

Initial metadata checker exits1 with `KeyError: 'kind'`:new amendment entries
name existing superproject paths without a kind field. Explicit superproject
classification at this metadata seam fixes only the checker;corrected full
check exits0. No gate weakening, source change or extra pricing invocation occurs.
The earlier optional `final_verification.json` lookup was not a delivered file;
the actual `artifact_manifest.json` is present and all27 entries verify.

Primary pytest log/XML/execution are byte-identically archived in
`handoff_parent_evidence/`; its archive manifest records3 files/8298B and prompt
removal of the three completed scratch duplicates. The metadata KeyError text
is retained separately. No active/cited caller or other-session input is deleted.

Accepted production remains9566source/9580global. Clean composed candidate audit,
independent Standards/Spec, final frozen strict77/full-unit/supplements,
production integration, current cohesive timing, worker/cloud execution readiness,
full6144/1024 CLI, T2/source/package/native/G3/E2E remain next-session gates.
Shape invocation allowance remains6/6used. No shape, pricing wave, fullCLI,
worker/GPU, cloud mount or quota-discovery run is performed for this handoff.

Quota sufficiency is explicitly user-confirmed, not independently measured.
Worker append throughput and whole-workload RSS remain unverified. Original
raw unit8028PASS/279FAIL/32ERROR/281SKIP/exit1 and strict77/77 apply only to the
previously accepted frozen source, not this ledger candidate.
