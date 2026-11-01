# Test report: SKILL-WARM-FIX

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded the six warm-target documentation fixes, exact checks, and verification limits. |

## Scope and environment

- Task ID: `SKILL-WARM-FIX`; inherited decision: `D-WARM-TARGET`.
- Working directory for all checks: `/data/ycfeng/Frontier`; base: `98b680a24f676b5381c334f49a9729ea45847ee0`.
- Existing six modified skill documents were inspected before editing and retained as the starting work. The final combined diff contains 12 skill documents; no simulator, harness code or README change.
- Python: `/usr/bin/python3`, 3.10.6; pandas: 2.0.1 for the small CPU arithmetic check; shell: zsh. No conda activation or GPU environment was needed.
- The primary's scope clarification superseded the initial task-record location. Only this subagent's additions to the unrelated `skills/task_memory/task_2026-11-01_frontier_skills_readme/` were removed; original line counts and final content boundaries were rechecked. No further writes occurred there. `.rebuttal/` was not edited.
- No GPU job, engine run, staging, commit or external action was performed. Primary owns integration and commit.

## Observed checks

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Existing structure validator | Existing catalog, routes, paths and normative metric definitions pass | `v2_pressure=PASS` | PASS |
| Existing skill validator | Valid skill frontmatter and package | `Skill is valid!` | PASS |
| Whitespace | No diff errors | `git diff --check -- skills/frontier-calibration` exits 0 | PASS |
| Documents and references | Balanced fences and resolvable local links/relative entry pointers | 31 skill Markdown documents, 1 Markdown link; relative code-path references pass | PASS |
| Registry structure | Preserve schema, lanes, 20 entry identities, owners, calls, permissions and stop statuses | All compared fields unchanged | PASS |
| Tracked scope, README and index | Only skill documents modified, README byte-identical to base, no staged files | 12 tracked changed paths, all under `skills/frontier-calibration/`; README unchanged; index empty | PASS |
| Per-repeat warm arithmetic | Retained sets differ; each side recomputes its own windows, then sums numerators/denominators | IDs `b,c` use 3 s; ID `a` uses 1 s; both sides pool to TTFT 20 ms, TPOT 2 ms, E2E 1000 ms, 0.75 req/s, 9 tok/s | PASS |
| Shared-ID limitation | Filtering rows without changing the native shared formal set fails native admission | Missing-ID gaps equal each repeat's excluded set | PASS |
| Retained S33 and mode pairs | Per-repeat retained offsets and bilateral mode-pair observations use existing functions | Retained S33 bounds 2 and 1 ms; both sides have identical pooled mode observations and 3 distinct IDs | PASS |
| Original rows | Derived selection does not mutate all-request frames | Every original frame still contains `a,b,c` | PASS |

The arithmetic values above are synthetic expected/observed calculator values, not measured engine latency or simulator accuracy. Direct code execution establishes the arithmetic properties; source/event exclusion accuracy is not tested by synthetic IDs.

## Exact validator and document commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 skills/frontier-calibration/tests/pressure_scenarios.py skills/frontier-calibration
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/codex-home/skills/.system/skill-creator/scripts/quick_validate.py skills/frontier-calibration
git diff --check -- skills/frontier-calibration
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
import json
import re
import subprocess
root = Path('skills/frontier-calibration')
paths = [p for p in root.rglob('*.md') if p.name != 'site.local.md']
links = 0
for path in paths:
    text = path.read_text()
    assert text.count('```') % 2 == 0, path
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
        if target.startswith(('https://', 'http://', '#')):
            continue
        assert (path.parent / target.split('#')[0]).exists(), (path, target)
        links += 1
    for target in re.findall(r'`(\.\./(?:entries|references)/[^`]+\.md)`', text):
        assert (path.parent / target).exists(), (path, target)
old = json.loads(subprocess.check_output(['git', 'show', '98b680a2:skills/frontier-calibration/references/registry.json'], text=True))
new = json.loads((root / 'references/registry.json').read_text())
for key in ['schema_version', 'routing_authority', 'lane_registry']:
    assert old[key] == new[key], key
fields = ['name', 'kind', 'owner', 'calls', 'allows_code_change', 'stop_status']
assert [[e.get(k) for k in fields] for e in old['entries']] == [[e.get(k) for k in fields] for e in new['entries']]
changed = subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines()
assert all(p.startswith('skills/frontier-calibration/') for p in changed), changed
assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True) == ''
assert Path('skills/README.md').read_bytes() == subprocess.check_output(['git', 'show', '98b680a2:skills/README.md'])
print(f'documents={len(paths)} links={links} relative_code_paths=PASS registry_structure=PASS entries=20 tracked_scope=PASS changed_files={len(changed)} README_unchanged=PASS index_empty=PASS')
PY
```

## Exact CPU arithmetic command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 - <<'PY'
import sys
import pandas as pd
from tests.comparison.calibration.e2e_metrics_gap import metric_values
from tests.comparison.e2e_gate.gate import evidence_gaps, s33_vllm_ttft, mode_pairs

formal = ['a', 'b', 'c']
def rows(ttft, arrivals, completions):
    return pd.DataFrame({
        'request_id': formal,
        'arrival_s': arrivals,
        'completion_s': completions,
        'ttft_ms': ttft,
        'tpot_ms': [2.0, 2.0, 2.0],
        'request_e2e_time_ms': [(c-a)*1000 for a, c in zip(arrivals, completions)],
        'request_num_prefill_tokens': [10, 10, 10],
        'request_num_decode_tokens': [2, 2, 2],
        'endpoint_offset_proxy_api_ms': [1.0, 2.0, 3.0],
        'ttft_mode': ['in_phase']*3,
        'tpot_mode': ['in_phase']*3,
    }).set_index('request_id')
frontier = rows([10.0, 20.0, 30.0], [0.0, 2.0, 4.0], [1.0, 3.0, 5.0])
reference = [rows([10.0, 20.0, 30.0], [0.0, 2.0, 4.0], [1.0, 3.0, 5.0]),
             rows([10.0, 20.0, 30.0], [0.0, 2.0, 4.0], [1.0, 3.0, 5.0])]
# Synthetic event-bound identities, fixed independently of measured latency.
excluded = [{'a'}, {'b', 'c'}]
per_side = {'reference': [], 'frontier': []}
for repeat, e in zip(reference, excluded):
    retained = [i for i in formal if i not in e]
    for side, frame in [('reference', repeat), ('frontier', frontier)]:
        values = metric_values(frame.loc[retained])
        per_side[side].append(values)
    print('retained', retained, 'reference_window_s', per_side['reference'][-1]['request_throughput_rps'][1],
          'frontier_window_s', per_side['frontier'][-1]['request_throughput_rps'][1])
pooled = {}
for side, repeats in per_side.items():
    pooled[side] = {}
    for metric in repeats[0]:
        numerator = sum(v[metric][0] for v in repeats)
        denominator = sum(v[metric][1] for v in repeats)
        pooled[side][metric] = float(numerator / denominator)
assert pooled['reference'] == pooled['frontier']
assert pooled['reference']['ttft_ms'] == 20.0
assert pooled['reference']['request_throughput_rps'] == 0.75
assert pooled['reference']['token_throughput_tps'] == 9.0
for e in excluded:
    retained = [i for i in formal if i not in e]
    gaps = evidence_gaps(frontier.loc[retained], formal, {})
    assert set(gaps['missing_request_ids']) == e
bound_values = [s33_vllm_ttft(frame.loc[[i for i in formal if i not in e]])
                for frame, e in zip(reference, excluded)]
assert [v[1]['bound_ms'] for v in bound_values] == [2.0, 1.0]
pairs = {}
for n, (frame, e) in enumerate(zip(reference, excluded)):
    retained = [i for i in formal if i not in e]
    result = mode_pairs({f'vllm/{n}': frame.loc[retained], 'frontier/member': frontier.loc[retained]})
    for key, value in result.items():
        aggregate = pairs.setdefault(key, {'vllm': [], 'frontier': [], 'requests': set()})
        for side in ['vllm', 'frontier']:
            aggregate[side].extend(value[side])
        aggregate['requests'].update(value['requests'])
assert all(v['vllm'] == v['frontier'] and v['requests'] == set(formal) for v in pairs.values())
assert list(frontier.index) == formal and all(list(r.index) == formal for r in reference)
print('python', sys.version.split()[0], 'pandas', pd.__version__)
print('matched_repeat_pooling=PASS shared_id_limitation=PASS retained_s33=PASS mode_pair_pooling=PASS original_rows_intact=PASS')
print('pooled', pooled)
PY
```

## Manual instruction-boundary inspection

This is a document/source inspection, not an independent model-behavior test.

| Requirement or boundary | Decisive final document | Observed route |
| --- | --- | --- |
| Different exclusions per repeat | `entries/e2e-metrics-gap.md`, steps 2/3 | Per-repeat `F`, `E_r`, `W_r`, bilateral windows and pooling; replay receipt; warm reading decides status; native reading remains immutable |
| Valid complete but incompletely warmed run | `entries/groundtruth-run-check.md`; `references/call-graph.md` | Validity PASS remains separate from readiness; complete event evidence permits warm analysis |
| Preflight before runtime evidence exists | `entries/groundtruth-run-check.md`; `references/contract.md`, Warm steady state | Record policy, expected coverage and verifier; actual coverage/drain checked at formal boundary and post-run |
| PP>1, heterogeneous TP, push/notif-only or other NIXL build | `references/tool-boundaries.md` | Verify connection/rank identities; PP1 pull same-TP raw-count shortcut is conditional and setup completion remains required |
| Warmup-only reference fix | `references/contract.md`, Error-Source Classes; `entries/error-correction.md`; correction/approval/pitfall routes | `ground_truth_tool` collection policy and new run generation; unchanged serving build retains its identity |
| Scope definition | `SKILL.md`; `references/contract.md` | Introduction points to Warm steady state; contract alone defines excluded one-time scope |
| Slow request with no one-time event binding | Warm steady state; E2E step 2 | No latency-based exclusion; retain observed queueing and timing |
| Empty retained repeat, unbound events, unknown readiness, or full-row admission gaps | Warm steady state; E2E steps 1/2 | Insufficient evidence, no discarded valid repeat or favorable cell-wide union/intersection |

## Source evidence

- Inspected `tests/comparison/e2e_gate/gate.py:97-111` for shared-ID admission, `122-142` for reference pooling against one Frontier set, and `145-210` for mode-pair arithmetic.
- Inspected `tests/comparison/calibration/e2e_metrics_gap.py:123-162` for per-side window arithmetic, numerators/denominators and metric decisions.
- Read pinned vLLM source with `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` from `/data/ycfeng/Frontier/.real-engine/vLLM-BS`. Lines 583-725 show remote PP/TP handshake loops and the compatibility log emitted before completed agent registration. This supports the restricted counting rule, not a claim that all vLLM builds behave identically.

## Deliverables and limits

- Combined skill diff against `98b680a2`: `skill_warm_fix.diff` in this task directory.
  Exact generation command: `git diff 98b680a24f676b5381c334f49a9729ea45847ee0 -- skills/frontier-calibration > task_memory/task_2026-11-01_calibration_warm_target/skill_warm_fix.diff`.
  Rechecked against the live diff: PASS, 35,291 bytes, SHA-256 `3c7611e44ed42cd7cd7d9325e72fcc33639cf2e3c88380b961af3c0a4427888c`.
- Progress self-check: `check_progress_skill_warm_fix.md` in this task directory.
- No requested documentation finding remains unresolved in this subagent's inspection. Primary review/integration is unfinished.
- The shared-ID harness limitation remains explicit. No production warm-reading calculator or native collector was implemented; future calibration cases must retain and verify their case-local calculator/event receipts as the skill now requires.
- No benchmark accuracy claim, GPU timing, real event-coverage test, native run, or original published reading was altered or validated here.

## Primary archive check

The first staged whitespace check reported the raw `skill_warm_fix.diff` artifact's 34 single-space blank-context lines. These are required unified-diff context prefixes, not whitespace added to skill source. The primary verified that every flagged artifact line is exactly one space and that the artifact remains byte-identical to `git diff 98b680a2 -- skills/frontier-calibration`. The staged source/documents are checked separately with `git diff --cached --check -- . ':!task_memory/task_2026-11-01_calibration_warm_target/skill_warm_fix.diff'`; the raw diff is retained unchanged for review reproducibility.

## Final branch integration

The active branch imported the prerequisite skill changes (`c3ba1712`, `7077960f`, `33ff10fd`) and root warm-target commit `0cc4118a` as `f31b2b2d`. The only cherry-pick conflict was `skills/README.md`; it was initially resolved by retaining existing branch bytes. The subsequent structure check failed with `deployment-specific text 'vLLM-BS' in .../skills/README.md`. Inspection of `pressure_scenarios.py:372–380` showed that the newer validator also checks the parent README.

The primary recovered the prior explicit README/generalization request from `skills/task_memory/task_2026-11-01_frontier_skills_readme/requirements.md:9–11,24–35` at repository root. Commit `24ddbbb6` imports exactly `98b680a2:skills/README.md`; no new README prose or checker exception was introduced. The requested root README remains unchanged.

Final commands in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, Python3.10.6:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 skills/frontier-calibration/tests/pressure_scenarios.py skills/frontier-calibration
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/codex-home/skills/.system/skill-creator/scripts/quick_validate.py skills/frontier-calibration
git diff --check
```

Observed: `v2_pressure=PASS`, `Skill is valid!`, exit0/zero whitespace diagnostics. Direct comparison of `git ls-files skills/frontier-calibration` plus README found **34/34 files byte-identical** to the requested root; README also equals its `98b680a2` blob, SHA-256 `594fa59a99a6e3b07e01095ea7a29bebe1163f82dfd2b344552e73153004473b`. No unresolved conflict, new simulator run or GPU operation remains in this slice.
