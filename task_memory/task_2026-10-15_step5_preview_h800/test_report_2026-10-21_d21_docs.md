# Test report: D21 documentation campaign update — 2026-10-21

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record D21-DOCS contract/link/current-reference, frozen-workload and doc-only staging verification; no runtime or GPU acceptance. |

## Scope and environment

- Task ID: **D21-DOCS**. Directory: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/`.
- Worktree/branch: `/data/ycfeng/Frontier/.worktrees/step5-preview`, `feat/step5-preview`; Linux/zsh.
- Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. No conda activation or package installation.
- Final D21 in `requirements.md` is parent-owned and read-only for this slice. Changes are public documentation and curated task Markdown only; no GPU/network/auth/push, source/test/config/README/frozen-package or other-worktree changes.
- Parent's supplied09:02UTC receipt, not a new platform query here: complete personal list0 nonterminal jobs;64 qualificationexp-1021-125354-923219 Succeeded08:20:59UTC;32 communicationexp-1021-143708-523626 Succeeded08:15:05UTC. Business judgments remain pending.

## Verification commands

Run from the worktree root. The following document-only Python command checks all changed Markdown file targets, current track references, critical release/resource clauses, Modification History placement, the30-ticket invariant, and frozen common-workload bytes. It does not execute native code, an E2E gate or a platform client.

```bash
/data/ycfeng/envs/frontier-py310/bin/python - <<'PY'
from pathlib import Path
import hashlib
import re
import subprocess

root = Path.cwd()
task = root / 'task_memory/task_2026-10-15_step5_preview_h800'
ticket_names = [
    '06-native-single-role-qualification.md', '08-native-pd-state-qualification.md',
    '09-native-h20064-qualification.md', '10-full-path-compute-profiles.md',
    '12-tp-ep-communication-measurements.md', '14-pd-metric-normalization.md',
    '15-mixed-model-request-path.md', '16-mixed-precision-compute-prediction.md',
    '22-predictor-bundle-cost-replay.md', '23-canonical-case-freeze.md',
    '28-h200-measurement-prediction.md', '29-three-model-regression.md',
    '30-evidence-support-acceptance.md',
]
files = [task / name for name in (
    'spec.md', 'plan.md', 'progress.md', 'summary.md',
    'd21_campaign_checkpoint.md', 'test_report_2026-10-21_d21_docs.md',
)] + [task / 'tickets' / name for name in ticket_names]
public = root / 'docs/general/step5-preview.md'
files.append(public)
links = fragments = checks = 0
for path in files:
    text = path.read_text()
    if path != public:
        assert 'Modification History' in '\n'.join(text.splitlines()[:8]), path
        checks += 1
    assert 'D21' in text, path
    checks += 1
    for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', text):
        url, _, fragment = match.group(1).partition('#')
        if not url or re.match(r'\w+://|mailto:', url):
            continue
        target = (path.parent / url).resolve()
        assert target.exists(), (path, url)
        links += 1
        if fragment:
            assert target.suffix == '.md', target
            headings = re.findall(r'^#{1,6}\s+(.+)$', target.read_text(), re.M)
            slugs = [re.sub(r'[^\w\s-]', '', h.lower()).replace(' ', '-') for h in headings]
            assert fragment in slugs, (target, fragment)
            fragments += 1
core = [task / 'spec.md', task / 'plan.md', public] + [task / 'tickets' / name for name in (
    '09-native-h20064-qualification.md', '23-canonical-case-freeze.md',
    '28-h200-measurement-prediction.md', '30-evidence-support-acceptance.md',
)]
for path in core:
    text = re.sub(r'\s+', '', path.read_text()).lower()
    for clause in ('h80032', 'h80064', 'h20032', 'h20064', '30%', '10%',
                   'basic', 'formal', 'preemptible', 'readiness'):
        assert clause in text, (path, clause)
        checks += 1
for path in [task / 'spec.md', task / 'plan.md', public]:
    text = re.sub(r'\s+', '', path.read_text()).lower()
    for clause in ('p16/d16', 'p32/d32', 'tp8/dp2/ep16/pp1', 'tp8/dp4/ep32/pp1',
                   '6144', '1024', '128', 'nonterminal', '<=32', 'sessions',
                   'queued', 'warmup', 'canonical02'):
        assert clause in text, (path, clause)
        checks += 1
assert len(list((task / 'tickets').glob('[0-9][0-9]-*.md'))) == 30
checks += 1
frozen = root / 'data/processed_traces/canonical_fixed_6144_1024_req128_qps1'
for path in sorted(frozen.iterdir()):
    if path.is_file():
        expected = subprocess.check_output(['git', 'show', '958d81a:' + str(path.relative_to(root))])
        assert path.read_bytes() == expected, path
        checks += 1
for row in (frozen / 'SHA256SUMS').read_text().splitlines():
    digest, name = row.split(maxsplit=1)
    assert hashlib.sha256((frozen / name.lstrip('*')).read_bytes()).hexdigest() == digest
    checks += 1
print('PASS files', len(files), 'relative_targets', links, 'fragments', fragments, 'assertions', checks)
PY
```

Staging/whitespace/scope checks:

```bash
git diff --cached --check
git diff --cached --name-only
git status --short
```

Only the explicit public doc and curated changed task Markdown are force-added; `requirements.md`, `.pi/` and `.cortexkit/` are excluded. The local commit's changed-file list is inspected after commit. No external push is performed.

## Results and limits

| Check | Expected / passing condition | Actual | Verdict |
| --- | --- | --- | --- |
| Final D21 contract | Four tracks, correct basic/formal and corresponding32 release edges,30% release distinct from10%, global personal H200 non-preemptible cap32 | Encoded in spec/plan/09/23/28/30 and public guide; D20 continues P0-first; original evidence/history preserved | PASS documentation scope |
| Markdown targets and current references | Every checked local target/heading exists; seven core docs contain four track/release/readiness labels |20 documents,213 local targets and7 heading fragments pass;158 assertions in the exact command above | PASS |
| Frozen common workload | Exact bytes equal958d81a plus all SHA256SUMS entries match |5 package files byte-identical to958d81a;4 checksum entries match; no workload edit | PASS |
|30-ticket invariant | Exactly30 existing ticket files; no new ticket |30 existing ticket files | PASS |
| Doc-only staging and whitespace | Only intended `.md` docs; no requirements/source/tests/config/README or unrelated paths; no whitespace error | Exact20-path staged/committed list and whitespace checked before completion | PASS after final scope check |
| Runtime/native/profile/calibration | Not part of this documentation slice | No run, no business/native verdict; all admission and formal dependencies remain open | NOT RUN |

Initial `python` command failed with `command not found` before reading files; rerun uses the explicit existing Python3.10.6 executable. One patch failed on incomplete context and made no edits; the scoped patch was corrected. First link pass checked212 targets with one missing planned report; the report is now present. First full-contract pass found the spec lacked the explicit `canonical02` supersession note; that current-reference gap was added from D21 and the rerun passed. No failed attempt changed product behavior or frozen bytes.

Final document command result: `PASS files 20 relative_targets 213 fragments 7 assertions 158`. This includes current links and retained historical Markdown links in all changed documents; code-formatted paths and external URLs are not network-verified. Current main-track tables/edges were also read against final D21: no H800-to-H200 accuracy edge, no30% basic-runtime blocker, no final-gate relaxation, no single-H20064 or retired-sweep requirement. Historical caps/statuses remain explicitly non-normative. First staged whitespace check found three pre-existing whitespace-only blank lines in the newly tracked historical progress record; their whitespace was removed without changing evidence, then the check was rerun.

Numerical observation/prediction values, absolute/relative errors and GPU-hour outcomes are not produced here.30% and10% are documented thresholds, not measured results. Runtime launch configuration is delivered by another owner; parent owns final acceptance/push.

Link validation resolves the local worktree, not a remotely published checkout. Parent-owned `requirements.md` and referenced historical records are not bulk-added by this slice; their publication/curation belongs to the parent. The later parent-owned D22 attention-reference decision was observed during the final requirements reread; D21 scheduling is unchanged, and D22 source/probe delivery is outside this slice.
