"""Validate deliverable links, statuses and pinned CPU receipts without new simulation."""
import json
from pathlib import Path
import re
from audit_inputs import HERE, load, digest


def main():
    ledger = load(HERE / 'ledger.json')
    audit = load(HERE / 'evidence/audit.json')
    retained = load(HERE / 'evidence/retained_checks.json')
    package = load(HERE / 'packages/nixl6144/cpu_check.json')
    assert audit['status'] == 'PASS_EVIDENCE_CHECKS_NOT_FORMAL_ACCEPTANCE'
    assert retained['status'] == 'PASS_RETAINED_CPU_CHECKS_NOT_FORMAL_ACCEPTANCE'
    assert package['status'] == 'PASS_CPU_PACKAGE_CHECKS'
    assert package['resource_policy']['planned_preemptible'] is True
    assert load(package['config'])['preemptible'] is True
    for path, sha in package['source_sha256'].items():
        assert digest(path) == sha, path
    assert len(ledger['inputs']) == 27 and len(ledger['attention_omissions']) == 48
    assert len(ledger['prior_audit_edge_map']) == 14
    assert sum(any(o['exact_workload_forward_queries'].values()) for o in ledger['attention_omissions']) == 7
    assert ledger['proposed_attention_check'] == audit['independent_attention']
    assert audit['independent_attention']['rows_per_independent_pass'] == 7105
    assert audit['independent_attention']['interpolated_queries'] == 0
    assert max(g['maximum_absolute_error'] for g in audit['independent_attention']['by_class']) < 0.05
    checked_links = 0
    for markdown in HERE.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', markdown.read_text()):
            assert (markdown.parent / target.split('#')[0]).exists(), (markdown, target)
            checked_links += 1
    for row in ledger['inputs']:
        assert row['status'] in {'QUALIFIED','OPEN','NEEDS_DECISION'}
        assert row['origin'] and row['consumer'] and row['criterion'] and row['evidence']
        if row['status'] != 'QUALIFIED': assert row['missing_evidence']
        for reference in row['evidence']:
            path, _, fragment = reference.partition('#')
            source = HERE / path
            assert source.exists(), source
            if source.suffix == '.json' and fragment.startswith('/'):
                value = load(source)
                for part in fragment[1:].split('/'):
                    value = value[int(part)] if isinstance(value,list) else value[part]
    for source in list(HERE.glob('*.py')):
        compile(source.read_text(), str(source), 'exec')
    report = dict(status='PASS_HANDOFF_INTEGRITY_NOT_FORMAL_ACCEPTANCE', links_checked=checked_links,
        input_rows=27, attention_omissions=48, queried_omissions=7, previous_audit_edges=14,
        status_counts={s:sum(r['status']==s for r in ledger['inputs']) for s in ['QUALIFIED','OPEN','NEEDS_DECISION']},
        package_source_digests_checked=len(package['source_sha256']),
        files={str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.rglob('*'))
               if p.is_file() and p.name not in ['handoff_check.json','handoff_check.log','codex_assignment_Q1_p2_input_qualification.md']})
    (HERE / 'evidence/handoff_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}, indent=2))


if __name__ == '__main__':
    main()
