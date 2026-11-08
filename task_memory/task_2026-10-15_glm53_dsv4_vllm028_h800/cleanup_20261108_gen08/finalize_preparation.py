#!/usr/bin/env python3
"""Correct timestamp precision using frozen hashes; keep initial classification.

No candidate content is reread. The four snapshots remain declined due to their
actual allocated bytes differing from the assignment table by 1,536 bytes.
"""
import ast
import datetime
from pathlib import Path
import prepare_cleanup as p


def main():
    original = p.load(p.RECORD/'scoped_inventory_before.json')
    initial = p.load(p.RECORD/'deletion_manifest.json')
    p.require(p.hash_row(p.RECORD/'scoped_inventory_before.json')['sha256']==initial['before_inventory_sha256'],
              'Initial baseline changed')
    amended_scripts = {str(p.RECORD/name) for name in
                       ('prepare_cleanup.py','execute_cleanup.py','verify_cleanup_final.py')}
    for row in original['entries']:
        if row['path'] not in amended_scripts:
            p.require(p.metadata(row['path'])['signature']==row['signature'],
                      'Scoped input changed since original hash: '+row['path'])
    rows=[p.hash_row(e['path']) if e['path'] in amended_scripts else e for e in original['entries']]
    extra=[p.RECORD/name for name in ('finalize_preparation.py','scoped_inventory_before.json','deletion_manifest.json')]
    rows.extend(p.hash_row(path) for path in extra)
    rows.sort(key=lambda e:e['path'])
    bypath={e['path']:e for e in rows}
    group=initial['groups'][1]
    log=group['entries'][0]
    evidence=p.load(p.LAUNCH/'launcher_log_evidence.json')
    ns=log['signature']['mtime_ns']
    timestamp=(datetime.datetime.fromtimestamp(ns//1000000000,datetime.timezone.utc)
               +datetime.timedelta(microseconds=(ns%1000000000)//1000)).isoformat()
    p.require(timestamp==evidence['mtime_utc'] and log['sha256']==evidence['sha256']
              and log['bytes']==evidence['logical_bytes'] and log['allocated_bytes']==evidence['allocated_bytes'],
              'Launcher content/identity actually differs from evidence')
    p.require(group['errors']==['Launcher size/allocation/hash/mtime differs from preserved evidence'],
              'Unexpected initial log rejection')
    group.update(eligible=True,errors=[],mtime_precision_correction={
        'expected_utc':evidence['mtime_utc'],'actual_utc_microsecond_truncated':timestamp,
        'original_check_error':'ns/1e9 float conversion rounded .161655503 seconds to .161656; evidence truncates to .161655',
        'sha256_read_count':1})
    p.require(initial['groups'][2]['eligible'] is False
              and initial['groups'][2]['observed']['allocated_bytes']==300761088,
              'Snapshot mismatch changed; inspect rather than relax classification')
    before=dict(original,entries=rows,summary=p.totals(rows),finished_utc=p.now(),
                individual_files=original['individual_files']+list(map(str,extra)),
                correction_note='Original hashes retained; only author-owned amended scripts rehashed; no second read of the launcher log.')
    p.write('scoped_inventory_before_final.json',before)
    entries=[e for g in initial['groups'] if g['eligible'] for e in g['entries']]
    initial.update(created_utc=p.now(),entries=entries,planned=p.totals(entries),
                   before_inventory_sha256=p.hash_row(p.RECORD/'scoped_inventory_before_final.json')['sha256'],
                   initial_manifest_sha256=p.hash_row(p.RECORD/'deletion_manifest.json')['sha256'])
    initial['limitations'].append('Four snapshots declined: 300,761,088 allocated bytes observed versus 300,759,552 expected; +1,536 bytes. All four recorded sizes and SHA256 match.')
    p.write('deletion_manifest_final.json',initial)
    for script in sorted(p.RECORD.glob('*.py')):
        ast.parse(script.read_text(),filename=str(script))
    p.require(p.totals(entries)=={'files':174,'bytes':4410876394,'allocated_bytes':2193342464},
              'Final eligible totals differ')
    print('PASS finalized: 174 eligible files, 4 retained snapshots, unchanged log SHA256 reused, 4 scripts parse',flush=True)


if __name__=='__main__':
    main()
