#!/usr/bin/env python3
"""Independent scoped audit: never delete files or import execution helpers.

Reruns hash the preserved files again and print a verdict without overwriting
existing verification records. The first run writes the after inventory/verdict.
"""
import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

R = Path(__file__).resolve().parent
T = R.parent
FORMAL = T / 'h3o_operator_costs/formal_p2_64_native_20261102'
GEN = FORMAL / 'generation08'
LAUNCH = T / 'native_track_20261020/g0d_launch'
LOG = LAUNCH / 'launcher.log'
PREVIOUS = T / 'cleanup_20261107_gen05_07'
SNAPSHOTS = {str(PREVIOUS / ('protected_hash_' + s + '.json'))
             for s in ('before', 'after', 'before_final', 'after_final')}
SSH = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh')
HOST_SHA = '20c553d72918382e08f7b636c9ab4d1896cd7f155c449fda5132534164340296'
ASSIGNMENT_SHA = '83fa01bd9d9374555bf791176a531437ac93767a67f21e6a838ed9a9d4ee7d16'


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(name):
    return json.loads((R / name).read_text())


def sig(s):
    return {k: getattr(s, 'st_' + k) for k in ('dev', 'ino', 'mode', 'nlink', 'uid', 'gid',
                                              'size', 'blocks', 'mtime_ns', 'ctime_ns')}


def audit(path):
    path = Path(path)
    if path.resolve() != path:
        raise RuntimeError('Noncanonical audit path: ' + str(path))
    s = path.lstat()
    kind = 'file' if stat.S_ISREG(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'other'
    if kind == 'other':
        raise RuntimeError('Unsupported scoped entry: ' + str(path))
    e = {'path':str(path),'kind':kind,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'signature':sig(s)}
    if kind == 'file':
        h = hashlib.sha256()
        with path.open('rb') as f:
            if sig(os.fstat(f.fileno())) != sig(s):
                raise RuntimeError('Changed before audit read: ' + str(path))
            for b in iter(lambda:f.read(8*1024*1024), b''):
                h.update(b)
            if sig(os.fstat(f.fileno())) != sig(s):
                raise RuntimeError('Changed during audit read: ' + str(path))
        if sig(path.lstat()) != sig(s):
            raise RuntimeError('Changed during audit hashing: ' + str(path))
        e['sha256'] = h.hexdigest()
    return e


def enumerate_root(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError('Missing/non-directory audit root: ' + str(root))
    out = [root]
    def raise_error(e):
        raise e
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=raise_error):
        out.extend(Path(parent)/s for s in dirs+files)
    return out


def total(rows):
    rows = [e for e in rows if e['kind']=='file']
    return {'files':len(rows),'bytes':sum(e['bytes'] for e in rows),
            'allocated_bytes':sum(e['allocated_bytes'] for e in rows)}


def authorized_file(s):
    p = Path(s)
    return s == str(LOG) or s in SNAPSHOTS or (p.is_relative_to(GEN) and p != GEN)


def save_once(name, obj):
    with (R/name).open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    started = stamp()
    errors = []
    def check(ok, reason):
        if not ok:
            errors.append(reason)
    m, receipt, before = load('deletion_manifest_final.json'), load('deletion_receipt.json'), load('scoped_inventory_before_final.json')
    check(audit(R/'deletion_manifest_final.json')['sha256']==receipt['manifest_sha256'], 'Frozen manifest hash differs')
    check(audit(R/'scoped_inventory_before_final.json')['sha256']==receipt['before_inventory_sha256']==m['before_inventory_sha256'],
          'Frozen baseline hash differs')
    check(audit(R/'assignment.md')['sha256']==receipt['assignment_sha256']==ASSIGNMENT_SHA, 'Assignment hash differs')
    check(audit(R/'cleanup_scope_processes.json')['sha256']==HOST_SHA, 'Supplied host-process evidence copy differs')
    check(receipt['process_prerequisite']==m['process_prerequisite'], 'Process prerequisite receipt differs')
    check(load('cleanup_scope_processes.json')['clear_for_deletion'] is True, 'Host evidence is not clear')
    check(audit(R/'deletion_journal.jsonl')['sha256']==receipt['journal_sha256'], 'Journal hash differs')
    bmap = {e['path']:e for e in before['entries']}
    planned = {e['path']:e for e in m['entries']}
    deleted = {e['path']:e for e in receipt['deleted']}
    removed = {e['path']:e for e in receipt['removed_directories']}
    check(len(bmap)==len(before['entries']) and len(planned)==len(m['entries'])
          and len(deleted)==len(receipt['deleted']) and len(removed)==len(receipt['removed_directories']), 'Duplicate receipt/inventory path')
    check(set(deleted)<=set(planned), 'Deleted path outside frozen manifest')
    check(set(removed)<={e['path'] for e in m['authorized_empty_directories']}, 'Removed directory outside frozen manifest')
    check(not receipt['errors'], 'Execution has errors')
    check(receipt['declined']==[{'group':g['id'],'paths':g['paths'],'reasons':g['errors']}
          for g in m['groups'] if not g['eligible']], 'Declined groups differ from classification')
    for s,e in deleted.items():
        check(authorized_file(s) and e==planned.get(s) and e['signature']['nlink']==1
              and re.fullmatch('[0-9a-f]{64}',e['sha256']) is not None, 'Unsafe or incomplete deleted-file record: '+s)
        check(not os.path.lexists(s), 'Deleted file still present: '+s)
        check(dict(bmap[s],group=e['group'])==e, 'Deleted-file size/hash differs from baseline: '+s)
    for s,e in removed.items():
        check(Path(s).is_relative_to(GEN) and not os.path.lexists(s), 'Removed directory unauthorized or still present: '+s)
        check(e['kind']=='directory' and all(e['signature'][k]==bmap[s]['signature'][k]
              for k in ('dev','ino','mode','uid','gid')), 'Removed directory identity differs: '+s)
    all_targets_absent = not os.path.lexists(GEN) and not os.path.lexists(LOG) and all(not os.path.lexists(s) for s in SNAPSHOTS)
    check(not os.path.lexists(GEN) and not os.path.lexists(LOG), 'One or more eligible assignment targets still present')
    check(all(os.path.lexists(s) for s in SNAPSHOTS), 'One or more declined snapshot files is missing')
    events = [json.loads(line) for line in (R/'deletion_journal.jsonl').read_text().splitlines()]
    by_event = collections.defaultdict(list)
    for e in events:
        by_event[e['event']].append(e)
    for name, rows in (('will_delete_file',deleted),('deleted_file',deleted),
                       ('will_remove_empty_directory',removed),('removed_empty_directory',removed)):
        check(len(by_event[name])==len(rows) and {e['path'] for e in by_event[name]}==set(rows), 'Journal count/path mismatch: '+name)
        for e in by_event[name]:
            check({k:v for k,v in e.items() if k not in ('event','utc')}==rows.get(e['path']), 'Journal data differs: '+name)
    check(not (set(by_event)-{'will_delete_file','deleted_file','will_remove_empty_directory','removed_empty_directory'}),
          'Unexpected journal event')
    for i in range(0,len(events),2):
        pair = events[i:i+2]
        check(len(pair)==2 and pair[0]['path']==pair[1]['path']
              and (pair[0]['event'],pair[1]['event']) in
              (('will_delete_file','deleted_file'),('will_remove_empty_directory','removed_empty_directory')),
              'Journal intent/completion order mismatch')
    roots = [FORMAL,LAUNCH,PREVIOUS,SSH]
    check(before['roots']==list(map(str,roots)), 'Scoped roots differ from required parents/SSH neighbour')
    paths = sorted({p for root in roots for p in enumerate_root(root)} | {Path(s) for s in before['individual_files']})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        after_rows = list(pool.map(audit,paths))
    amap = {e['path']:e for e in after_rows}
    expected_after = set(bmap)-set(deleted)-set(removed)
    missing = sorted(expected_after-set(amap))
    added = sorted(set(amap)-expected_after)
    changed_files, changed_dirs = [], []
    permitted_parent_metadata = set(map(str,(FORMAL,LAUNCH,PREVIOUS)))
    for s in sorted(expected_after & set(amap)):
        a,b = amap[s],bmap[s]
        if a['kind'] != b['kind']:
            changed_files.append(s)
        elif a['kind']=='file' and a!=b:
            changed_files.append(s)
        elif a['kind']=='directory':
            fields = ('dev','ino','mode','uid','gid') if s in permitted_parent_metadata else tuple(b['signature'])
            if any(a['signature'][k]!=b['signature'][k] for k in fields):
                changed_dirs.append(s)
    check(not missing and not added and not changed_files and not changed_dirs,
          {'unexpected_missing':missing,'unexpected_added':added,'changed_files':changed_files,'changed_directories':changed_dirs})
    summary = total(receipt['deleted'])
    expected = {'files':174,'bytes':4410876394,'allocated_bytes':2193342464}
    check(summary==receipt['deleted_summary']==m['planned']==expected, 'Deleted totals do not reconcile to assignment')
    check(len(removed)==receipt['removed_directory_count']==36, 'Removed directory count differs')
    check(sum(e['allocated_bytes'] for e in removed.values())==receipt['directory_allocated_bytes_at_removal'],
          'Removed directory allocation total differs')
    evidence = []
    for row in m['protected_evidence']:
        a=amap.get(row['path'])
        match = a is not None and a['sha256']==row['sha256'] and a['bytes']==row['bytes']
        evidence.append({'path':row['path'],'expected_sha256':row['sha256'],
                         'actual_sha256':a.get('sha256') if a else None,'matches_recorded_hash':match})
        check(match, 'Preserved evidence hash differs: '+row['path'])
    log_evidence = json.loads((LAUNCH/'launcher_log_evidence.json').read_text())
    deleted_log=deleted.get(str(LOG),{})
    check(deleted_log.get('sha256')==log_evidence['sha256'] and deleted_log.get('bytes')==log_evidence['logical_bytes'],
          'Deleted log hash/size differs from surviving launcher evidence')
    acceptance = json.loads((PREVIOUS/'lead_acceptance.json').read_text())
    snapshot_checks=[]
    for s in sorted(SNAPSHOTS):
        e=amap.get(s,{})
        old=acceptance['snapshot_files_not_committed'][Path(s).name]
        match = e.get('sha256')==old['sha256'] and e.get('bytes')==old['bytes']
        snapshot_checks.append({'path':s,'expected':old,'actual_sha256':e.get('sha256'),'actual_bytes':e.get('bytes'),'matches_recorded_hash':match})
        check(match, 'Retained snapshot differs from surviving acceptance: '+s)
    siblings=log_evidence['surviving_sibling_records']
    check(len(siblings)==29 and all(str(LAUNCH/s) in amap for s in siblings), 'One of 29 launcher sibling records is missing')
    def neighbour(root):
        rows=[e for s,e in amap.items() if Path(s).is_relative_to(root)]
        return {'root':str(root),'files':total(rows)['files'], 'matched_files':sum(e['kind']=='file'
                and e==bmap[e['path']] for e in rows), 'all_match':all(e==bmap[e['path']] for e in rows if e['kind']=='file')}
    neighbours=[neighbour(FORMAL/'generation08_run2'),neighbour(SSH),neighbour(PREVIOUS)]
    check(all(e['all_match'] for e in neighbours), 'Required preserved neighbour differs')
    completed=json.loads((FORMAL/'generation08_run2/result.json').read_text())
    check(completed['status']=='TOOLS_FINISHED' and completed['comparison_complete'] is True
          and len(completed['steps'])==7 and all(e['status']=='PASS' for e in completed['steps']), 'Preserved completed comparison no longer PASS')
    historical_checks=[]
    for row in json.loads((PREVIOUS/'record_files_final.json').read_text())['entries']:
        if row['path'] in SNAPSHOTS:
            continue
        a=amap.get(row['path'])
        if a:
            historical_checks.append({'path':row['path'],'expected_sha256':row['sha256'],'actual_sha256':a.get('sha256'),
                                      'matches_recorded_hash':a.get('sha256')==row['sha256'] and a['bytes']==row['bytes']})
    # Prior inventories may predate lead edits; report any historical divergence independently of before/after protection.
    out={'started_utc':started,'finished_utc':stamp(),'status':'PASS' if not errors else 'FAIL','errors':errors,
         'interpreter':sys.executable,'python_version':sys.version,'expected_deleted':expected,'actual_deleted':summary,
         'removed_directory_count':len(removed),'journal_event_counts':{k:len(v) for k,v in by_event.items()},
         'scoped_protection':{'expected_remaining_paths':len(expected_after),'actual_remaining_paths':len(amap),
             'hash_matched_files':sum(e['kind']=='file' and e==bmap[s] for s,e in amap.items()),
             'missing':missing,'added':added,'changed_files':changed_files,'changed_directories':changed_dirs,
             'allowed_parent_directory_metadata_changes':sorted(permitted_parent_metadata)},
         'preserved_evidence':evidence,'snapshot_recorded_hash_checks':snapshot_checks,'preserved_neighbours':neighbours,
         'launcher_sibling_records':{'expected':29,'actual_matched':sum(amap[str(LAUNCH/s)]==bmap[str(LAUNCH/s)] for s in siblings)},
         'previous_recorded_hash_checks':historical_checks,'process_prerequisite':receipt['process_prerequisite'],
         'limitations':m['limitations'],'declined':receipt['declined'],
         'assignment_all_targets_absent':all_targets_absent,'assignment_outcome':'implemented' if all_targets_absent and not errors else 'partial'}
    after={'observed_utc':out['finished_utc'],'roots':before['roots'],'individual_files':before['individual_files'],
           'entries':after_rows,'summary':total(after_rows)}
    if not (R/'verification_final.json').exists():
        save_once('scoped_inventory_after.json',after)
        save_once('verification_final.json',out)
    print(json.dumps({k:out[k] for k in ('status','errors','expected_deleted','actual_deleted','removed_directory_count',
              'journal_event_counts','scoped_protection','preserved_evidence','preserved_neighbours','launcher_sibling_records')},indent=2),flush=True)
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
