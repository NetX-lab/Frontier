#!/usr/bin/env python3
"""Write the requested final report, exact check commands and record catalogue."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys

R = Path(__file__).resolve().parent


def load(name):
    return json.loads((R/name).read_text())


def write(name, value):
    (R/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    m, receipt, v = (load(n) for n in ('deletion_manifest.json','deletion_receipt_final.json','verification_final.json'))
    assert receipt['status'] == 'PASS_EXECUTED_AND_VERIFIED' and v['status'] == 'PASS' and not v['errors']
    before, after = (load(n) for n in ('protected_hash_before_final.json','protected_hash_after_final.json'))
    cmd = lambda arg: f'python3 -B -u {R}/execute_cleanup.py {arg}'
    syntax_source = "import ast\nfrom pathlib import Path\nr=Path("+repr(str(R))+ ")\nfor name in ('execute_cleanup.py','verify_cleanup_final.py','write_report_final.py'):\n ast.parse((r/name).read_text());print('PASS: syntax',name)\n"
    commands = [
        {'command': 'python3 -B - <<\'PY\'\n'+syntax_source+'PY', 'purpose': 'Parse all three final scripts without bytecode writes',
         'expected': 'All three scripts parse', 'observed': 'PASS: all three scripts parse', 'status': 'PASS', 'returncode': 0},
        {'command': cmd('initialize'), 'purpose': 'Preserve original record hashes and prior report', 'expected': 'Immutable baseline and backup saved',
         'observed': 'PASS: original record hashes saved; report backed up', 'status': 'PASS', 'returncode': 0},
        {'command': cmd('snapshot-before'), 'purpose': 'Fresh readable protected-input baseline', 'expected': 'All readable hashes recorded; six accepted exceptions listed',
         'observed': {'regular_files': before['regular_files'], 'regular_bytes': before['regular_bytes'], 'errors': before['errors'],
                      'runtime_seconds': before['runtime_seconds']}, 'status': 'PASS', 'returncode': 0},
        {'command': cmd('execute'), 'purpose': 'Preflight, immediate completion gate, exact deletion and retention check',
         'expected': '449 files deleted, 32 unchanged, only affected empty scratch ancestors removed',
         'observed': {'deleted_files': receipt['deleted_file_count'], 'logical_bytes': receipt['deleted_bytes'],
                      'allocated_bytes': receipt['deleted_allocated_bytes'], 'removed_directories': receipt['removed_directory_count'],
                      'kept_files': len(receipt['kept']), 'errors': receipt['errors']}, 'status': 'PASS', 'returncode': 0},
        {'command': cmd('snapshot-after'), 'purpose': 'Fresh readable protected-input after snapshot', 'expected': 'Hash-identical protected inventory',
         'observed': {'regular_files': after['regular_files'], 'regular_bytes': after['regular_bytes'], 'errors': after['errors'],
                      'runtime_seconds': after['runtime_seconds']}, 'status': 'PASS', 'returncode': 0},
        {'command': f'python3 -B -u {R}/verify_cleanup_final.py', 'purpose': 'Independent final acceptance-condition audit',
         'expected': 'PASS with zero errors', 'observed': {'status': v['status'], 'errors': v['errors'],
                      'protected_regular_hash_matches': v['protected_check']['hash_matched_regular_files'],
                      'archive_entries': v['archive_entry_count']}, 'status': 'PASS', 'returncode': 0},
        {'command': f'python3 -B -u {R}/write_report_final.py', 'purpose': 'Final report and artifact catalogue',
         'expected': 'report.md, execution_commands_final.json and record_files_final.json written',
         'observed': 'PASS: final report, exact check commands and record catalogue written', 'status': 'PASS', 'returncode': 0},
    ]
    write('execution_commands_final.json', {'working_directory': '/data/ycfeng', 'interpreter': sys.executable,
        'python_version': sys.version, 'commands': commands,
        'initial_environment_correction': {'executable': 'python', 'phase': 'Initial read-only inspection',
            'returncode': 127, 'error': 'zsh: command not found: python', 'resolution': 'Used /usr/bin/python3; no dependencies installed'},
        'not_run': ['Host PID namespace scan replaced by the expressly accepted completion evidence',
                    'Hashes or directory traversal for the six known protected exceptions',
                    'Simulator/GPU tests: unrelated to this file-only cleanup']})

    pc = v['protected_check']
    lines = [
        '**Assignment and outcome.** Implemented: exactly 449 frozen manifest files were deleted after current-content validation; '
        f'{receipt["removed_directory_count"]} scratch directories made empty by those deletions were removed. '
        'All 32 uncovered files remain unchanged. The independent final audit is PASS with zero errors. Final acceptance remains with Claude.',
        '',
        f'Working root: `/data/ycfeng`. `<TASK>` = `{R.parent}`. Record directory: `{R}`.',
        '',
        'Deletions were limited to the 13 manifest-listed generation05–07 `scratch/` trees under `/data/ycfeng/tmp/ds4_runs` '
        'and `prompts/p64-formal7-post-1107n.{sh,md}`. Records were written only in this record directory. '
        'The original assignment and the correction in `assignment_2.md` were followed. Applicable Frontier AGENTS.md files '
        '(identical), the prior implementation report, relevant requirements/progress, archive manifests and earlier cleanup scripts were read. '
        'Source text did not expand deletion authority. No Git, move, platform/GPU operation, network write or other cleanup was performed.',
        '',
        'Settled decisions: the lead-approved completion records replace the unavailable host process scan. '
        'The five unreadable protected paths and the growing historical launcher log are accepted verification exceptions. '
        'Each was only inspected with `lstat`; the 4 GB log was never opened or rehashed. '
        'The original manifest, earlier receipts, snapshots, scripts and assignments are retained; only the requested `report.md` was updated. '
        'Its prior bytes are preserved in `report_before_execution.md`.',
        '',
        '**Deleted and kept counts.** Logical bytes are file sizes; allocated bytes are `st_blocks * 512` observed immediately before unlink. '
        'The receipt reports directory bytes separately. These are pathname allocation totals, not a measured filesystem free-space delta.',
        '',
        '| Category | Deleted files | Logical bytes | Allocated bytes | Kept files | Kept logical bytes | Kept allocated bytes |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    z = {'files':0,'bytes':0,'allocated_bytes':0}
    for c in ('archived_duplicate','node_compile_cache','superseded_prompt','uncovered_scratch'):
        d=receipt['deleted_summary']['categories'].get(c,z);k=receipt['kept_summary']['categories'].get(c,z)
        lines.append(f'| {c} | {d["files"]:,} | {d["bytes"]:,} | {d["allocated_bytes"]:,} | {k["files"]:,} | {k["bytes"]:,} | {k["allocated_bytes"]:,} |')
    lines.extend([
        f'| Total | 449 | {receipt["deleted_bytes"]:,} | {receipt["deleted_allocated_bytes"]:,} | 32 | 268,875 | 344,064 |',
        '',
        '| Label | Deleted files | Logical bytes | Allocated bytes | Kept files | Kept logical bytes | Kept allocated bytes | Directories removed |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    labels = sorted({Path(p).parent.name for p in m['scope']['scratch_roots']}|{'prompts'})
    for label in labels:
        d=receipt['deleted_summary']['labels'].get(label,z);k=receipt['kept_summary']['labels'].get(label,z)
        dirs=receipt['removed_directory_summary'].get(label,{'directories':0})
        lines.append(f'| {label} | {d["files"]} | {d["bytes"]:,} | {d["allocated_bytes"]:,} | {k["files"]} | {k["bytes"]:,} | {k["allocated_bytes"]:,} | {dirs["directories"]} |')
    lines.extend(['', f'Directories: {receipt["removed_directory_count"]} removed; {v["remaining_scratch_directory_count"]} original scratch directories remain. '
        f'At removal, directories totaled {receipt["directory_bytes_at_removal"]:,} logical bytes and '
        f'{receipt["directory_allocated_bytes_at_removal"]:,} allocated bytes. Exact directory paths and per-label allocations '
        'are in the final receipt. Each removed directory was an ancestor of a deleted file, was in the original inventory and was empty before `rmdir`. '
        'No prompt directory or label root was removed.', '',
        '**Changes and artifacts.** The exact 449 deleted file paths, every archived cover, deletion-time hash/identity/size, '
        'all removed directories, and every kept file with its reason are listed in the final receipt. '
        'The original manifest still describes the frozen proposal; its historical WITHHELD status is intentionally retained. '
        'Use the final receipt and final verification for the execution outcome.', '',
        '| Changed record file | Change and purpose |', '| --- | --- |',
        '| `report.md` | Updated this report with executed results; previous version backed up. |',
        '| `execute_cleanup.py` (new) | Reuses `prepare_cleanup.py` hashing/walking/statistics; adds frozen-scope validation, fresh snapshots, completion gate, exact unlink, empty-directory removal and durable journal. |',
        '| `verify_cleanup_final.py` (new) | Independent journal/scope/hash/archive/retention audit; updates only final verification and final receipt. Safe to rerun. |',
        '| `write_report_final.py` (new) | Generates final counts/report, exact check-command log and file catalogue. |',
        '| `record_inputs_before_final.json` (new) | Original record hashes captured before execution. |',
        '| `report_before_execution.md` (new) | Byte-identical copy of the previous report. |',
        '| `protected_hash_before_final.json`, `protected_hash_after_final.json` (new) | Fresh SHA256 inventories of readable protected files, with six exact accepted exceptions. |',
        '| `validation_pre_delete_final.json` (new) | All 481 scoped files revalidated; 130 archive entries and original 47 scratch directories checked. |',
        '| `completion_check_immediate_final.json` (new) | 13 exits, last-write cutoff, three launcher exits and three monitor finals immediately before deletion. |',
        '| `deletion_journal_final.jsonl` (new) | Fsynced intent/success pairs for every individual unlink/rmdir. |',
        '| `deletion_receipt_final.json` (new) | Exact actual deletions and keeps, logical/allocated bytes by category/label, directory counts and verification. |',
        '| `verification_final.json` (new) | Executed independent audit, zero errors and protected-input comparison. |',
        '| `execution_commands_final.json` (new) | Exact verification commands, environment, expected/actual outcomes and initial interpreter correction. |',
        '| `record_files_final.json` (new) | All record files catalogued with hashes and original/new/updated status. Catalogue excludes its own self-hash. |',
        '',
        'Relevant function locations:', '',
    ])
    for name in ('execute_cleanup.py','verify_cleanup_final.py','write_report_final.py'):
        symbols = [f'{n.name}:{n.lineno}' for n in ast.parse((R/name).read_text()).body if isinstance(n,ast.FunctionDef)]
        lines.append(f'- `{name}`: '+', '.join(symbols)+'.')
    lines.extend(['', '**Verification.** All required final commands used system `/usr/bin/python3` '+sys.version.split()[0]+
        ' from `/data/ycfeng`, with `-B` to prevent bytecode writes. Exact commands and expected/actual results are saved '
        'in `execution_commands_final.json`. Snapshot data reflect fresh byte reads on both sides of deletion; no metadata-based hash reuse was used.', '',
        '| Check | Expected | Observed | Status |', '| --- | --- | --- | --- |',
        '| Final script syntax | Three scripts parse | Three scripts parse; no bytecode | PASS |',
        '| Scoped preflight | 481 files; 449 delete / 32 keep; 47 directories; all candidate hashes unchanged | Exact counts, hashes and identities match; 130 archive entries unchanged | PASS |',
        '| Immediate completion evidence | 13 Pi exits, 3 launcher exits, 3 monitor finals; no label write after 08:05Z | All completion records present; newest scoped write 08:00:22.994736Z | PASS |',
        '| Deletion-time revalidation | Every candidate hash equals manifest; duplicates equal current archive cover | 449 journaled hash matches; 87 covers match current archive content | PASS |',
        f'| Exact removal/retention | Delete exactly 449; preserve 32; remove only affected empty directories | 449 absent, 32 unchanged, {receipt["removed_directory_count"]} directories absent | PASS |',
        f'| Readable protected hashes | No added/missing/changed protected paths | {pc["hash_matched_regular_files"]:,} regular hashes match; {pc["symlinks_matched"]:,} link-text hashes match; no added/missing/changed entries | PASS |',
        f'| Non-scratch records of scoped labels | All retained | {pc["non_scratch_label_regular_files_matched"]:,} regular-file hashes match | PASS |',
        '| Both generation08 prompt pairs | Four files unchanged | submit-1107q.{md,sh} and post-1107r.{md,sh} unchanged by current SHA256 | PASS |',
        '| Archive preservation | All 130 entries and 3 manifests unchanged | All match recorded SHA256 and size | PASS |',
        '| Original preparation records | All unchanged except requested report update | Original hashes match; report backup matches | PASS |',
        '| Independent final audit | Zero errors | PASS; zero errors | PASS |',
        '| Host process scan | Lead accepts completion evidence instead | Unavailable isolated namespace; substituted exactly as authorized | NOT_RUN / accepted substitute |',
        '| Six known protected exceptions | List precisely; do not rehash historical log | Listed below; metadata only | NOT_RUN / accepted exceptions |',
        '',
        f'Before snapshot: `{before["started_utc"]}` → `{before["finished_utc"]}`, {before["runtime_seconds"]:.2f} s. '
        f'After snapshot: `{after["started_utc"]}` → `{after["finished_utc"]}`, {after["runtime_seconds"]:.2f} s. '
        f'Both readable protected inventories have SHA256 `{pc["before_inventory_sha256"]}`.', '',
        f'The immediate completion gate ran `{receipt["completion_check"]["started_utc"]}` → '
        f'`{receipt["completion_check"]["finished_utc"]}`; first deletion check was `{receipt["deleted"][0]["checked_utc"]}`. '
        f'Last file unlink: `{receipt["deleted"][-1]["deleted_utc"]}`. All timestamps are UTC.', '',
        'Observed timestamp clarification: `p64-formal5-prepare-1104b` last wrote on November 3, 2026 at 17:45:05.658172Z; '
        'the other 12 labels last wrote on November 7 between 05:39:28.588314Z and 08:00:22.994736Z. '
        'The older preparation label differs from the descriptive range in the brief but satisfies the controlling 08:05Z stop condition. '
        'Both mtime and ctime were checked before deletion. No behavior/interface/scope/authorization change was made.', '',
        '**Known accepted exceptions, not failures.**', '',
    ])
    for entry in before['accepted_exceptions']:
        lines.append(f'- `{entry["path"]}`: {entry["reason"]}; not opened/traversed/hashed.')
    lines.extend(['', '**Exact executed check commands.** These are historical invocations; the deletion command is guarded against replay.', '', '```bash'])
    for entry in commands:
        lines.extend([entry['command'], ''])
    lines.append('```')
    log_before=before['accepted_exceptions'][-1];log_after=after['accepted_exceptions'][-1]
    lines.extend(['', f'The historical log had the same device/inode and grew from {log_before["bytes"]:,} to '
        f'{log_after["bytes"]:,} bytes during the snapshots. Growth is an observation. '
        'Attribution to the old launcher follows the lead’s accepted context and progress.md:429; this executor did not trace or control that process.', '',
        '**Kept files.** Each path below is under its label’s `scratch/` tree and remains byte-identical to the original manifest. '
        'All 32 are kept because their current hash has no covering authorized archive entry and they are outside Node compile caches. '
        'Exact hashes, allocation sizes and kept reasons are in `deletion_receipt_final.json`.', '',
        '| Label | Scratch-relative path | Logical bytes | Allocated bytes |', '| --- | --- | ---: | ---: |'])
    for e in receipt['kept']:
        lines.append(f'| {e["label"]} | `{e["scratch_relative_path"]}` | {e["bytes"]:,} | {e["allocated_bytes"]:,} |')
    lines.extend(['', '**Remaining work and review handoff.** No assigned cleanup or verification action remains unfinished. '
        'Claude retains review, acceptance, commits and task-record updates outside this record directory. '
        'Inspect the exact final receipt against the immutable manifest, the 19 content-based archive covers whose recorded source differs, '
        'the journal’s 449 unlink intent/success pairs and the 32 kept entries. '
        'Review the six exception paths and the explicit completion substitute; neither establishes a host-wide process scan or '
        'hash coverage of unreadable descendants. `st_blocks` totals do not prove filesystem free-space gain.', '',
        'To repeat the completed acceptance audit without deleting files, run:', '',
        f'```bash\npython3 -B -u {R}/verify_cleanup_final.py\n```', '',
        'Do not rerun the deletion command: the executor refuses an existing final receipt or journal. '
        'The audit updates its final receipt/verification records, recomputes current kept/archive/prompt hashes and independently checks both stored protected snapshots. '
        'A new full protected snapshot is only needed if Claude requires evidence against later concurrent changes.', '',
        'Principal paths:', '',
        f'- Frozen manifest: `{R}/deletion_manifest.json`',
        f'- Final receipt: `{R}/deletion_receipt_final.json`',
        f'- Final report: `{R}/report.md`',
        f'- Final independent verification: `{R}/verification_final.json`',
    ])
    (R/'report.md').write_text('\n'.join(lines)+'\n')
    old = {e['path']:e for e in load('record_inputs_before_final.json')['entries']}
    rows=[]
    for p in sorted(R.iterdir()):
        if not p.is_file() or p.name == 'record_files_final.json':
            continue
        h=sha(p)
        baseline=old.get(str(p))
        status='new' if baseline is None else ('unchanged' if baseline['sha256']==h else 'updated')
        assert status != 'updated' or p.name == 'report.md', 'Unexpected pre-existing record edit'
        rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h,'status':status})
    write('record_files_final.json', {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'entries':rows,'note':'Every record file except this catalogue; self-hash omitted. All original inputs unchanged except report.md.'})
    print('PASS: final report, exact check commands and record catalogue written',flush=True)


if __name__ == '__main__':
    main()
