"""CPU post-run ground-truth reader, persisted inputs, all-request and warm reading; soft gate.

compare.py --case PACKAGE_CASE --download DOWNLOAD_TAG_DIR --output NEW_DIR
           [--controller-run LOCAL_RUN_WITH_RECEIPTS] [--reference-checker r4|r5]
compare.py --control --output NEW_DIR
The lead supplies the hash-identical cloud copy and terminal receipt. This tool
never submits a job, copies from the cloud, or changes the formal replay. The
reference reader defaults to the generation 01 R4 checker; R5 is the approved
startup-repair successor and keeps the same reading and soft-gate behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from coverage import R4, load_module, safe_analyze, save

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
REPO = TASK.parents[1]
REPLAY = Path('/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/runs/p2_64')
COMMAND = TASK / 'h3o_operator_costs/formal_p2_20261102/p2_64_command.json'
CONTROL_TAG = 'dsv4-h200-p2-64-canonical-20261025-01'
CONTROL_CASE = Path('/data/ycfeng/tmp/glm53_dsv4/h200/cases/p2_64r_case_20261025')
CONTROL_DOWNLOAD = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads') / CONTROL_TAG
CHECKERS = {
    'r4': R4,
    'r5': HERE.parents[2] / 'task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r5.py',
}
PERSISTED = TASK / 'h200_mainline/diagnostic/inputs'
CONTROL_FILES = [r['path'] for r in json.loads((PERSISTED / 'manifest.json').read_text())['p2_64r']['files']]
RECEIPTS = {'launcher_receipt.json', 'platform_terminal_receipt.json'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_reference_checker(choice='r4'):
    require(choice in CHECKERS, f'Unknown reference checker: {choice}')
    path = CHECKERS[choice]
    require(path.is_file(), f'Missing reference checker: {path}')
    return path


def checked_copy(source, target):
    before = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)  # R4 checks output mtimes relative to the manifest
    require(sha(target) == before, f'Copy changed bytes: {source}')
    return dict(path=str(target), source=str(source), sha256=before, bytes=target.stat().st_size)


def verify_download(download):
    inventory_path = download / 'cloud_inventory.json'
    receipt_path = download / 'download_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    require(receipt['status'] == 'HASH_IDENTICAL_DOWNLOAD' and not receipt['mismatched']
            and sha(inventory_path) == receipt['inventory_sha256'], 'Cloud-copy receipt/inventory not bound')
    rows = inventory['files']
    root = download / 'artifacts'
    recorded = {r['path'] for r in rows}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(rows and len(recorded) == len(rows) and recorded <= actual and actual - recorded <= RECEIPTS,
            'Unexpected/missing cloud-copy files (only local launcher and terminal receipts may be added)')
    for row in rows:
        path = root / row['path']
        require(not Path(row['path']).is_absolute() and path.resolve().is_relative_to(root.resolve())
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                f'Hash-identical copy check failed: {row["path"]}')
    return root, dict(status='PASS', inventory=dict(path=str(inventory_path), sha256=sha(inventory_path)),
                      receipt=dict(path=str(receipt_path), sha256=sha(receipt_path)), files=len(rows),
                      local_receipt_additions=sorted(actual - recorded))


def persist(run, r4, download, out):
    inputs = out / 'inputs'
    target = inputs / 'p2_64r'
    rows = []
    for relative in CONTROL_FILES:
        if relative == 'run_check_status.groundtruth_clean.json':
            source = r4
        elif relative in ('cloud_inventory.json', 'download_receipt.json'):
            source = download / relative
        else:
            source = run / relative
        entry = checked_copy(source, target / relative)
        entry['path'] = relative
        rows.append(entry)
    save(inputs / 'manifest.json', dict(p2_64r=dict(files=rows)))
    return inputs


def run_pipeline(case_source, download, out, controller=None, control=False, r4_python=None,
                 reference_checker='r4'):
    checker = select_reference_checker(reference_checker)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='UNVERIFIED', soft_gate=True, control=control, steps=[], errors=[],
                  reference_checker=dict(name=reference_checker, path=str(checker),
                                         sha256=sha(checker)))
    def stage(name, action):
        try:
            value = action()
            result['steps'].append(dict(name=name, status='PASS'))
            return value
        except (Exception, SystemExit) as error:
            result['steps'].append(dict(name=name, status='UNVERIFIED', error=f'{type(error).__name__}: {error}'))
            result['errors'].append(name)
            return None
    def command(argv):
        started = datetime.now(timezone.utc).isoformat()
        process = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=300)
        result.setdefault('commands', []).append(dict(argv=argv, cwd=str(REPO), controller_python=sys.version,
            interpreter=argv[0], started_utc=started, exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr))
        require(process.returncode == 0, f'Command failed ({process.returncode}): {process.stdout} {process.stderr}')
    source = stage('hash_identical_download', lambda: verify_download(download))
    if source is None:
        save(out / 'result.json', result)
        return result
    artifacts, download_evidence = source
    result['download'] = download_evidence
    def read_manifest():
        document = json.loads((artifacts / 'run_manifest.json').read_text())
        require(isinstance(document.get('run_tag'), str) and document['run_tag']
                and Path(document['run_tag']).name == document['run_tag'], 'Invalid run tag')
        return document
    manifest = stage('read_run_manifest', read_manifest)
    if manifest is None:
        save(out / 'result.json', result)
        return result
    tag = manifest['run_tag']
    case = out / 'r4_case'
    run = case / 'runs/groundtruth_clean' / tag
    def prepare():
        checked_copy(case_source / 'manifest.yaml', case / 'manifest.yaml')
        for path in (case_source / 'inputs').rglob('*'):
            if path.is_file():
                checked_copy(path, case / 'inputs' / path.relative_to(case_source / 'inputs'))
        for path in artifacts.rglob('*'):
            if path.is_file():
                checked_copy(path, run / path.relative_to(artifacts))
        for name in RECEIPTS:
            if not (run / name).is_file():
                require(controller is not None, f'Missing local controller receipt: {name}')
                checked_copy(controller / name, run / name)
        # Independent policy check because R4's platform predicate does not
        # compare the platform's preemptible label.
        if not control:
            config = json.loads((case / manifest['execution_config']).read_text())
            worker = load_module('postrun_receipt_validator', Path(config['harness_worktree']) /
                                 'tests/comparison/calibration/native_pd_worker.py')
            binding = worker.validate_promotion_receipt(config)
            platform = json.loads((run / 'platform_terminal_receipt.json').read_text())
            require(binding and manifest.get('promotion_receipt') == binding
                    and config.get('pair_warmup') is True and config.get('preemptible') is True
                    and manifest['resources'].get('preemptible') is True
                    and platform['preemptible_label'] == 'yes', 'Formal receipt/resource binding is unverified')
            issued = datetime.fromisoformat(json.loads(Path(binding['path']).read_text())['issued_utc'])
            require(issued < datetime.fromisoformat(manifest['written_before_launch_utc']),
                    'Promotion receipt does not precede the native run manifest')
        return True
    prepared = stage('prepare_case_and_policy', prepare)
    coverage = stage('coverage', lambda: safe_analyze(artifacts))
    if coverage is not None:
        save(out / 'coverage.json', coverage)
        if coverage['openers'] is not None:
            save(out / 'openers.json', coverage['openers'])
        result['coverage'] = dict(status=coverage['status'], evidence_verified=coverage['evidence_verified'])
    r4_path = case / 'analysis/run_check_status.groundtruth_clean.json'
    if prepared:
        stage('r4', lambda: command([str(r4_python or sys.executable), str(checker), str(case), tag, 'clean']))
    # The command returns None on success; inspect the selected reader's artifact.
    r4_pass = r4_path.is_file() and json.loads(r4_path.read_text())['runs'][tag]['status'] == 'PASS'
    result['r4_status'] = 'PASS' if r4_pass else 'UNVERIFIED'
    inputs = stage('persist_native_inputs', lambda: persist(run, r4_path, download, out)) if r4_path.is_file() else None
    def compare():
        require(r4_pass and inputs is not None, 'R4 PASS and persisted inputs are required by the unchanged gate')
        a10 = load_module('a10_gate', TASK / 'h3o_operator_costs/a10/a10_gate.py')
        # Reuse native_layout and matrix; change only the case's local input and
        # actual R4 digest. No retained script or case table is edited.
        entry = dict(a10.CASES['p2_64'], tag=tag, case_id=manifest['case_id'], inputs=inputs,
                     input_name='p2_64r', r4=inputs / 'p2_64r/run_check_status.groundtruth_clean.json',
                     r4_sha256=sha(inputs / 'p2_64r/run_check_status.groundtruth_clean.json'))
        gate_dir = out / 'gate'
        gate_dir.mkdir()
        local_run, receipt = a10.h4a4.native_layout('p2_64', entry, gate_dir)
        cell = f"{entry['case_id']}/a10_operator_path"
        matrix = a10.matrix(entry, cell, gate_dir, receipt)
        save(out / 'case.json', {k: str(v) if isinstance(v, Path) else v for k, v in entry.items()})
        command_record = json.loads(COMMAND.read_text())
        require(command_record['exit_code'] == 0 and Path(command_record['run_root']) == REPLAY,
                'Formal replay command not finished/bound')
        # Read-only hashes of the unchanged replay are recorded before and after.
        before = {p.name: sha(p) for p in REPLAY.iterdir() if p.is_file()}
        gate = [sys.executable, str(REPO / 'tests/comparison/e2e_gate/run_gate.py')]
        common = ['--matrix', str(matrix), '--cell', cell]
        command([*gate, 'normalize', *common, '--artifacts-root', str(gate_dir), '--vllm-run-dir', str(local_run),
                 '--frontier-run', f'a10={REPLAY}', '--output-dir', str(gate_dir / 'rows')])
        command([*gate, 'gate', *common, '--rows', str(gate_dir / 'rows'), '--output', str(gate_dir / 'reading.json')])
        reading = json.loads((gate_dir / 'reading.json').read_text())
        result['all_request_verdict'] = reading['verdict']
        if coverage is not None and coverage['openers'] is not None:
            command([sys.executable, str(REPO / 'tests/analysis/e2e_warm_reading.py'), '--rows', str(gate_dir / 'rows'),
                     '--reading', str(gate_dir / 'reading.json'), '--openers', str(out / 'openers.json'),
                     '--output', str(out / 'warm_reading.json')])
        empty = {side: [] for side in reading['repetitions']}
        save(out / 'empty_openers.json', empty)
        command([sys.executable, str(REPO / 'tests/analysis/e2e_warm_reading.py'), '--rows', str(gate_dir / 'rows'),
                 '--reading', str(gate_dir / 'reading.json'), '--openers', str(out / 'empty_openers.json'),
                 '--output', str(out / 'empty_reading.json')])
        require(before == {p.name: sha(p) for p in REPLAY.iterdir() if p.is_file()}, 'Formal replay bytes changed')
        result['formal_replay'] = dict(command_path=str(COMMAND), command_sha256=sha(COMMAND), files=before)
        result['comparison_complete'] = (out / 'warm_reading.json').is_file()
        return True
    stage('comparison', compare)
    result['status'] = 'TOOLS_FINISHED' if not result['errors'] and result.get('comparison_complete') else 'UNVERIFIED'
    save(out / 'result.json', result)
    return result


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--control', action='store_true')
        parser.add_argument('--case', type=Path)
        parser.add_argument('--download', type=Path)
        parser.add_argument('--controller-run', type=Path)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--r4-python', type=Path, help='Existing CPU interpreter with torch for the reader; default this interpreter')
        parser.add_argument('--reference-checker', choices=tuple(CHECKERS), default='r4',
                            help='Immutable reference reader; r4 is generation 01, r5 the startup-repair successor')
        args = parser.parse_args()
        if args.control:
            require(args.case is None and args.download is None, 'Control fixes the retained case and download')
            case, download = CONTROL_CASE, CONTROL_DOWNLOAD
            controller = CONTROL_CASE / 'runs/groundtruth_clean' / CONTROL_TAG
        else:
            require(args.case is not None and args.download is not None, '--case and --download are required')
            case, download, controller = args.case.resolve(), args.download.resolve(), args.controller_run
        sys.path.insert(0, str(TASK / 'h3o_operator_costs/a10'))
        result = run_pipeline(case, download, args.output.resolve(), controller, args.control, args.r4_python,
                              args.reference_checker)
        print(json.dumps({k: result.get(k) for k in ('status', 'r4_status', 'all_request_verdict', 'coverage', 'errors')}))
    except (Exception, SystemExit) as error:
        print(f'P2_COMPARISON_UNVERIFIED {type(error).__name__}: {error}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
