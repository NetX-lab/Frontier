"""Verify the integrated native-key source against the frozen accepted evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCOPE = HERE / 'evidence/frozen_scope'
SOURCE = '376983d2ecc441d1e0a8cb8b943e65a2dcb217b5'
MERGED = 'c668a938'
SUB = 'frontier/cc_backend/backends/collective-sim'
KEY_PATHS = (
    'frontier/cc_backend/backends/collective_sim_cc_backend.py',
    'tests/unit/test_collective_sim_cc_backend.py',
)


def git(*args, repo=REPO):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), *args], timeout=60)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_files(directory, manifest):
    for name, expected in manifest.items():
        raw = (directory / name).read_bytes()
        assert len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], name


def main():
    assert sys.dont_write_bytecode
    manifest_raw = (SCOPE / 'artifact_sha256.json').read_bytes()
    assert sha(manifest_raw) == 'aec423731411d86c2272303137159f7b0f7f0010b37a93da02d3da1dbb709e28'
    verify_files(SCOPE, json.loads(manifest_raw)['files'])
    receipt = json.loads((SCOPE / 'frozen_scope_receipt.json').read_bytes())
    assert receipt['source_commit'] == SOURCE and not receipt['violations']
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert not git('diff', '--name-only', SOURCE, 'HEAD', '--', '.', ':(exclude)task_memory')
    assert sorted(git('diff-tree', '--no-commit-id', '--name-only', '-r', MERGED).decode().splitlines()) == sorted(KEY_PATHS)
    pin = receipt['non_source']['submodule_pin']
    assert git('rev-parse', 'HEAD', repo=REPO / SUB).decode().strip() == pin
    native = {row['path']: row for row in receipt['non_source']['native_rows']}
    bindings = {}
    for row in receipt['bindings']['rows']:
        label = row['binding']
        if label.startswith('native:'):
            item = native[label.removeprefix('native:')]
            for side in ('base', 'derived'):
                assert sha(Path(item[side + '_path']).read_bytes()) == item[side + '_sha256']
            bindings[label] = item['derived_sha256']
            continue
        if label.startswith('submodule:'):
            path = label.removeprefix('submodule:').removeprefix(SUB + '/')
            repo, revision, local = REPO / SUB, pin, REPO / label.removeprefix('submodule:')
        else:
            repo, revision, path, local = REPO, 'HEAD', label, REPO / label
        entry = git('ls-tree', revision, '--', path, repo=repo).decode().strip().split('\t')[0].split()
        assert dict(zip(('mode', 'type', 'oid'), entry)) == row['candidate'], label
        bindings[label] = entry[2]
        if entry[1] == 'blob':
            raw = local.read_bytes()
            assert sha(raw) == row['sha256'] and len(raw) == row['bytes'], label
    assert len(bindings) == 124
    digest = sha('\n'.join(f'{path} {bindings[path]}' for path in sorted(bindings) if path not in KEY_PATHS).encode())
    assert digest == receipt['bindings']['candidate_non_key_digest']
    cumulative = {}
    for line in git('diff', '--numstat', receipt['original_billing_base'], 'HEAD', '--', *KEY_PATHS, 'frontier/cc_backend/collective_plan.py').decode().splitlines():
        added, deleted, path = line.split('\t')
        cumulative[path] = int(added) + int(deleted)
    assert cumulative == {path: row['gross'] for path, row in receipt['fresh_cumulative'].items()}
    frozen = HERE / 'evidence/frozen_regressions'
    evidence = json.loads((frozen / 'artifact_sha256.json').read_bytes())
    verify_files(frozen, {**evidence['files'], **evidence['retained_scripts']})
    result = {
        'status': 'PASS_PRIMARY_SOURCE_AND_FROZEN_BINDINGS',
        'integrated_head': git('rev-parse', 'HEAD').decode().strip(),
        'integrated_tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
        'source_commit': SOURCE,
        'source_merge': git('rev-parse', MERGED).decode().strip(),
        'scope_artifacts_verified': 9,
        'frozen_artifacts_verified': len(evidence['files']) + len(evidence['retained_scripts']),
        'bindings_verified': len(bindings),
        'unchanged_non_key_digest': digest,
        'fresh_cumulative': cumulative,
        'totals': receipt['totals'],
        'preserved_untracked': git('ls-files', '--others', '--exclude-standard').decode().splitlines(),
        'script_sha256': sha(Path(__file__).read_bytes()),
        'python': sys.version,
        'executable': sys.executable,
        'limits': 'No suites or probes rerun. Source equivalence carries the frozen results, not native admission or full-CLI acceptance. Unrelated untracked files are preserved.',
    }
    output = HERE / 'evidence/primary_integration.json'
    assert not output.exists(), output
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
