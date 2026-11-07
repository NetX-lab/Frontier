from pathlib import Path
import hashlib
import json
import subprocess
import sys

r = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
t = r / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
h = t / 'h200_mainline'
a = h / 'p2_64_formal_retry07/package_01'
b = h / 'p2_64_formal_retry08/package_01'
OLD_HARNESS = 'f33306d40443cfc0c44842d6ee044f6703c736e7'
NEW_HARNESS = '57caf2f79f79c8bc19e199ba64f75ca49e8baaf6'
NATIVE = ('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/p2-dp-startup-coordination',
          'a06a72a4326cba02234df6cfbdc21b78d3a15d26')
WORKER = 'tests/comparison/calibration/native_pd_worker.py'
HEARTBEAT = dict(heartbeat_interval_s=10, peer_heartbeat_timeout_s=180)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


x = json.loads((a / 'manifest.json').read_text())
y = json.loads((b / 'manifest.json').read_text())
assert set(x['files']) == set(y['files']) and len(y['files']) == 47
for base, seal in [(a, x), (b, y)]:
    for name, digest in seal['files'].items():
        assert sha(base / name) == digest, (base, name)
changed = [n for n in x['files'] if x['files'][n] != y['files'][n]]
assert set(changed) == {'config.json', 'case/inputs/native_config.json', 'overlay.json', 'source_lock.json',
                        'case/manifest.yaml', 'evidence_readiness.json', 'native_pd_worker.py'}, changed
ca = json.loads((a / 'config.json').read_text())
cb = json.loads((b / 'config.json').read_text())
harness = Path(cb['harness_worktree'])


def committed(commit):
    return subprocess.run(['git', '-C', str(harness), 'show', f'{commit}:{WORKER}'],
                          check=True, capture_output=True).stdout


# The only code change: the staged worker is the committed worker before and after the heartbeat commit.
assert (a / 'native_pd_worker.py').read_bytes() == committed(OLD_HARNESS)
assert (b / 'native_pd_worker.py').read_bytes() == committed(NEW_HARNESS)
diff = subprocess.run(['git', '-C', str(harness), 'diff', '--name-only', OLD_HARNESS, NEW_HARNESS],
                      check=True, capture_output=True, text=True).stdout.split()
assert diff == [WORKER, 'tests/unit/test_native_pd_worker.py'], diff
replacements = {
    str(a): str(b),
    ca['run_tag']: cb['run_tag'],
    'v4-native-v028-' + ca['run_tag'].removeprefix('dsv4-'): 'v4-native-v028-' + cb['run_tag'].removeprefix('dsv4-'),
    ca['output_dir']: cb['output_dir'],
    ca['promotion_receipt']['build_utc']: cb['promotion_receipt']['build_utc'],
    x['files']['overlay.json']: y['files']['overlay.json'],
    x['files']['native_pd_worker.py']: y['files']['native_pd_worker.py'],
    OLD_HARNESS: NEW_HARNESS,
    json.loads((a / 'evidence_readiness.json').read_text())['purpose']: json.loads((b / 'evidence_readiness.json').read_text())['purpose'],
}
for name in changed:
    if name == 'native_pd_worker.py':
        continue
    text = (a / name).read_text()
    for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = text.replace(old, new)
    if name in ('config.json', 'case/inputs/native_config.json'):
        # The heartbeat fields are the one added configuration.
        assert json.loads(text) | HEARTBEAT == json.loads((b / name).read_text()), name
    else:
        assert text == (b / name).read_text(), name
assert {key: cb[key] for key in HEARTBEAT} == HEARTBEAT
for name in ['launcher_started.json', 'launcher_exit.json', 'case/runs']:
    assert not (b / name).exists()
assert y['submitted'] is False
assert not (b.parent / 'approval_receipt.json').exists()
assert (cb['gpu'], cb['replica'], cb['gpu_type'], cb['charged_group'], cb['preemptible'], cb['timeout_seconds']) == (8, 8, 'H200', 'step_main', True, 3600)
assert cb['promotion_receipt']['sha256'] == 'd505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596'
commits = {}
for repo, expected in [(harness, NEW_HARNESS), (Path(NATIVE[0]), NATIVE[1])]:
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    assert commit == expected and not dirty, (repo, commit, dirty)
    commits[str(repo)] = commit
argv = ['/data/ycfeng/envs/openmopd-py312/bin/python', '-B', str(b / 'launch.py'), '--check-only']
check = subprocess.run(argv, capture_output=True, text=True, check=True)
assert check.stdout.strip() == 'INPUT_HASHES_PASS_NO_SUBMISSION'
out = h / 'p2_64_formal_retry08/lead_package_acceptance.json'
out.write_text(json.dumps(dict(
    status='PASS_PACKAGE_IDENTITY_HEARTBEAT_AND_CHECK_ONLY', python=sys.version.split()[0],
    command='python3 h200_mainline/p2_64_formal_retry08/lead_package_audit.py',
    script_sha256=sha(Path(__file__)), package=str(b), manifest_sha256=sha(b / 'manifest.json'),
    sealed_files=47, byte_identical_files=40, changed_files=changed,
    allowed_changes=('Exact replacement of package root, run tag, cloud output, promotion build time, purpose, '
                     'harness commit and the derived overlay and worker digests; the staged worker equals the '
                     'heartbeat commit; config adds only the two heartbeat fields; all other bytes equal generation07.'),
    heartbeat=HEARTBEAT, harness_diff=diff, commits=commits,
    check_only=dict(argv=argv, exit_code=check.returncode, stdout=check.stdout),
    old_manifest_unchanged=sha(a / 'manifest.json'),
    decision='D-P2-64-SIXTH', jobs_used_under_decision=0, jobs_authorized_under_decision=3,
    submission_status='PENDING_NIGHT_SUBMISSION',
), indent=2) + '\n')
print(json.dumps(dict(status='PASS', manifest_sha256=sha(b / 'manifest.json'), sealed_files=47,
                      identical_files=40, check_only=check.stdout.strip()), indent=2))
