"""Copy finished run directories from the cloud volume to this host with a CPU-only RJob.

This runtime has no route to `shai-ycfeng`, so the copy runs on a worker that mounts both the
cloud volume and a new directory of this host over NFS.

    python cloud_copy.py submit <package config.json> <copy dir> <cloud run root>...   # this host
    python cloud_copy.py copy <copy dir>/copy_config.json                                # the worker

`submit` creates <copy dir> (it must not exist), places this script and `copy_config.json` in
it, and submits one replica with no GPU, using the image, cloud mount, GPU tag and charged group
of the given package. Keep the launcher alive until it prints FINAL_STATUS.

For each run root `copy` writes, under <copy dir>/<run tag>/:
- `artifacts/`: the run tree, file mtimes kept;
- `cloud_inventory.json`: {'root', 'files': [{'path', 'bytes', 'sha256'}]}, hashed from the
  bytes read on the cloud volume, in the form of the earlier SSH inventories;
- `download_receipt.json`: status HASH_IDENTICAL_DOWNLOAD when every local file rehashes to its
  inventory digest, and the inventory's own digest.
Everything it writes is owned by the local user who ran `submit`.
"""
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CLOUD_SCOPE = Path('/mnt/codesign-exp/ycfeng')
CREDENTIALS = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
CPU, MEM_GB, TIMEOUT_SECONDS = 16, 64, 3600
BLOCK = 8 << 20


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def in_scope(root):
    if not Path(root).is_absolute() or not Path(os.path.normpath(root)).is_relative_to(CLOUD_SCOPE):
        raise SystemExit(f'Cloud run root outside {CLOUD_SCOPE}: {root}')
    return Path(os.path.normpath(root))


def copy_run(root, target):
    start, files = now(), []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise SystemExit(f'Path leaves the run root: {path}')
        relative = path.relative_to(root)
        local = target / 'artifacts' / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open('rb') as source, local.open('xb') as copy:
            for block in iter(lambda: source.read(BLOCK), b''):
                digest.update(block)
                copy.write(block)
        stat = path.stat()
        os.utime(local, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        files.append(dict(path=str(relative), bytes=stat.st_size, sha256=digest.hexdigest()))
    inventory = (json.dumps(dict(root=str(root), files=files)) + '\n').encode()
    (target / 'cloud_inventory.json').write_bytes(inventory)
    mismatched = [row['path'] for row in files if file_sha256(target / 'artifacts' / row['path']) != row['sha256']]
    receipt = dict(status='HASH_MISMATCH' if mismatched or not files else 'HASH_IDENTICAL_DOWNLOAD',
                   start_utc=start, end_utc=now(), cloud_root=str(root), local_root=str(target / 'artifacts'),
                   files=len(files), bytes=sum(row['bytes'] for row in files),
                   inventory_sha256=hashlib.sha256(inventory).hexdigest(), mismatched=mismatched,
                   method='CPU-only RJob with the cloud mount and an NFS mount of this host; '
                          'local copies rehashed after writing',
                   script_sha256=file_sha256(Path(__file__)), rjob_task=os.environ.get('EXP_ID'))
    (target / 'download_receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


def copy(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    status = 0
    try:
        for root in map(in_scope, config['roots']):
            target = config_path.parent / root.name
            target.mkdir()
            receipt = copy_run(root, target)
            print(json.dumps({key: receipt[key] for key in ('status', 'cloud_root', 'files', 'bytes')}), flush=True)
            status |= receipt['status'] != 'HASH_IDENTICAL_DOWNLOAD'
    finally:
        for path in [config_path.parent, *config_path.parent.rglob('*')]:
            os.chown(path, config['uid'], config['gid'], follow_symlinks=False)
    raise SystemExit(status)


def submit(package_config, copy_dir, roots):
    package = json.loads(Path(package_config).read_text())
    roots, copy_dir = [str(in_scope(root)) for root in roots], Path(copy_dir).resolve()
    if len({Path(root).name for root in roots}) != len(roots):
        raise SystemExit('Run tags must be distinct')
    assert os.environ['STEPMIND_BACKEND'] == 'rjob'
    copy_dir.mkdir(parents=True)
    script = copy_dir / Path(__file__).name
    shutil.copy2(__file__, script)
    config = dict(roots=roots, uid=os.getuid(), gid=os.getgid(), package_config=str(Path(package_config).resolve()),
                  script_sha256=file_sha256(script))
    (copy_dir / 'copy_config.json').write_text(json.dumps(config, indent=1) + '\n')
    os.environ['BRAINPP_ACCESS_KEY'] = (CREDENTIALS / 'accesskey_id').read_text().strip()
    os.environ['BRAINPP_SECRET_KEY'] = (CREDENTIALS / 'accesskey_secret').read_text().strip()
    from steptron.exp.base_exp import ResourceConfig
    from steptron.utils.stepmind import spawn_tasks
    os.chdir(copy_dir)
    os.environ['EXP_ID'] = f'cloud-copy-{copy_dir.name}'
    resource = ResourceConfig(
        cpu=CPU, gpu=0, mem_gb=MEM_GB, replica=1, image=package['image'], positive_tags=[package['gpu_type']],
        extra_requirements=[], mounts=[package['cloud_mount']], custom_resources=[],
        envs={'PYTHONDONTWRITEBYTECODE': '1'}, command='{COMMAND}', task_specs={'default': {'is_critical': True}})
    command = shlex.join(['timeout', str(TIMEOUT_SECONDS), 'python3', '-B', str(script), 'copy',
                          str(copy_dir / 'copy_config.json')])
    workers = spawn_tasks(resource, command='bash -c ' + shlex.quote(command), charged_group=package['charged_group'],
                          use_image=True, code_mount_point=str(copy_dir))
    print('RJOB_NAME', workers.rjob.meta.name, flush=True)
    (copy_dir / 'launcher.json').write_text(json.dumps(dict(job=workers.rjob.meta.name, submitted_utc=now(),
                                                            launcher_pid=os.getpid()), indent=1) + '\n')
    print('FINAL_STATUS', workers.poll(), flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['copy'] and len(sys.argv) == 3:
        copy(sys.argv[2])
    elif sys.argv[1:2] == ['submit'] and len(sys.argv) >= 5:
        submit(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        raise SystemExit(__doc__)
