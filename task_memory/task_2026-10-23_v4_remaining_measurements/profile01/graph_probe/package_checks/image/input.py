"""Validate the actual staged entry in the pinned CPU image."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys

package = Path(sys.argv[1])
config = json.loads((package / 'config.json').read_text())
manifest = json.loads((package / 'manifest.json').read_text())
assert len(manifest['files']) == 8
assert {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()} == set(manifest['files']) | {'manifest.json'}
for name, expected in manifest['files'].items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
sys.argv = [str(package / 'launch.py')]
with contextlib.redirect_stdout(io.StringIO()) as captured:
    runpy.run_path(str(package / 'launch.py'), run_name='__main__')
dry = json.loads(captured.getvalue())
assert dry['submitted'] is False and dry['config'] == config
Path('/scratch/launcher_dry_run.json').write_text(json.dumps(dry, indent=2) + '\n')
Path('/scratch/diagnostics').mkdir()
os.environ['FRONTIER_GRAPH_DIAGNOSTICS_DIR'] = '/scratch/diagnostics'
sys.argv = [config['script'], 'inspect-api', '--output-dir', '/scratch/probe',
            '--image-digest', config['image'].split('@', 1)[1]]
runpy.run_path(config['script'], run_name='__main__')
report = json.loads(Path('/scratch/probe/probe_report.json').read_text())
assert report['script_sha256'] == config['source_sha256']
assert all(report[k] == 'NOT_RUN' for k in ('U1', 'U2', 'U3', 'U4'))
assert report['api']['cuda_initialized'] is False
paths = [Path(report['api']['cupti_library']), Path(config['cuda_compat_path']) / 'libcuda.so.1']
binaries = {}
for path in paths:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    binaries[str(path)] = dict(resolved=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)
assert binaries[report['api']['cupti_library']]['sha256'] == 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'
result = dict(status='PASS_FROZEN_PACKAGE_IMAGE_ENTRY', python=sys.version,
              manifest_sha256=hashlib.sha256((package / 'manifest.json').read_bytes()).hexdigest(),
              source_sha256=config['source_sha256'], input_count=8, cuda_initialized=False,
              binaries=binaries, limits='Actual package dry-run and inspect-api; CPU output paths substituted. No worker/GPU callback evidence.')
Path('/scratch/receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
