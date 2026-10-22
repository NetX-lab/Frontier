"""Read-only sealed-generation and native-pin preservation check."""

import hashlib
import json
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parent
BASE = Path('/data/ycfeng/tmp/step5_d24_native_20261022/capture03')
NATIVE = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = BASE/'stage_manifest.json'
    expected='cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d'
    assert digest(manifest)==expected
    rows=json.loads(manifest.read_text())
    assert rows['balanced_routing']=='BLOCKED_NATIVE_BALANCED_UNSUPPORTED'
    checked=[]
    for relative,row in rows['entries'].items():
        path=BASE/relative
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],relative
        checked.append(relative)
    pins=[]
    for line in (BASE/'installed_source_pins.txt').read_text().splitlines():
        sha,relative=line.split(maxsplit=1)
        relative=str(Path(relative.lstrip('*').strip()).relative_to('/usr/local/lib/python3.12/dist-packages'))
        path=NATIVE/relative
        assert digest(path)==sha,relative
        pins.append(relative)
    assert len(pins)==50
    assert not list(BASE.rglob('*submission*attempt*'))
    assert not list(BASE.rglob('*submitted*'))
    assert not list(BASE.rglob('*job*receipt*'))
    for name in ('contract_probe.py','semantic_probe.py','preservation_check.py'):
        compile((TASK/name).read_text(),str(TASK/name),'exec')
    result=dict(task_id='D29-NATIVE-BALANCED',status='PASS_READ_ONLY_PRESERVATION',
        interpreter=sys.executable,manifest=str(manifest),manifest_sha256=expected,
        sealed_manifest_entries_checked=len(checked),native_pins_checked=len(pins),
        capture03_not_balanced=True,no_attempt_or_submitted_marker=True,
        job_created=False,source_image_package_unchanged=True,compiled_probe_files=3)
    with (TASK/'preservation_receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
