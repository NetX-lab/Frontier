"""Compare the fourteen changed cache files with frozen bundle SHA256 values without loading models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
root=Path('/data/ycfeng/tmp/d29-parent-position-fixture03/cache')
observation=Path('/data/ycfeng/tmp/d29_main09_post_run_binding_observation_20261025.json')
changes=json.loads(observation.read_bytes())['fixture_metadata_changes']
bundles=[root/'bundle_594b60e71174496473a041cee333d8f20b5907ddc83b1d3805c07f62f0c7ac26.json',root/'bundle_0d1f560050cac3803a37e544fa99e5ec8fbf062bbec57745c259327f6464dd50.json']
expected={}
for path in bundles:
    for label,entry in json.loads(path.read_bytes())['models'].items():
        filename=f"{entry['cache_model_name']}_{entry['model_hash']}.pkl"
        expected.setdefault(filename,set()).add(entry['sha256'])
rows=[]
for change in changes:
    path=Path(change['path'])
    before=path.stat()
    assert [before.st_size,before.st_mtime_ns]==change['actual_size_mtime_ns']
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    after=path.stat()
    sha=digest.hexdigest()
    row={'path':str(path),'bytes':before.st_size,'actual_sha256':sha,'expected_bundle_sha256':sorted(expected[path.name]),
        'matches_frozen_bundle':sha in expected[path.name],
        'metadata_unchanged_during_read':[before.st_size,before.st_mtime_ns]==[after.st_size,after.st_mtime_ns]}
    rows.append(row)
print(json.dumps({'status':'PASS_CHANGED_CACHE_BYTES_MATCH_FROZEN_BUNDLES' if all(row['matches_frozen_bundle'] and row['metadata_unchanged_during_read'] for row in rows) else 'FAIL_CHANGED_CACHE_BYTES',
    'observed_utc':datetime.now(timezone.utc).isoformat(),'files':len(rows),'bytes':sum(row['bytes'] for row in rows),'rows':rows,
    'cache_models_loaded':0,'new_pricing_workload_shape_platform_calls':0,
    'original_source_after':'Still unavailable; the original metadata guard remains a recorded failure.',
    'limits':'This compares fourteen cache files after the job; it does not repeat the complete workflow or establish full-workload memory.'},indent=2))
