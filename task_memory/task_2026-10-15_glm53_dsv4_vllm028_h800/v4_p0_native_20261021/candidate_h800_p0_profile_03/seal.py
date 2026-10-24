"""Seal launch inputs after offline verification scripts have been prepared."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = sorted([HERE / 'case/manifest.yaml'] + [path for path in HERE.glob('*') if path.suffix in ('.py', '.json')
                and path.name != 'manifest.json' and not path.name.startswith('offline_')]
               + [path for directory in ('model', 'tests', 'trace', 'case/inputs') for path in (HERE / directory).rglob('*') if path.is_file()])
payload = dict(files={str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
               submitted=False, native_qualification=False)
(HERE / 'manifest.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(sealed_files=len(files))))
