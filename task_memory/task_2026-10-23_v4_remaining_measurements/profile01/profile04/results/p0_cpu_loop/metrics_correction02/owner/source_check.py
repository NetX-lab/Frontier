"""Compile allocated sources and retain source/allocation bindings without execution."""

import hashlib
import json
import sys
import time
from pathlib import Path

OWNER = Path(__file__).resolve().parent
REPO = Path.cwd()
started = time.monotonic()
allocation_path = OWNER.parent / 'primary/allocation.json'
allocation = json.loads(allocation_path.read_text())
rows = []
for relative, cap in zip(allocation['source_paths'],
                         (allocation['source_gross_cap'], allocation['test_gross_cap'])):
    path = REPO / relative
    source = path.read_bytes()
    compile(source, str(path), 'exec')
    lines = len(source.splitlines())
    assert lines <= cap, (relative, lines, cap)
    rows.append(dict(path=str(path), sha256=hashlib.sha256(source).hexdigest(),
                     gross_lines=lines, cap=cap, expected='Compiles without execution; within cap',
                     actual='Compiled without execution; within cap', status='PASS'))
inputs = [allocation_path, Path(allocation['evidence']['path']),
          Path(allocation['frozen_adapter_path']),
          OWNER.parent.parent / 'metrics_tail/primary/tiny_receipt_binding.json']
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
assert hashes[allocation['evidence']['path']] == allocation['evidence']['sha256']
assert hashes[allocation['frozen_adapter_path']] == allocation['frozen_adapter_sha256']
result = dict(command=f'timeout 60s /usr/bin/python3 -B {Path(__file__).relative_to(REPO)}',
              cwd=str(REPO), executable=sys.executable, python=sys.version, conda='not used',
              elapsed_seconds=time.monotonic() - started, status='PASS', compiled_sources=rows,
              input_hashes=hashes, unit_invocations=0, complete_pipeline_invocations=0)
with (OWNER / 'source_check_result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, sort_keys=True))
