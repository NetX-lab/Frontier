"""Compile the two new sources without importing or executing their bodies."""

import hashlib
import json
import sys
import time
from pathlib import Path

owner = Path(__file__).resolve().parent
paths = [Path.cwd() / 'tests/analysis/v4_trace_audit_metrics_timing.py',
         owner / 'tiny_control.py']
started = time.monotonic()
rows = []
for path in paths:
    source = path.read_bytes()
    compile(source, str(path), 'exec')
    rows.append(dict(path=str(path), sha256=hashlib.sha256(source).hexdigest(),
                     gross_lines=len(source.splitlines()), expected='Compiles without execution',
                     actual='Compiled without execution', status='PASS'))
result = dict(command=f'timeout 60s /usr/bin/python3 -B {Path(__file__).relative_to(Path.cwd())}',
              cwd=str(Path.cwd()), python=sys.version, executable=sys.executable,
              conda='not used; /usr/bin/python3', elapsed_seconds=time.monotonic() - started,
              status='PASS', compiled_sources=rows, equivalence_invocations=0)
with (owner / 'static_compile_result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, sort_keys=True))
