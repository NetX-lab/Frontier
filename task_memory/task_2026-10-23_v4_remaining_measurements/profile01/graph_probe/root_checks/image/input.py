"""CPU-only checks using the pinned image's actual interpreter and headers."""
import dis
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys

candidate = Path('/candidate/probe_diagnostic.py')
baseline = Path('/baseline.py')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(candidate) == 'a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0'
assert sha(baseline) == 'f7314cfbc70a1ba347339245754e385d24239eff9f24ceb485e5f3fba379a1e0'
assert sys.version_info[:3] == (3, 12, 3)
owner = json.loads(Path('/candidate/abi_source_binding.json').read_text())
distribution = importlib.metadata.distribution('nvidia-cuda-cupti')
headers = [Path(distribution.locate_file(p)) for p in distribution.files if p.name == 'cupti_callbacks.h']
assert len(headers) == 1
header = headers[0]
assert sha(header) == owner['header_sha256']
stdin = owner['compiler_stdin'].replace(owner['header'], str(header))
command = ['c++', '-std=c++11', '-I' + str(header.parent), '-x', 'c++', '-fsyntax-only', '-']
compiled = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=30)
Path('/scratch/compiler_stdin.cpp').write_text(stdin)
Path('/scratch/compiler_stderr.log').write_text(compiled.stderr)
assert compiled.returncode == 0, compiled.stderr
values = runpy.run_path(str(candidate))
assert values['callback_abi']() == owner['ctypes_abi']
old = runpy.run_path(str(baseline))
instructions = [dict(opcode=i.opname, offset=i.offset, line=i.positions.lineno,
                     end_line=i.positions.end_lineno, column=i.positions.col_offset)
                for i in dis.get_instructions(old['CuptiGraphCallbacks']._callback)
                if i.opname == 'CALL' and i.positions.lineno in (196, 197)]
assert any(i['line'] == i['end_line'] == 196 for i in instructions)
assert any(i['line'] == i['end_line'] == 197 for i in instructions)
import torch
assert not torch.cuda.is_initialized()
Path('/scratch/diagnostics').mkdir()
sys.argv = [str(candidate), 'inspect-api', '--output-dir', '/scratch/inspect',
            '--image-digest', 'sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635']
runpy.run_path(str(candidate), run_name='__main__')
assert not torch.cuda.is_initialized()
report = json.loads(Path('/scratch/inspect/probe_report.json').read_text())
assert report['command'] == 'inspect-api'
assert all(report[key] == 'NOT_RUN' for key in ('U1', 'U2', 'U3', 'U4'))
assert report['script_sha256'] == sha(candidate)
result = dict(status='PASS_CPU_IMAGE_ENTRY_ABI_AND_BYTECODE', python=sys.version,
              torch=torch.__version__, torch_cuda=torch.version.cuda,
              cupti_version=distribution.version, header=str(header), header_sha256=sha(header),
              compiler_command=command, compiler_returncode=compiled.returncode,
              ctypes_abi=values['callback_abi'](), old_callback_calls=instructions,
              cuda_initialized=torch.cuda.is_initialized(), original_fault='old source line197',
              interpretation='Actual Python3.12.3 distinguishes node and original-node call sites; this does not prove the old crash cause.',
              limits='No graph capture, subscription, native ID query or runtime payload decoding.')
Path('/scratch/receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
