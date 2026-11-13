"""Reuse the prior runner verbatim except fresh outputs and clean source binding."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key')
script = ROOT/'precommit_checks.py'
source = script.read_text()
expected = subprocess.check_output(['git','-C',str(REPO),'rev-parse','376983d2^{commit}'],text=True,timeout=10).strip()
assert len(expected) == 40
old = "destination = ROOT/('precommit-'+check)"
new = "destination = ROOT/('frozen-'+check)"
assert source.count(old) == 1
source = source.replace(old, new)
old = "    return dict(head=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'),"
new = (f"    assert git('rev-parse', 'HEAD') == {expected!r}\n"
       "    assert not git('status', '--porcelain=v1', '--untracked-files=all')\n"+old)
assert source.count(old) == 1
source = source.replace(old, new)
old = 'Precommit source has exactly the two owned, SHA-bound files; do not claim clean frozen provenance.'
new = f'Clean frozen source {expected};exact two-path delta and source hashes unchanged from the verified precommit run.'
assert source.count(old) == 1
source = source.replace(old, new)
assert subprocess.check_output(['git','-C',str(REPO),'status','--porcelain=v1','--untracked-files=all'],text=True,timeout=10).strip() == ''
bindings = dict(precommit_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                frozen_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_revision=expected)
namespace = {'__name__':'__main__','__file__':str(script),'frozen_binding':bindings}
exec(compile(source,str(script),'exec'),namespace)
