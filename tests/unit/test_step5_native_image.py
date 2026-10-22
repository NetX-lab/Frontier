"""Offline sparse-context allowlist checks; no container operations."""
from pathlib import Path
import hashlib
import pytest
from tests.integration.step5_native_image.build_image import PATHS, INSTALLED, prepare


def test_context_requires_exact_native_changes_and_new_destination(tmp_path):
    base, derived = tmp_path/'base', tmp_path/'derived'
    for name in PATHS:
        for root, text in ((base, '# original\n'), (derived, '# derived\n')):
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    listing = tmp_path/'pins'
    listing.write_text(''.join(hashlib.sha256((base/name).read_bytes()).hexdigest()+'  '+str(Path(INSTALLED).parent/name)+'\n' for name in PATHS))
    assert prepare(base, derived, listing, tmp_path/'context')['paths'] == list(PATHS)
    with pytest.raises(ValueError, match='exclusive'): prepare(base, derived, listing, tmp_path/'context')
    (derived/PATHS[0]).write_text('# original\n')
    with pytest.raises(ValueError, match='allowlist'): prepare(base, derived, listing, tmp_path/'wrong')
    (derived/PATHS[0]).write_text('# derived\n')
    (base/'foreign.py').write_text('# base\n'); (derived/'foreign.py').write_text('# foreign\n')
    with pytest.raises(ValueError, match='allowlist'): prepare(base, derived, listing, tmp_path/'foreign')
    (derived/'foreign.py').write_text('# base\n')
    listing.write_text(listing.read_text().replace(hashlib.sha256((base/PATHS[0]).read_bytes()).hexdigest(), '0'*64))
    with pytest.raises(ValueError, match='installed pin'): prepare(base, derived, listing, tmp_path/'hash')
