"""Prepare an exclusive four-file derived context; never build or publish an image."""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from tests.integration.step5_pd.build_stage import installed_source_hashes, sha256
from tests.comparison.dp_placement_pp.vllm_replay import STEP5_IMAGE

PATHS = ('vllm/forward_context.py', 'vllm/models/step5/nvidia/model.py',
         'vllm/model_executor/layers/fused_moe/router/router_factory.py',
         'vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py')
INSTALLED = '/usr/local/lib/python3.12/dist-packages/vllm'


def prepare(base: Path, derived: Path, listing: Path, output: Path):
    if output.exists():
        raise ValueError('Build context must be new and exclusive')
    pins = installed_source_hashes(listing, INSTALLED)
    if not set(PATHS) <= pins.keys():
        raise ValueError('Installed base pins must cover all four native changes')
    if any(root.is_symlink() or any(path.is_symlink() for path in root.rglob('*'))
           for root in (base, derived)):
        raise ValueError('Native source symlink traversal is not supported')
    before, after, patch = {}, {}, []
    originals = {str(path.relative_to(base)): path for path in base.rglob('*') if path.is_file()}
    candidates = {str(path.relative_to(derived)): path for path in derived.rglob('*') if path.is_file()}
    if originals.keys() != candidates.keys():
        raise ValueError('Derived source file inventory differs from the locked base')
    changed = []
    for name, original in originals.items():
        candidate = candidates[name]
        if (original.is_symlink() or candidate.is_symlink()
                or not original.resolve().is_relative_to(base.resolve())
                or not candidate.resolve().is_relative_to(derived.resolve())):
            raise ValueError('Native source symlink traversal is not supported')
        if name in pins and sha256(original) != pins[name]:
            raise ValueError('Base source disagrees with installed pin: '+name)
        if sha256(original) != sha256(candidate):
            changed.append(name)
    if set(changed) != set(PATHS):
        raise ValueError('Derived native changes must be exactly the four-file allowlist')
    for name in PATHS:
        before[name], after[name] = sha256(originals[name]), sha256(candidates[name])
        patch.extend(difflib.unified_diff(originals[name].read_text().splitlines(True),
                     candidates[name].read_text().splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    output.mkdir(parents=True)
    for name in PATHS:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidates[name], target)
    shutil.copyfile(Path(__file__).with_name('Dockerfile.position_cyclic'), output/'Dockerfile')
    for name, values in [('base_source_sha256', before), ('derived_source_sha256', after)]:
        (output/(name+'.json')).write_text(json.dumps(values, indent=2)+'\n')
    (output/'native.patch').write_text(''.join(patch))
    manifest = dict(base_image=STEP5_IMAGE, installed_native_root=INSTALLED, paths=list(PATHS),
                    base_source_sha256=before, derived_source_sha256=after,
                    patch_sha256=sha256(output/'native.patch'))
    (output/'derivation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'derived', 'listing', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.derived, args.listing, args.output), indent=2))
