#!/usr/bin/env python3
"""Hash-bound R4 successor; replace only the V4 source-binding identity.

The approved DP-startup repair (a06a72a) carries the unchanged C26 model leaf
plus four committed startup sources in one runtime overlay. Generation 01 of
the same P2 case keeps the original exactly-2cf0a69 plus-C26 binding, so the
frozen R4 predicate stays in force for that build. This module changes only
that predicate: every other R4/R3/R2 check, threshold and CLI entry point is
executed from the hash-pinned dependency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

R4_PATH = Path(__file__).with_name('check_groundtruth_run_successor_r4.py')
R4_SHA256 = '67f3192302b6d12c0b8296581ee8b2788a772bce80b158ad45bc349eacb32a7b'

# Generation 01: the frozen R4 binding stays authoritative, unchanged.
BASE_SOURCE_SHA = '2cf0a6915ce544dc493a0990f2ea38d81601128a'
# Generation 02: the approved DP-startup repair under D-P2-64-STARTUP-RECOVERY.
STARTUP_SOURCE_SHA = 'a06a72a4326cba02234df6cfbdc21b78d3a15d26'
STARTUP_SOURCES = {
    'vllm/config/parallel.py': (
        'native_config_parallel.py',
        'f29f984ac39f4ae11622c2c64f92a657951f3a28e624a2a60d1040b49e025311',
        'f489eb26a3784ce5691faebad99a33b6e2e7c5988605bd7427415efc685f7440'),
    'vllm/distributed/parallel_state.py': (
        'native_distributed_parallel_state.py',
        'c6d65b96a260ea0d779da1073d1799392813eee74c9ed44753dfa56f4e7a9e4c',
        'a9b20efd63ece3bd4e4436f774a7be7156bfc410b310944c4ef4d7e19a091572'),
    'vllm/distributed/utils.py': (
        'native_distributed_utils.py',
        '792146b357139e705064a68d3b66858de8a0baaca4dc7db1232fa217d45db231',
        'a06292492a5cc24a17be299a5890d617fc8d1fe82e9a95b1cfc6be47d0d250ab'),
    'vllm/v1/engine/utils.py': (
        'native_engine_utils.py',
        '6ce83b0552b6207505c9bd7ac1fd67d2771628eac01572738765923aa91c1c0f',
        '533d60761f2505e93b318c349baf733a1ec7e386f1c2ad3f768e41bdab017599'),
}
# The approved C26 exception is the same model leaf in both builds.
C26_MODEL = 'vllm/models/deepseek_v4/nvidia/model.py'
C26_MODEL_BASE_SHA256 = 'c01be6d313f494eb38982c8f56c314267bbf67310e7f13ca53b596b31aef4e44'
C26_MODEL_APPROVED_SHA256 = 'd3556a47056bd52680b295f1585906149310397ec53ab90aa0008ff7d57e86ee'
OVERLAY_FIELDS = {'source', 'sha256', 'base_sha256'}


def native_v4_source_binding(manifest: dict, engine: dict, config: dict) -> None:
    """Bind the approved startup repair, and only that exact native identity.

    Generation 02 must stage exactly the unchanged C26 model leaf and the four
    startup sources of a06a72a, each at its approved base/committed bytes, and
    must seal, retain, commit and prelaunch-bind every one of them. Any other
    native commit, overlay member or hash is rejected; the generation 01 build
    keeps the frozen R4 predicate.
    """
    declaration = config['runtime_source_overlay']
    reference = engine.get('native_reference') or {}
    identities = (manifest.get('groundtruth_commit'), reference.get('source_sha'),
                  declaration.get('base_source_sha'))
    if identities == (BASE_SOURCE_SHA,) * 3:
        return frozen_native_v4_source_binding(manifest, engine, config)
    if any(identity != STARTUP_SOURCE_SHA for identity in identities):
        raise ValueError(f'Unknown V4 native source identity: {identities}')
    if (declaration.get('approved_c26_model_sha256') != C26_MODEL_APPROVED_SHA256
            or not declaration.get('effective_runtime_source')):
        raise ValueError('V4 runtime must declare the approved startup-repair source identity')
    stage = Path(config['script']).parent
    seal_path, lock_path = stage / 'manifest.json', stage / 'source_lock.json'
    seal, lock = json.loads(seal_path.read_text()), json.loads(lock_path.read_text())
    if lock['native_commit'] != STARTUP_SOURCE_SHA or lock['harness_commit'] != manifest['harness_commit']:
        raise ValueError('V4 source_lock does not bind the actual native and harness commits')
    overlay_path = Path(config['overlay_manifest'])
    if overlay_path != stage / 'overlay.json':
        raise ValueError('V4 overlay manifest is outside the sealed execution package')
    overlay = json.loads(overlay_path.read_text())
    if lock['runtime_overlay'] != overlay:
        raise ValueError('V4 source_lock runtime overlay differs from the sealed manifest')
    checkout = Path(manifest['groundtruth_checkout_path'])
    approved_paths = {C26_MODEL, *STARTUP_SOURCES}
    rows = overlay['files']
    if set(rows) != {path.removeprefix('vllm/') for path in approved_paths}:
        raise ValueError('V4 runtime overlay is not exactly the approved C26 model and four startup sources')
    # The C26 exception keeps its original predicate: unchanged checkout bytes,
    # staged exception bytes, and the declared approved hash.
    leaf = C26_MODEL.removeprefix('vllm/')
    row = rows[leaf]
    source = Path(row['source'])
    if (set(row) != OVERLAY_FIELDS or source != stage / 'overlay_model.py'
            or row['base_sha256'] != C26_MODEL_BASE_SHA256
            or row['sha256'] != C26_MODEL_APPROVED_SHA256
            or sha(source) != C26_MODEL_APPROVED_SHA256
            or sha(checkout / C26_MODEL) != C26_MODEL_BASE_SHA256
            or config['expected_source_hashes'].get(C26_MODEL) != C26_MODEL_APPROVED_SHA256):
        raise ValueError('V4 approved C26 model/base/observed-source mapping mismatch')
    for path, (staged_name, base, approved) in STARTUP_SOURCES.items():
        row = rows[path.removeprefix('vllm/')]
        source = Path(row['source'])
        staged = stage / staged_name
        sealed = sha(staged)
        if (set(row) != OVERLAY_FIELDS or source != staged
                or row['base_sha256'] != base or row['sha256'] != approved
                or sealed != approved or sha(checkout / path) != approved
                or config['expected_source_hashes'].get(path) != approved):
            raise ValueError(f'V4 startup source overlay differs from the approved commit: {path}')
        if (seal['files'].get(staged_name) != sealed
                or config['retained_files'].get(staged_name) != sealed):
            raise ValueError(f'V4 startup source is not sealed and prelaunch-bound: {staged_name}')
        if manifest['inputs_sha256'].get(str(staged)) != sealed:
            raise ValueError(f'V4 startup source is not bound to the launch inputs: {staged_name}')
        committed = lock['committed_files'].get(staged_name) or {}
        if (set(committed) != {'commit', 'path', 'sha256'} or committed.get('commit') != STARTUP_SOURCE_SHA
                or committed.get('path') != path or committed.get('sha256') != approved):
            raise ValueError(f'V4 startup source commit binding mismatch: {staged_name}')
    if {entry['path'] for entry in lock['committed_files'].values()
            if str(entry.get('path', '')).startswith('vllm/')} != set(STARTUP_SOURCES):
        raise ValueError('V4 source_lock binds unapproved native sources')
    for name in ('overlay.json', 'overlay_model.py'):
        staged = stage / name
        actual = sha(staged)
        if (seal['files'].get(name) != actual or config['retained_files'].get(name) != actual
                or manifest['inputs_sha256'].get(str(staged)) != actual
                or (name == 'overlay_model.py'
                    and (lock['retained_files'].get(name) or {}).get('sha256') != actual)):
            raise ValueError(f'V4 overlay is not sealed, source-locked and prelaunch-bound: {name}')
    if (seal['files'].get('source_lock.json') != sha(lock_path)
            or seal['files'].get('config.json') != manifest['inputs_sha256'].get(manifest['execution_config'])
            or json.loads((stage / 'config.json').read_text()) != config
            or seal_path.stat().st_mtime_ns > (Path(config['case_dir']) / 'runs' /
                'groundtruth_clean' / config['run_tag'] / 'run_manifest.json').stat().st_mtime_ns):
        raise ValueError('V4 sealed package/source lock differs from its actual prelaunch config')
    for name, expected in config['expected_source_hashes'].items():
        if name != C26_MODEL and sha(checkout / name) != expected:
            raise ValueError(f'Unapproved V4 native source change: {name}')


def frozen_r4_source(path=R4_PATH) -> str:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != R4_SHA256:
        raise ValueError('Frozen R4 dependency hash mismatch')
    return content.decode()


def load_frozen_r4(path=R4_PATH) -> dict:
    """Execute the pinned R4 module; its binding stays available under its own name."""
    namespace = {'__name__': 'frozen_r4_dispatch', '__file__': str(path)}
    exec(compile(frozen_r4_source(path), str(path) + ':frozen_r4', 'exec'), namespace)
    return namespace


def install_frozen_r4():
    """Rebind the inherited exec namespace, where the frozen call chain resolves names."""
    binding = native_v4_source_binding
    chain = load_frozen_r4()
    frozen = chain.pop('native_v4_source_binding')
    chain['frozen_native_v4_source_binding'] = frozen
    chain['native_v4_source_binding'] = binding
    globals().update({name: value for name, value in chain.items() if not name.startswith('__')})
    globals()['native_v4_source_binding'] = binding


install_frozen_r4()


if __name__ == '__main__':
    sys.exit(main())
