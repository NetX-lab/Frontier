#!/usr/bin/env python3
"""Record bounded RDMA device and provider observations without altering runtime."""

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def command(argv):
    if not shutil.which(argv[0]):
        return {'command': argv, 'unavailable': True}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        partial = {'stdout': error.stdout or '', 'stderr': error.stderr or ''}
        return {'command': argv, 'timeout_seconds': error.timeout,
                **{k: (v.decode(errors='replace') if isinstance(v, bytes) else v)[-24000:]
                   for k, v in partial.items()}}
    return {'command': argv, 'return_code': result.returncode,
            'stdout': result.stdout[-24000:], 'stderr': result.stderr[-8000:]}


def inspect():
    record = {'hostname': socket.gethostname(), 'devices': {}, 'libraries': {},
              'environment': {k: os.environ[k] for k in (
                  'NCCL_IB_HCA', 'NCCL_IB_DISABLE', 'NCCL_SOCKET_IFNAME',
                  'NCCL_NET', 'LD_LIBRARY_PATH') if k in os.environ}}
    for root in ('/dev/infiniband', '/sys/class/infiniband', '/etc/libibverbs.d'):
        record['devices'][root] = sorted(glob.glob(root + '/*'))
    record['ports'] = {}
    for pattern in ('/sys/class/infiniband/*/ports/*/state',
                    '/sys/class/infiniband/*/ports/*/link_layer'):
        for path in glob.glob(pattern):
            record['ports'][path] = Path(path).read_text().strip()
    record['commands'] = [command(argv) for argv in (
        ['ibv_devices'], ['ibv_devinfo'], ['rdma', 'link', 'show'],
        ['ldconfig', '-p'], ['nvidia-smi'])]
    library_paths = set()
    for pattern in ('/usr/lib*/**/libmlx5*.so*', '/lib*/**/libibverbs*.so*'):
        library_paths.update(str(Path(p).resolve()) for p in glob.glob(pattern, recursive=True))
    for path in sorted(library_paths):
        if Path(path).is_file():
            record['libraries'][path] = {'ldd': command(['ldd', path]),
                                        'versions': command(['readelf', '--version-info', path])}
    library = ctypes.util.find_library('ibverbs')
    record['ibverbs_library'] = library
    if library:
        try:
            verbs = ctypes.CDLL(library, use_errno=True)
            verbs.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
            verbs.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
            verbs.ibv_get_device_name.argtypes = [ctypes.c_void_p]
            verbs.ibv_get_device_name.restype = ctypes.c_char_p
            verbs.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            count = ctypes.c_int()
            devices = verbs.ibv_get_device_list(ctypes.byref(count))
            record['ibv_device_count'] = count.value
            record['ibv_devices'] = [verbs.ibv_get_device_name(devices[i]).decode()
                                     for i in range(count.value)] if devices else []
            record['ibv_errno'] = ctypes.get_errno()
            if devices:
                verbs.ibv_free_device_list(devices)
        except OSError as error:
            record['ibverbs_error'] = str(error)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observation = inspect()
    args.output.write_text(json.dumps(observation, indent=2) + '\n')
    print('RDMA_OBSERVATION', observation.get('ibv_device_count'),
          observation.get('ibv_devices'), flush=True)
