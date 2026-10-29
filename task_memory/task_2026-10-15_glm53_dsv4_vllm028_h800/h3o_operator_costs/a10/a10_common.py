"""File receipts and path imports shared by the A10/A9 CPU tools."""

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[1]
HASH_CHUNK_BYTES = 1024 * 1024
LOCK_SUFFIX = "_lock.file"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_record(path):
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows)  # CSV header; missing/empty inputs are errors.
        count = sum(1 for _ in rows)
    return {"path": str(path), "sha256": sha256(path), "rows": count}


def cache_inventory(directory):
    return [
        {"name": str(path.relative_to(directory)),
         **({} if path.name.endswith(LOCK_SUFFIX) else {"sha256": sha256(path)})}
        for path in sorted(Path(directory).rglob("*")) if path.is_file()
    ]
