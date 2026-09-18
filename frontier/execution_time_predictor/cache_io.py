"""Durable local cache persistence helpers for predictor artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


@contextmanager
def _atomic_destination(cache_file: str | os.PathLike[str], mode: str):
    """Publish a cache file only after the complete payload is on disk.

    The temporary file lives beside the destination so ``os.replace`` is an
    atomic same-filesystem operation.  A failed or interrupted dump therefore
    cannot expose a truncated final cache file to another predictor process.
    """

    destination = Path(cache_file)
    temporary_path: str | None = None
    try:
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        with os.fdopen(file_descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise


def atomic_pickle_dump(value: Any, cache_file: str | os.PathLike[str]) -> None:
    """Atomically publish a complete estimator or prediction pickle."""

    with _atomic_destination(cache_file, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def atomic_json_dump(value: Any, cache_file: str | os.PathLike[str]) -> None:
    """Atomically publish a complete JSON artifact manifest."""

    with _atomic_destination(cache_file, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def dataset_fingerprint(path: str | os.PathLike[str]) -> str:
    """Identify the exact source bytes to validate reuse of fitted artifacts."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
