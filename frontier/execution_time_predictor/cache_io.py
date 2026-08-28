"""Durable local cache persistence helpers for predictor artifacts."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def atomic_pickle_dump(value: Any, cache_file: str | os.PathLike[str]) -> None:
    """Publish a pickle only after the complete payload is on disk.

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
        with os.fdopen(file_descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
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
