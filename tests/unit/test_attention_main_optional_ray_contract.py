"""The non-Ray profiling path must tolerate an unusable optional Ray install."""

from __future__ import annotations

import builtins

from frontier.profiling.attention import main as attention_main


def test_optional_ray_loader_treats_binary_load_error_as_unavailable(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "ray":
            raise OSError("GLIBC_2.38 not found while loading _raylet.so")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert attention_main._try_import_ray() is None
