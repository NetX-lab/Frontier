"""Run-lifetime collective profiling inputs, shared by role backends."""

import json
import os
from pathlib import Path


class CollectiveProfileInputs:
    """Retain the first successfully parsed bytes for each configured path."""

    def __init__(self):
        self._inputs = {}

    def read(self, profiling_data_dir, device_type):
        path = Path(os.path.abspath(Path(profiling_data_dir) / device_type / "collective_local.json"))
        if path not in self._inputs:
            raw = path.read_bytes()
            json.loads(raw.decode("utf-8"))
            self._inputs[path] = raw
        return self._inputs[path]
