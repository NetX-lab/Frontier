"""Suspend cyclic GC only during JSON loading, then restore before audit work."""
import gc
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

source = Path(__file__).with_name("observe_gc.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == "3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c"
spec = importlib.util.spec_from_file_location("gc_observation", source)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
original = observer.stage._obs


def observation(name, **values):
    if name == "load_start":
        gc.disable()
        original("load_gc_disabled", enabled=gc.isenabled())
    original(name, **values)
    if name == "load_done":
        started = time.monotonic()
        gc.freeze()
        gc.enable()
        original("freeze_after_load", seconds=time.monotonic() - started,
                 objects=gc.get_freeze_count(), gc_enabled=gc.isenabled())


observer.stage._obs = observation
try:
    observer.run(sys.argv[1], sys.argv[2])
finally:
    gc.enable()
