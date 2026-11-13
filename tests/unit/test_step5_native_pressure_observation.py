"""Exercise the existing pregrant observation at the locked native allocator seam."""

import ast
from enum import Enum
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import pytest

from tests.unit.test_step5_native_pd_r2_observation import configuration, events, r2


@pytest.mark.parametrize("status,reserved,scheduled,full_fit,demand", [
    ("RUNNING", 0, True, False, 4),
    ("WAITING", 0, True, False, 2),
    ("RUNNING", 2, True, False, 2),
    ("WAITING", 0, False, False, 4),
    ("WAITING", 2, True, True, 4),
])
def test_actual_allocate_slots_failure_retains_pregrant_demand(
        tmp_path, status, reserved, scheduled, full_fit, demand):
    source = os.environ.get("STEP5_NATIVE_SOURCE_ROOT")
    if not source:
        pytest.skip("Requires the locked native source root")
    path = Path(source) / "vllm/v1/core/kv_cache_manager.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "595b9fef9a1f5fcf67fb4a6cd4af817c24d411a2d4f8d2cf3bd22cb954703bff")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "allocate_slots")

    class Status(Enum):
        RUNNING = 1
        WAITING = 2
        PREEMPTED = 3

    namespace = {"RequestStatus": Status}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    calls = []

    class Coordinator:
        def remove_skipped_blocks(self, request_id, processed_tokens, num_prompt_tokens):
            calls.append(("retire", processed_tokens))
            manager.ids, manager.free_count = [0, 0, 0, 4], 3

        def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                num_encoder_tokens, total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=False, prefill_end=0):
            calls.append(("estimate", apply_admission_cap, manager.free_count))
            return demand

        def allocate_new_blocks(self, *args):
            pytest.fail("A native None must never enter grant")

    class Manager:
        allocate_slots = namespace["allocate_slots"]

        def get_block_ids(self, request_id):
            return [self.ids[:]]

    manager = Manager()
    manager.ids, manager.free_count = [1, 2, 3, 4], 0
    manager.coordinator = Coordinator()
    manager.max_model_len, manager.watermark_blocks = 7168, 2
    manager.empty_kv_cache_blocks = NS(blocks=((),))
    manager.block_pool = NS(get_num_free_blocks=lambda: manager.free_count)
    # Weak references are the same caller relationship used by scheduler_ready.
    class Scheduler:
        pass
    scheduler = Scheduler()
    scheduler.kv_cache_manager = manager
    observer = r2.Observation(configuration(tmp_path, resident_identity={}))
    observer.manager_schedulers[manager.coordinator] = weakref.ref(scheduler)
    r2.wrap(Coordinator, "get_num_blocks_to_allocate", observer, after=observer.allocation_demand)
    r2.wrap(Manager, "allocate_slots", observer, before=observer.allocation_before,
            after=observer.allocated, allocation_scope=True)
    request = NS(request_id="case", num_computed_tokens=6400, num_in_flight_tokens=1,
        num_prompt_tokens=6144, num_tokens=6401, status=Status[status])
    assert manager.allocate_slots(request, 1, reserved_blocks=reserved,
        has_scheduled_reqs=scheduled, full_sequence_must_fit=full_fit) is None
    row = events(observer)[-1]
    assert row["event"] == "pressure" and row["cause"] == "native_allocate_slots_returned_none"
    assert row["before"]["pool_free_blocks"] == 0
    assert row["pool_free_blocks_after"] == (0 if full_fit else 3)
    estimate = row["before"]["native_demands"]
    assert estimate == [dict(native_blocks_to_allocate=demand,
        pool_free_blocks=0 if full_fit else 3, apply_admission_cap=full_fit,
        reserved_blocks=reserved, has_scheduled_reqs=scheduled,
        request_status=status, native_watermark_blocks=2)]
    assert calls == ([('estimate', True, 0)] if full_fit else [
        ('retire', 6399), ('estimate', False, 3)])
    assert observer.local.allocation is None and not observer.disabled
