"""Execute pinned native retention/connector bodies on CPU-only block fixtures."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace


REVISION = "2cf0a6915ce544dc493a0990f2ea38d81601128a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {}

    def body(path, cls, name):
        text = subprocess.check_output(
            ["git", "show", f"{REVISION}:{path}"], cwd=args.native_repo, text=True,
        )
        tree = ast.parse(text)
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
        evidence[f"{path}:{node.lineno}-{node.end_lineno}"] = hashlib.sha256(text.encode()).hexdigest()
        node.decorator_list = []
        namespace = {"cdiv": lambda a, b: (a + b - 1) // b, "SupportsHMA": HMA}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
        return namespace[name]

    class HMA:
        pass

    manager_path = "vllm/v1/core/single_type_kv_cache_manager.py"
    base = type("NativeManager", (), {name: body(manager_path, "SingleTypeKVCacheManager", name) for name in (
        "remove_skipped_blocks", "_remove_blocks_in_range", "add_local_computed_blocks",
        "allocate_external_computed_blocks",
    )})
    skip = body(manager_path, "SlidingWindowManager", "get_num_skipped_tokens")
    full_skip = body(manager_path, "SingleTypeKVCacheManager", "get_num_skipped_tokens")
    clip = body("vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py", "NixlBaseConnectorScheduler", "get_exchange_clipped_blocks")
    trim = body("vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py", "NixlBaseConnectorWorker", "_apply_prefix_caching")
    finish = body("vllm/v1/core/sched/scheduler.py", "Scheduler", "_connector_finished")
    destination_ids = body("vllm/v1/core/kv_cache_manager.py", "KVCacheBlocks", "get_unhashed_block_ids_all_groups")
    source_ids = body("vllm/v1/core/kv_cache_manager.py", "KVCacheManager", "get_block_ids_for_computed_tokens")
    coordinator_reclaim = body("vllm/v1/core/kv_cache_coordinator.py", "KVCacheCoordinator", "remove_skipped_blocks")
    # AttentionSpec dispatch is represented by the fixture specs below.
    source_ids.__globals__.update(AttentionSpec=SimpleNamespace, CrossAttentionSpec=type("Cross", (), {}), EncoderOnlyAttentionSpec=type("Encoder", (), {}))
    groups = [("compressed", 256, 0), ("swa_0", 64, 128), ("swa_1", 64, 128), ("csa_state_0", 4, 8), ("hca_state_0", 8, 128)]
    results = []
    for prompt in (191, 6144, 6145):
        next_id = 1

        def block():
            nonlocal next_id
            value = SimpleNamespace(block_id=next_id, is_null=False, block_hash=None)
            next_id += 1
            return value

        null = SimpleNamespace(block_id=0, is_null=True, block_hash=None)

        def manager(size, window, source):
            m = base()
            m.block_size, m.sliding_window, m.extra_retained_tokens = size, window, 0
            m.get_num_skipped_tokens = (skip if window else full_skip).__get__(m)
            m._null_block, m.enable_caching, m._record_new_block_ids = null, False, False
            m._has_partial_local_hit = lambda *args: False
            m.req_to_blocks = defaultdict(list)
            m.num_cached_block = {}
            m.block_pool = SimpleNamespace(free_blocks=lambda blocks: None, get_new_blocks=lambda n: [block() for _ in range(n)])
            if source:
                m.req_to_blocks["r"] = [block() for _ in range((prompt + size - 1) // size)]
            else:
                m.add_local_computed_blocks("r", [], 0, prompt)
                m.allocate_external_computed_blocks("r", 0, prompt)
            return m

        sources = [manager(size, window, True) for _, size, window in groups]
        targets = [manager(size, window, False) for _, size, window in groups]
        before = [len(m.req_to_blocks["r"]) for m in sources]
        connector = HMA()
        connector._is_hma_required = True
        connector.blocks_per_sw = [(window + size - 1) // size + 1 if window else 0 for _, size, window in groups]
        connector._ssm_spec_blocks = [None] * 5
        connector.request_finished_all_groups = lambda req, ids: clip(connector, ids)
        specs = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size)) for _, size, _ in groups])
        coordinator = SimpleNamespace(single_type_managers=sources)
        adapter = SimpleNamespace(
            remove_skipped_blocks=lambda **kwargs: coordinator_reclaim(coordinator, **kwargs),
            get_block_ids=lambda request_id: tuple([b.block_id for b in m.req_to_blocks[request_id]] for m in sources),
            kv_cache_config=specs,
        )
        adapter.get_block_ids_for_computed_tokens = lambda **kwargs: source_ids(adapter, **kwargs)
        scheduler = SimpleNamespace(connector=connector, kv_cache_manager=adapter)
        request = SimpleNamespace(request_id="r", num_computed_tokens=prompt, num_in_flight_tokens=0, num_prompt_tokens=prompt)
        exchanged = finish(scheduler, request)
        held = [[b.block_id for b in m.req_to_blocks["r"] if not b.is_null] for m in sources]
        destination = destination_ids(SimpleNamespace(blocks=[m.req_to_blocks["r"] for m in targets]))
        destination = clip(connector, destination)
        _, transferred = trim(SimpleNamespace(_has_mamba=False), destination, exchanged, 1, 1)
        assert list(map(len, held)) == list(map(len, destination)) == list(map(len, transferred))
        assert all(list(actual) == expected for actual, expected in zip(transferred, held))
        if prompt == 6144:
            assert before == [24, 96, 96, 1536, 768]
            assert list(map(len, held)) == [24, 2, 2, 2, 16]
            assert list(map(len, exchanged)) == [24, 3, 3, 3, 17]
        results.append({"prompt": prompt, "initial_allocated": before, "source_held": list(map(len, held)), "connector_exchanged_including_null": list(map(len, exchanged)), "destination_reserved": list(map(len, destination)), "transferred": list(map(len, transferred))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"native_revision": REVISION, "checks": results, "source_sha256": evidence, "scope": "extracted actual native bodies; fixture objects, no native runtime import or device execution"}, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
