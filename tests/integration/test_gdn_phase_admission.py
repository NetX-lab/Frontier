"""Non-dummy Simulator regression for staggered hybrid request admission."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_staggered_gdn_admission(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
    )
    (tmp_path / "run.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-15000:]
    assert "GDN mixed batch" in result.stdout
    assert "prefill approximation" in result.stdout


def run_case(root):
    import pandas as pd

    from frontier.entities import Request
    from frontier.request_generator.synthetic_request_generator import SyntheticRequestGenerator
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from tests.integration.test_pr33_nondummy_acceptance import _config

    with pytest.MonkeyPatch.context() as patch:
        _, config = _config(root, "hybrid", "co-location", 1, 1, patch)
        config.cluster_config.replica_scheduler_config.batch_size_cap = 3
        # Full-attention layers retain their real mixed-batch estimator path.
        for filename in ("attention.csv", "attention_kernel_only.csv"):
            path = root / "profiles" / filename
            frame = pd.read_csv(path, keep_default_na=False)
            mixed = []
            for count in (1, 31):
                row = frame.iloc[0].to_dict()
                row.update(
                    is_true_mixed_batch=True, is_mixed_batch=True, is_prefill=True,
                    batch_size=2, total_batch_size=2, num_prefill_seqs=1,
                    prefill_chunk_size=count, total_prefill_tokens=count,
                    total_tokens=count + 1, decode_batch_size=1,
                    decode_avg_kv_cache_size=16, batch_composition_ratio=.5,
                    prefill_seq_lens=json.dumps([count]),
                    prefill_kv_cache_sizes=json.dumps([0]),
                )
                row["time_stats.attn_decode.median"] = .05
                mixed.append(row)
            pd.concat([frame, pd.DataFrame(mixed)], ignore_index=True).to_csv(path, index=False)
        requests = [Request(0, 16, 12), Request(.002, 32, 3),
                    Request(.002, 16, 3), Request(.2, 16, 2)]
        patch.setattr(SyntheticRequestGenerator, "generate", lambda self: requests)
        observed = []
        ownership = {}
        create = VLLMv1EngineReplicaScheduler._create_batch

        def observe(self, batch_requests, tokens):
            observed.append([(r.id, r.is_prefill_complete, n) for r, n in zip(batch_requests, tokens)])
            for request in batch_requests:
                slot = self._gdn_state_slot_manager.retain(request.id)
                assert ownership.setdefault(request.id, slot) is slot
            return create(self, batch_requests, tokens)

        patch.setattr(VLLMv1EngineReplicaScheduler, "_create_batch", observe)
        simulator = Simulator(config)
        try:
            simulator.run()
        finally:
            (root / "phase_admission_evidence.json").write_text(
                json.dumps({"batches": observed}, indent=2) + "\n"
            )
        assert all(r.completed for r in requests)
        assert observed
        assert any(len({phase for _, phase, _ in batch}) == 2 for batch in observed)
        assert observed[1] == [(requests[1].id, False, 31), (requests[0].id, True, 1)]
        assert ownership[requests[-1].id].slot_id == ownership[requests[0].id].slot_id
        assert any(not phase and n == 1 for batch in observed for _, phase, n in batch)
        cluster = simulator._global_scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
        replica_id = next(iter(simulator._clusters[ClusterType.MONOLITHIC].replicas))
        scheduler = cluster.get_replica_scheduler(replica_id, 0)
        assert scheduler._gdn_state_slot_manager.capacity == 3
        assert scheduler._gdn_state_slot_manager.active_request_ids == ()
        assert scheduler._allocation_map == {}
        assert scheduler.num_allocated_blocks == 0


if __name__ == "__main__":
    run_case(Path(sys.argv[1]))
