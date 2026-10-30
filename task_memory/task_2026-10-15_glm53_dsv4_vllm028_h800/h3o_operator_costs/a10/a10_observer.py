"""Task-local, read-only A11 observer; patches exist only in the A10 process.

Scope times are TP rank zero's operator costs including their gaps. Attention
uses the predictor's composed critical path. EP rank-zero scopes are attribution,
not a reconstruction of EP barriers. The signed remainder preserves coordination
and overlap rather than inventing a native segment for them.
"""

import bisect
import gzip
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
from unittest.mock import patch

from a10_common import sha256
from frontier.operators.families import (
    COMM_FAMILY, DEEPSEEK_V4_ATTENTION_FAMILY, DEEPSEEK_V4_MHC_FAMILY,
    MOE_FAMILY, SHARE_EXPERT_FAMILY,
)

PREPARE_SCOPES = {"dsv4_input_preparation"}
POST_SCOPES = {"dsv4_lm_head", "dsv4_logits_gather", "dsv4_sampling"}
OWN_SCOPES = {"kv_cache_block_zero", "dsv4_block_table_write"}
DISPATCH_SCOPES = ({op.name for family in (
    DEEPSEEK_V4_ATTENTION_FAMILY, DEEPSEEK_V4_MHC_FAMILY, MOE_FAMILY, SHARE_EXPERT_FAMILY,
) for op in family.operators}
    | {op.name for op in COMM_FAMILY.operators if op.name.startswith("dsv4_")}) - POST_SCOPES
SCOPE_MAPPING = {
    "prepare": sorted(PREPARE_SCOPES),
    "post": sorted(POST_SCOPES),
    "own_columns": sorted(OWN_SCOPES),
    "dispatch": sorted(DISPATCH_SCOPES),
    "unknown_scopes": "each is retained as its own column",
    "coordination_and_overlap": "signed composed time minus mapped segments and own columns",
}


def batch_shape(batch):
    """Use the V2 runner flags, not Frontier's request-level KV credit flags."""
    from frontier.attention.dsv4_features import dsv4_request_is_decoding
    from frontier.entities import DummyForwardBatch
    dummy = isinstance(batch, DummyForwardBatch)
    flags = [True] * batch.size if dummy else dsv4_request_is_decoding(batch)
    tokens = list(batch.num_tokens)
    step = {
        "context_requests": sum(not flag for flag in flags),
        "context_tokens": sum(q for q, flag in zip(tokens, flags) if not flag),
        "generation_requests": sum(flags),
        "generation_tokens": sum(q for q, flag in zip(tokens, flags) if flag),
    }
    return {
        "real": not dummy, "step": step,
        "local_tokens": sum(tokens), "local_requests": batch.size,
        "classifier": {"has_prefill": bool(step["context_requests"]),
                       "max_query_len": max(tokens, default=0),
                       "uniform_token_count": 1 if dummy or not step["context_requests"] else None},
    }


def frontier_step_class(role, batch):
    # Import the native reducer, so the two classifiers have a single owner.
    from a10_common import REPO, load_module
    import sys
    native = sys.modules.get("v4_runtime_census") or load_module(
        "v4_runtime_census", REPO / "tests/analysis/v4_runtime_census.py")
    return native.step_class(role.lower(), batch_shape(batch))


def collective_request(backend, alias, implementation, data_size_bytes,
                       num_devices, comm_domain, sp_rows, predicted_ms):
    key = backend._curve_key(alias, implementation, num_devices, comm_domain)
    rows = tuple(sp_rows) if sp_rows is not None else None
    unequal = key[:2] == ("reduce_scatterv", "ep") and rows is not None and max(rows) != min(rows)
    if unequal:
        # Unequal curves admit only an exact (ordered rows, bytes) measurement.
        exact, lower, upper, ratio = True, data_size_bytes, data_size_bytes, None
        lower_us = upper_us = backend._unequal_combine_times[(key, rows, data_size_bytes)]
    else:
        rows = None
        curve = backend._curves[key]
        index = bisect.bisect_left(curve.sizes, data_size_bytes)
        exact = curve.sizes[index] == data_size_bytes
        lo, hi = (index, index) if exact else (index - 1, index)
        lower, upper = curve.sizes[lo], curve.sizes[hi]
        lower_us, upper_us = curve.times_us[lo], curve.times_us[hi]
        ratio = None if exact else math.log(data_size_bytes / lower) / math.log(upper / lower)
    return {"role": backend.cluster_type.name.lower(), "curve_key": list(key),
            "requested_bytes": data_size_bytes, "unequal_rows": rows,
            "exact": exact, "lower_bytes": lower, "upper_bytes": upper,
            "lower_time_us": lower_us, "upper_time_us": upper_us,
            "log_distance_ratio": ratio, "predicted_ms": predicted_ms}


class ForwardObserver:
    def __init__(self, root):
        self.root = root
        self.row_count = 0
        self.stage_end_events = 0
        self.stale_stage_end_events = 0
        self._scope_times = defaultdict(Counter)
        self._segments = defaultdict(Counter)
        self._seen_layers = defaultdict(set)
        self._sync = {}
        self._collectives = {}
        self._wave_stack = []
        self._pending = {}
        self._formed = {}
        self._patches = ExitStack()

    def _key(self, role, batch):
        return (role.name.lower(), batch.id, batch.schedule_epoch)

    def _add_scopes(self, key, timing):
        """Retain each exposed scope, without flattening attention concurrency."""
        layer = timing.layer_execution_times[0]
        gaps = getattr(layer, "_dsv4_operator_gap_ms", {})
        values = {name: value + gaps.get(name, 0.0) for name, value in timing.op_times.items()}
        self._scope_times[key].update(values)
        segments = self._segments[key]
        for name, value in values.items():
            segment = ("prepare" if name in PREPARE_SCOPES else "post" if name in POST_SCOPES
                       else "dispatch" if name in DISPATCH_SCOPES else name)
            segments[segment] += value
        if layer.attention_family_id == "deepseek_v4_attention":
            # Replace serial attention attribution by its exposed critical path.
            attention = getattr(layer, "_dsv4_attention_schedule", None)
            if attention is not None:
                raw = sum(values.get(name, 0.0) for name in attention.start_ms)
                segments["dispatch"] += attention.elapsed_ms - raw

    def __enter__(self):
        from frontier.events.batch_stage_end_event import BatchStageEndEvent
        from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
        from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
        from frontier.scheduler.utils import dp_dummy_forward
        from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import VLLMv1EngineReplicaScheduler
        from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend
        self._stream = gzip.open(self.root / "forwards.jsonl.gz", "wt")
        observer = self

        def install(owner, name, wrapper, static=False):
            original = getattr(owner, name)
            replacement = wraps(original)(wrapper(original))
            self._patches.enter_context(patch.object(owner, name, staticmethod(replacement) if static else replacement))

        def schedule_wrapper(original):
            def call(lane, time):
                batches = original(lane, time)
                for batch in batches:
                    key = observer._key(lane._cluster_type, batch)
                    if lane._runs_dp_forwards_ahead:
                        unfinished = list(lane._unfinished_forwards)
                        ahead = next(index for index, forward in enumerate(unfinished)
                                     if forward.batch_id == batch.id)
                    else:
                        ahead = 0
                    observer._formed[key] = {
                        "formed_s": time,
                        "formation_index": batch._forward_cohort_provisional_id,
                        "queued_ahead_at_formation": ahead,
                        "request_ids": [request.id for request in batch.requests],
                    }
                return batches
            return call

        def sync_wrapper(original):
            def call(scheduler, replica_id, batches):
                local = {lane: {**batch_shape(batch), "desired_mode": batch.get_decode_cuda_graph_runtime_mode()}
                         for lane, batch in batches.items()}
                result = original(scheduler, replica_id, batches)
                widths = [batch.get_effective_total_tokens_for_compute(scheduler._cluster_type)
                          for _, batch in sorted(batches.items())]
                for lane, batch in batches.items():
                    observer._sync[observer._key(scheduler._cluster_type, batch)] = {
                        "dp": lane, "num_tokens_across_dp": widths,
                        "local_classifier": local[lane]["classifier"],
                        "desired_mode": local[lane]["desired_mode"],
                        "peer_modes": [peer.get_decode_cuda_graph_runtime_mode()
                                       for peer_lane, peer in sorted(batches.items()) if peer_lane != lane],
                        "peer_step_classes": [frontier_step_class(scheduler._cluster_type.name, peer)
                                             for peer_lane, peer in sorted(batches.items()) if peer_lane != lane],
                    }
                return result
            return call

        def predict_wrapper(original):
            def call(predictor, batch, stage_id, cluster_type, *args, **kwargs):
                result = original(predictor, batch, stage_id, cluster_type, *args, **kwargs)
                # The A10 source attention path calls one layer, include_ffn=False.
                # Final metrics/full-stage predictions and repeated last layers are excluded.
                if result.num_layers == 1 and kwargs.get("include_ffn") is False:
                    key = observer._key(cluster_type, batch)
                    layer_id = result.layer_execution_times[0].global_layer_id
                    if layer_id not in observer._seen_layers[key]:
                        observer._seen_layers[key].add(layer_id)
                        observer._add_scopes(key, result)
                return result
            return call

        def wave_wrapper(original):
            def call(scheduler, **kwargs):
                observer._wave_stack.append((scheduler, kwargs))
                try:
                    return original(scheduler, **kwargs)
                finally:
                    observer._wave_stack.pop()
            return call

        def phases_wrapper(original):
            def call(execution_time, **kwargs):
                result = original(execution_time, **kwargs)
                if observer._wave_stack:
                    scheduler, wave = observer._wave_stack[-1]
                    tp = scheduler._config.replica_config.attn_tensor_parallel_size
                    dp, rank = divmod(kwargs["ep_id"], tp)
                    if rank == 0:
                        batch = wave["wave_inputs"].source_batches[dp]
                        observer._add_scopes(observer._key(scheduler._cluster_type, batch), execution_time)
                return result
            return call

        def init_wrapper(original):
            def call(event, *args, **kwargs):
                original(event, *args, **kwargs)
                batch, stage, role = event._batch, event._batch_stage, event._cluster_type
                key = observer._key(role, batch)
                shape = batch_shape(batch)
                metadata = batch.decode_cuda_graph_metadata
                composed_ms = stage.execution_time * 1000
                parts = dict(observer._segments.pop(key, {}))
                parts.setdefault("prepare", 0.0)
                parts.setdefault("dispatch", 0.0)
                parts.setdefault("post", 0.0)
                parts["coordination_and_overlap"] = composed_ms - sum(parts.values())
                observer._pending[id(event)] = {
                    "role": role.name.lower(), "replica": event._replica_id,
                    "dp": event._replica_local_id, "stage_id": event._stage_id,
                    "batch_id": batch.id, "batch_stage_id": stage.id,
                    "schedule_epoch": batch.schedule_epoch, "phase": "serving",
                    "start_s": stage.scheduled_at, "end_s": event.time, **shape,
                    "step_class": frontier_step_class(role.name, batch),
                    "decode_batch_size": shape["step"]["generation_requests"] if not shape["step"]["context_requests"] else None,
                    "mode": batch.get_decode_cuda_graph_runtime_mode(),
                    "synchronized_width": batch.get_effective_total_tokens_for_compute(role),
                    "descriptor": asdict(metadata) if metadata is not None else None,
                    "composed_ms": composed_ms, "segment_ms": parts,
                    **observer._formed.pop(key),
                    "scope_ms": dict(observer._scope_times.pop(key, {})),
                    **observer._sync.pop(key, {}),
                }
                observer._seen_layers.pop(key, None)
            return call

        def end_wrapper(original):
            def call(event, *args, **kwargs):
                observer.stage_end_events += 1
                row = observer._pending.pop(id(event))
                stale = event._batch.schedule_epoch != event._batch_schedule_epoch
                result = original(event, *args, **kwargs)
                if stale:
                    observer.stale_stage_end_events += 1
                else:
                    observer._stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    observer.row_count += 1
                return result
            return call

        def collective_wrapper(original):
            def call(backend, alias, implementation, data_size_bytes, num_devices, comm_domain, sp_rows=None):
                predicted = original(backend, alias, implementation, data_size_bytes,
                                     num_devices, comm_domain, sp_rows)
                # No row is created if original raises; its exception is untouched.
                row = collective_request(backend, alias, implementation, data_size_bytes,
                                         num_devices, comm_domain, sp_rows, predicted)
                key = (row["role"], tuple(row["curve_key"]), data_size_bytes, row["unequal_rows"])
                if key not in observer._collectives:
                    observer._collectives[key] = {**row, "call_count": 0, "predicted_total_ms": 0.0}
                observer._collectives[key]["call_count"] += 1
                observer._collectives[key]["predicted_total_ms"] += predicted
                return predicted
            return call

        install(VLLMv1EngineReplicaScheduler, "on_schedule", schedule_wrapper)
        install(dp_dummy_forward, "_synchronize_dp_descriptors", sync_wrapper)
        install(SklearnMoEExecutionTimePredictor, "predict_stage_execution_time", predict_wrapper)
        install(BaseClusterScheduler, "_prepare_moe_ep_wave_plan", wave_wrapper)
        install(BaseClusterScheduler, "_get_shared_ep_phase_times_ms", phases_wrapper, static=True)
        install(BatchStageEndEvent, "__init__", init_wrapper)
        install(BatchStageEndEvent, "handle_event", end_wrapper)
        install(MeasuredCCBackend, "predict_operator_collective", collective_wrapper)
        return self

    def __exit__(self, *exc):
        self._patches.close()
        self._stream.close()
        path = self.root / "collectives.json"
        path.write_text(json.dumps({"schema": "a11-collectives-v1", "rows": list(self._collectives.values())}, indent=1) + "\n")
        return False

    def receipt(self):
        forwards, collectives = self.root / "forwards.jsonl.gz", self.root / "collectives.json"
        return {"forwards": {"path": str(forwards), "rows": self.row_count, "sha256": sha256(forwards)},
                "collectives": {"path": str(collectives), "sha256": sha256(collectives),
                                "calls": sum(row["call_count"] for row in self._collectives.values())},
                "stage_end_events": self.stage_end_events,
                "stale_stage_end_events": self.stale_stage_end_events,
                "scope_mapping": SCOPE_MAPPING}
