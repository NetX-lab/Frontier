"""Run full-structure native GLM startup and retain unqualified G0-B observations."""

import argparse
import dataclasses
import json
import math
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path

from native_dummy_initialization import inspect_model
from probe_nixl_read import SYSFS_INFINIBAND, gid_table, roce_v2_ipv4_ports
from probe_runtime_inventory import primary_ipv4, read_text, sha256, wait_for, weight_candidates, write_json


def identity(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__name__


class FullModelInventory:
    def full_model_inventory(self):
        import torch
        from vllm.distributed import get_world_group, get_tp_group, get_dp_group, get_ep_group
        from vllm.v1.worker.workspace import current_workspace_manager
        runner, config = self.model_runner, self.vllm_config
        result = {"host": socket.gethostname(), "worker_rank": self.rank,
                  "device": str(self.device), "request_owner": config.parallel_config.data_parallel_rank}
        result["groups"] = {name: {"rank": group.rank, "ranks": group.ranks,
            "world_size": group.world_size, "rank_in_group": group.rank_in_group,
            "communicator": identity(group.device_communicator)} for name, group in
            (("world", get_world_group()), ("tp", get_tp_group()),
             ("dp", get_dp_group()), ("ep", get_ep_group()))}
        result["resolved"] = {name: str(getattr(config, name)) for name in
            ("parallel_config", "scheduler_config", "compilation_config", "kernel_config")}
        result["scheduler_class"] = identity(config.scheduler_config.get_scheduler_cls())
        result["moe_backends"] = {name: {"method": identity(method), "backend": method.fp8_backend.name,
            "experts": identity(method.moe_kernel.fused_experts),
            "prepare_finalize": identity(method.moe_kernel.prepare_finalize),
            "block_shape": method.moe_quant_config.block_shape, "scale_validity": "unverified"}
            for name, module in self.get_model().named_modules()
            if (method := getattr(module, "quant_method", None)) is not None and type(method).__name__ == "Fp8MoEMethod"}
        result["cache"] = dataclasses.asdict(runner.kv_cache_config)
        result["attention_groups"] = [{"layers": g.layer_names, "backend": identity(g.backend),
            "builders": [identity(b) for b in g.metadata_builders]} for groups in runner.attn_groups for g in groups]
        result["memory"] = {"model_load_delta_bytes": runner.model_memory_usage,
            "allocator_high_water_bytes": torch.cuda.max_memory_allocated(self.device),
            "allocator_reserved_high_water_bytes": torch.cuda.max_memory_reserved(self.device),
            "current_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "native_scalars": {field: getattr(self, field, None) for field in
                ("requested_memory", "total_consumed", "peak_activation_memory",
                 "cudagraph_memory_estimate", "available_kv_cache_memory_bytes")},
            "independent_stage_peaks": "unverified; these scalars are not separate stage peaks"}
        workspace = current_workspace_manager()
        result["workspace_bytes"] = [0 if t is None else t.numel() * t.element_size()
                                     for t in workspace._current_workspaces]
        try:
            result["initialization"] = inspect_model(self.get_model(), config)
        except Exception:
            result["initialization_error"] = traceback.format_exc()
        return json.dumps(result, default=str)


def routing_rows(array, prompt_tokens, output_tokens, hf):
    import numpy as np
    expected = (prompt_tokens + output_tokens - 1, hf["num_hidden_layers"], hf["num_experts_per_tok"])
    if array.shape != expected:
        raise ValueError(f"Routing shape {array.shape} differs from expected {expected}")
    layers = [i for i, kind in enumerate(hf["mlp_layer_types"]) if kind == "sparse"]
    rows = []
    for phase, values in (("prompt_inputs", array[:prompt_tokens]),
                          ("decode_inputs", array[prompt_tokens:])):
        for layer in layers:
            ids = values[:, layer].reshape(-1)
            if not np.issubdtype(ids.dtype, np.integer) or not ((ids >= 0) & (ids < hf["n_routed_experts"])).all():
                raise ValueError(f"Invalid routing IDs in layer {layer}, phase {phase}")
            rows.append({"phase": phase, "layer": layer, "token_inputs": len(values),
                         "counts": np.bincount(ids.astype(np.int64), minlength=hf["n_routed_experts"]).tolist()})
    return {"rows": rows, "non_moe_layers": [i for i in range(expected[1]) if i not in layers],
            "execution_shape_coverage": "unverified; final request arrays do not encode native batch shapes"}


def run(config):
    import numpy as np
    root = Path(config["output_dir"])
    if not root.is_relative_to(Path("/mnt/codesign-exp/ycfeng")):
        raise ValueError("Output must remain in the personal cloud directory")
    node = root / "nodes" / socket.gethostname()
    node.mkdir(parents=True, exist_ok=False)
    with (node / "worker.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    result = {"scope": "G0-B full structure; diagnostic observations, not complete qualification",
              "configuration": config, "passed": False, "observation_errors": []}

    def stage(name):
        result["stage"] = name
        write_json(node / "result.json", result)
        print("G0B_STAGE", name, flush=True)

    try:
        for path, digest in config["tool_hashes"].items():
            if sha256(path) != digest:
                raise RuntimeError(f"Tool hash mismatch: {path}")
        scratch = Path(tempfile.mkdtemp(prefix="g0b_"))
        for name, child in config["cache_environment"].items():
            os.environ[name] = str(scratch / child)
        stage("rendezvous")
        write_json(node / "ready.json", {"host": socket.gethostname(), "ip": primary_ipv4()})
        if not wait_for(lambda: len(list(root.glob("nodes/*/ready.json"))) == config["replica"], config["peer_wait_seconds"]):
            raise TimeoutError("Not all distinct physical nodes arrived")
        peers = sorted((json.loads(p.read_text()) for p in root.glob("nodes/*/ready.json")), key=lambda p: p["host"])
        rank = [p["host"] for p in peers].index(socket.gethostname())
        result["peers"], result["request_owner"] = peers, rank
        os.environ.update(VLLM_DP_SIZE=str(config["replica"]), VLLM_DP_RANK=str(rank),
            VLLM_DP_RANK_LOCAL="0", VLLM_DP_MASTER_IP=peers[0]["ip"], VLLM_DP_MASTER_PORT=str(config["dp_master_port"]))
        rdma = {d.name: {p.name: {"state": read_text(p / "state"), "link_layer": read_text(p / "link_layer"),
            "gids": gid_table(p)} for p in (d / "ports").glob("*")} for d in SYSFS_INFINIBAND.glob("*")}
        result["ucx_devices"] = roce_v2_ipv4_ports(rdma)
        if not result["ucx_devices"]:
            raise RuntimeError("No ACTIVE RoCE v2 IPv4 port")
        os.environ["UCX_NET_DEVICES"] = ",".join(result["ucx_devices"])
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        result["versions"] = {"python": sys.version, "torch": torch.__version__, "vllm": vllm.__version__}
        result["gpus"] = [{"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
        source = Path(vllm.__file__).parent.parent
        if vllm.__version__ != config["expected_vllm_version"] or any(sha256(source / p) != h for p, h in config["expected_source_hashes"].items()):
            raise RuntimeError("Pinned runtime/source identity mismatch")
        model = Path(config["llm"]["model"])
        if {p.name: sha256(p) for p in model.iterdir()} != config["model_hashes"]:
            raise RuntimeError("Metadata allowlist/hash mismatch")
        stage("full_initialization")
        llm = LLM(**config["llm"], worker_extension_cls="native_full_structure.FullModelInventory")
        stage("inventory")
        try:
            result["inventory"] = [json.loads(row) for row in llm.collective_rpc("full_model_inventory")]
        except Exception:
            result["observation_errors"].append(traceback.format_exc())
        stage("generation")
        hf = json.loads((model / "config.json").read_text())
        prompts = [{"prompt_token_ids": [config["input_token_id"]] * n} for n in config["prompt_lengths"][rank]]
        responses = llm.generate(prompts, SamplingParams(temperature=0, ignore_eos=True,
            max_tokens=config["decode_tokens"], logprobs=1), use_tqdm=False)
        result["responses"] = []
        for index, response in enumerate(responses):
            output = response.outputs[0]
            row = {"request_id": response.request_id, "prompt_tokens": len(response.prompt_token_ids),
                   "token_ids": output.token_ids, "finish_reason": output.finish_reason,
                   "logprobs_finite": all(math.isfinite(v.logprob) for token in output.logprobs for v in token.values())}
            row["routing"] = {"status": "unverified", "reason": "native routing-return observation disabled"}
            if config["llm"]["enable_return_routed_experts"]:
                try:
                    np.save(node / f"routing_{index}.npy", output.routed_experts, allow_pickle=False)
                    row["routing"] = routing_rows(output.routed_experts, row["prompt_tokens"], len(output.token_ids), hf)
                except Exception:
                    result["observation_errors"].append(traceback.format_exc())
            result["responses"].append(row)
        result["checks"] = {"prompt_counts": [r["prompt_tokens"] for r in result["responses"]] == config["prompt_lengths"][rank],
            "decode_counts": all(len(r["token_ids"]) == config["decode_tokens"] for r in result["responses"]),
            "finite_logprobs": all(r["logprobs_finite"] for r in result["responses"])}
        result["weight_audit"] = weight_candidates([scratch / "hf", model], tuple(config["weight_suffixes"]))
        result["checks"]["no_weight_files"] = not result["weight_audit"]
        result["passed"] = all(result["checks"].values())
        stage("generation_returned")
        write_json(node / "drained.json", {"request_owner": rank})
        if not wait_for(lambda: len(list(root.glob("nodes/*/drained.json"))) == config["replica"], config["peer_wait_seconds"]):
            raise TimeoutError("Not all DP owners returned from paired generation")
        llm.llm_engine.engine_core.shutdown()
        stage("shutdown_returned")
        return 0 if result["passed"] else 1
    except Exception:
        result["exception"] = traceback.format_exc()
        stage("failed")
        print(result["exception"], flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    sys.exit(run(json.loads(parser.parse_args().config.read_text())))
