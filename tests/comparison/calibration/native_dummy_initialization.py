"""Run a diagnostic-only native dummy-weight model and inspect its initialization."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path


class TensorInventory:
    def native_tensor_inventory(self):
        return json.dumps(inspect_model(self.get_model(), self.vllm_config), default=str)


def inspect_model(model, config):
    import torch
    tensors = []
    modules = dict(model.named_modules())
    for kind, entries in (("parameter", model.named_parameters()),
                          ("buffer", model.named_buffers())):
        for name, tensor in entries:
            owner_name, _, field = name.rpartition(".")
            owner = modules[owner_name]
            method = getattr(owner, "quant_method", None)
            row = {"name": name, "kind": kind, "shape": list(tensor.shape),
                   "dtype": str(tensor.dtype), "elements": tensor.numel()}
            values = tensor.detach().float()
            row["finite"] = bool(torch.isfinite(values).all().item())
            if tensor.numel():
                row.update(minimum=values.min().item(), maximum=values.max().item())
            if name.endswith("tid2eid"):
                row["valid_expert_ids"] = bool(
                    ((tensor >= 0) & (tensor < config.model_config.hf_config.n_routed_experts))
                    .all().item())
            if "scale" in field:
                row["positive"] = bool(tensor.numel() and (values > 0).all().item())
                if (type(method).__name__ == "Fp8MoEMethod"
                        and method.block_quant
                        and method.fp8_backend.name == "TRITON"
                        and field in ("w13_weight_scale_inv", "w2_weight_scale_inv")):
                    row["scale_role"] = "triton_fp32_block_multiplier"
                    weight_field = "w13_weight" if field == "w13_weight_scale_inv" else "w2_weight"
                    weight = getattr(owner, weight_field)
                    block_n, block_k = owner.weight_block_size
                    expected = [weight.shape[0], math.ceil(weight.shape[1] / block_n),
                                math.ceil(weight.shape[2] / block_k)]
                    row["scale_valid"] = (tensor.dtype == torch.float32 and row["finite"]
                                          and weight.dtype == torch.float8_e4m3fn
                                          and row["shape"] == expected)
                elif (type(method).__name__ == "Fp8LinearMethod" and method.block_quant
                      and field == "weight_scale_inv"
                      and type(method.fp8_linear).__name__ == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
                      and method.fp8_linear.fallback.use_deep_gemm_e8m0):
                    row["scale_role"] = "deepgemm_ue8m0_fp32_multiplier"
                    _, exponents = torch.frexp(values)
                    expected = [math.ceil(size / block) for size, block in
                                zip(owner.weight.shape, owner.weight_block_size)]
                    row["scale_valid"] = (tensor.dtype == torch.float32 and row["finite"]
                        and owner.weight.dtype == torch.float8_e4m3fn
                        and row["shape"] == expected and row["positive"] and bool((values == torch.ldexp(
                            torch.ones_like(values), exponents - 1)).all().item()))
                elif (type(owner).__name__ == "MLAAttention" and owner.kv_cache_dtype == "auto"
                      and field in ("_q_scale", "_k_scale", "_v_scale", "_prob_scale")):
                    row["scale_role"] = "unquantized_attention_unit_scale"
                    row["scale_valid"] = (tensor.dtype == torch.float32 and row["shape"] == []
                                          and bool((values == 1).all().item()))
                else:
                    row["scale_role"] = "unverified"
                    row["scale_valid"] = False
            tensors.append(row)
    auxiliary = []
    backends = {}
    for name, module in modules.items():
        method = getattr(module, "quant_method", None)
        if type(module).__name__ == "MLAAttention":
            host = {f"_{field}_scale_float": getattr(module, f"_{field}_scale_float")
                    for field in ("q", "k", "v", "prob")}
            copies = {field: getattr(module, f"_{field}_scale_cpu") for field in ("k", "v")}
            valid = all(math.isfinite(value) and value == getattr(module, field[:-6]).item()
                        for field, value in host.items())
            valid = valid and all(tensor.device.type == "cpu" and tensor.dtype == torch.float32
                and tensor.numel() == 1 and tensor.item() == getattr(module, f"_{field}_scale").item()
                for field, tensor in copies.items())
            auxiliary.append({"name": name, "role": "attention_host_scale_copies", "host": host,
                              "cpu": {field: tensor.item() for field, tensor in copies.items()},
                              "valid": valid})
            backends[name] = {field: str(getattr(module, field)) if isinstance(getattr(module, field), type)
                             else type(getattr(module, field)).__module__ + "." + type(getattr(module, field)).__name__
                             for field in ("attn_backend", "impl", "prefill_backend")}
            expected_weights = {"W_UK_T": [module.num_heads, module.qk_nope_head_dim, module.kv_lora_rank],
                                "W_UV": [module.num_heads, module.kv_lora_rank, module.v_head_dim]}
            auxiliary.append({"name": name, "role": "mla_derived_weights", "valid": all(
                list(getattr(module, field).shape) == shape and getattr(module, field).dtype == config.model_config.dtype
                for field, shape in expected_weights.items())})
            backends[name]["dcp_q_replicate"] = module.dcp_q_replicate
            if module.dcp_q_replicate:
                tensor = module.W_UK_T_dcp_qrep
                auxiliary.append({"name": name + ".W_UK_T_dcp_qrep", "role": "dcp_derived_weight",
                                  "shape": list(tensor.shape), "valid": bool(torch.isfinite(tensor.float()).all().item())})
        if type(module).__name__ == "Indexer":
            backends[name] = {field: getattr(module, field) for field in
                             ("scale_fmt", "quant_block_size", "softmax_scale", "n_head_scale", "use_fused_indexer_q")}
            backends[name]["topk_workspace"] = {"shape": list(module.topk_indices_buffer.shape),
                "dtype": str(module.topk_indices_buffer.dtype), "lifecycle": "forward_written_valid_region"}
        if type(method).__name__ == "Fp8LinearMethod":
            backends[name] = {"quant_method": type(method).__name__,
                              "linear_kernel": type(method.fp8_linear).__name__}
        if type(method).__name__ == "Fp8MoEMethod" and method.block_quant and method.fp8_backend.name == "TRITON":
            kernel = method.moe_kernel
            quant = method.moe_quant_config
            backends[name] = {"quant_method": type(method).__name__, "backend": method.fp8_backend.name,
                              "experts": type(kernel.fused_experts).__name__,
                              "prepare_finalize": type(kernel.prepare_finalize).__name__,
                              "block_shape": quant.block_shape}
            auxiliary.append({"name": name, "role": "moe_quant_scale_aliases",
                              "valid": quant.w1_scale is module.w13_weight_scale_inv
                                  and quant.w2_scale is module.w2_weight_scale_inv
                                  and list(quant.block_shape) == list(module.weight_block_size)
                                  and quant.a1_scale is None and quant.a2_scale is None})
    return {"architecture": type(model).__name__, "tensors": tensors,
            "auxiliary": auxiliary, "backends": backends,
            "modules": {name: type(module).__module__ + "." + type(module).__name__
                        for name, module in model.named_modules()},
            "resolved": {name: {key: getattr(getattr(config, name), key) for key in keys}
                         for name, keys in {
                             "model_config": ("architecture", "quantization", "dtype"),
                             "load_config": ("load_format",),
                             "parallel_config": ("tensor_parallel_size", "enable_expert_parallel", "all2all_backend"),
                             "scheduler_config": ("async_scheduling", "max_num_batched_tokens", "max_num_seqs"),
                             "kernel_config": ("moe_backend",),
                             "cache_config": ("cache_dtype", "block_size"),
                         }.items()},
            "hf_config": config.model_config.hf_config.to_dict()}


def run(config):
    output = Path(config["output_dir"])
    if not output.is_relative_to(Path("/mnt/codesign-exp/ycfeng")):
        raise ValueError("Output must remain inside the personal cloud directory")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "worker.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    result = {"scope": "G0-A diagnostic only; reduced layers and expert count",
              "configuration": config, "host": socket.gethostname(), "passed": False}

    def record(stage):
        result["stage"] = stage
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print("G0A_STAGE", stage, flush=True)

    try:
        scratch = tempfile.mkdtemp(prefix="g0a_")
        for variable, child in (("HF_HOME", "hf"), ("VLLM_CACHE_ROOT", "vllm"),
                                ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                ("FLASHINFER_WORKSPACE_BASE", "flashinfer")):
            os.environ[variable] = str(Path(scratch) / child)
        record("imports")
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        result["versions"] = {"python": sys.version, "torch": torch.__version__,
                              "vllm": vllm.__version__, "cuda": torch.version.cuda}
        result["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "capability": torch.cuda.get_device_capability(0)}
        if vllm.__version__ != config["expected_vllm_version"]:
            raise RuntimeError("Unexpected vLLM version")
        source_root = Path(vllm.__file__).parent.parent
        result["source_hashes"] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                   for name in config["expected_source_hashes"]}
        if result["source_hashes"] != config["expected_source_hashes"]:
            raise RuntimeError("Installed model/initializer sources differ from the pinned commit")
        record("dummy_initialization")
        llm_args = dict(config["llm"],
                        worker_extension_cls="native_dummy_initialization.TensorInventory")
        llm = LLM(**llm_args)
        record("tensor_inventory")
        result["inventory"] = [json.loads(value) for value in
                               llm.collective_rpc("native_tensor_inventory")]
        record("generation")
        prompts = [{"prompt_token_ids": [config["input_token_id"]] * length}
                   for length in config["prompt_lengths"]]
        responses = llm.generate(prompts, SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=config["decode_tokens"],
            logprobs=1), use_tqdm=False)
        result["responses"] = [{"prompt_tokens": len(response.prompt_token_ids),
                                "token_ids": response.outputs[0].token_ids,
                                "finish_reason": response.outputs[0].finish_reason,
                                "logprobs": response.outputs[0].logprobs}
                               for response in responses]
        result["checks"] = {
            "finite_initialization": all(row["finite"] for rank in result["inventory"]
                                          for row in rank["tensors"]),
            "hash_ids_valid": all(row.get("valid_expert_ids", True)
                                  for rank in result["inventory"] for row in rank["tensors"]),
            "scale_roles_valid": all(row.get("scale_valid", True)
                                      for rank in result["inventory"] for row in rank["tensors"]),
            "auxiliary_valid": all(row["valid"] for rank in result["inventory"]
                                   for row in rank["auxiliary"]),
            "finite_selected_logprobs": all(math.isfinite(value.logprob)
                for response in responses for token in response.outputs[0].logprobs
                for value in token.values()),
            "decode_counts": all(len(row["token_ids"]) == config["decode_tokens"]
                                 for row in result["responses"]),
            "prompt_counts": [row["prompt_tokens"] for row in result["responses"]]
                             == config["prompt_lengths"],
        }
        result["mapped_libraries"] = sorted({line.split()[-1] for line in
            Path("/proc/self/maps").read_text().splitlines() if ".so" in line and "/" in line})
        cudart = [path for path in result["mapped_libraries"] if "libcudart.so" in path]
        result["cudart"] = {}
        for path in cudart:
            version = ctypes.c_int()
            status = ctypes.CDLL(path).cudaRuntimeGetVersion(ctypes.byref(version))
            result["cudart"][path] = {"runtime_version": version.value, "returncode": status}
        result["passed"] = all(result["checks"].values())
        record("completed")
        llm.llm_engine.engine_core.shutdown()
        record("shutdown_returned")
        return 0 if result["passed"] else 1
    except Exception:
        result["exception"] = traceback.format_exc()
        record("failed")
        print(result["exception"], flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(json.loads(args.config.read_text())))
