"""Run a diagnostic-only native dummy-weight model and inspect its initialization."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


class TensorInventory:
    def native_tensor_inventory(self):
        return json.dumps(inspect_model(self.get_model(), self.vllm_config), default=str)


def _tensor_layout(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()), "device": str(tensor.device),
            "storage_offset": tensor.storage_offset()}


def _v4_attention_inventory(model, config, modules):
    import torch

    core, hf = model.model, config.model_config.hf_config
    device = next(model.parameters()).device
    rows = []
    for name, module in modules.items():
        kind = type(module).__name__
        cache_shapes = {"DeepseekV4SWACache": ("head_dim", torch.uint8),
                        "DeepseekV4IndexerCache": ("head_dim", torch.uint8),
                        "CompressorStateCache": ("state_dim", torch.float32)}
        if kind in cache_shapes or kind == "DeepseekV4FlashMLAAttention":
            cache = module.kv_cache
            if kind in ("DeepseekV4SWACache", "DeepseekV4FlashMLAAttention"):
                # Native fp8_ds_mla: NoPE bytes, BF16 RoPE and UE8M0 blocks plus padding.
                width = hf.head_dim + hf.qk_rope_head_dim + (hf.head_dim - hf.qk_rope_head_dim) // 64 + 1
                dtype = torch.uint8
            else:
                field, dtype = cache_shapes[kind]
                width = getattr(module, field)
            unused = kind == "DeepseekV4FlashMLAAttention" and module.compress_ratio <= 1
            valid = cache.numel() == 0 if unused else (
                cache.ndim == 3 and min(cache.shape) > 0 and cache.shape[-1] == width
                and cache.dtype == dtype and cache.device == device
                and cache.stride(-1) == 1 and cache.stride(-2) == width
                and cache.stride(0) >= cache.shape[1] * width)
            rows.append({"name": name + ".kv_cache", "role": "v4_cache_allocation",
                         "valid": valid, "layout": _tensor_layout(cache),
                         "lifecycle": "unused" if unused else "forward_written_valid_region",
                         "content_checked": False})
        if kind == "DeepseekCompressor":
            state = module.state_cache
            target = module._static_forward_context.get(module.k_cache_prefix)
            checks = {"state": state.state_dim == 2 * module.coff * module.head_dim
                          and state.dtype == torch.float32,
                      "ape": list(module.ape.shape) == [module.compress_ratio, module.coff * module.head_dim]
                          and module.ape.dtype == torch.float32,
                      "cache_owner": target is not None and modules.get(module.k_cache_prefix) is target,
                      "scratch_owner": module.eager_scratch_pool is core.eager_scratch_pool}
            scratch = module._compress_scratch
            checks["compress_scratch"] = ((isinstance(scratch, torch.Tensor)
                and list(scratch.shape) == [module.max_num_batched_tokens, module.head_dim]
                and scratch.dtype == torch.float32 and scratch.device == device)
                if module._use_two_stage_fused_compressor else scratch is None)
            rows.append({"name": name, "role": "v4_compressor_state", "checks": checks,
                         "valid": all(checks.values()), "content_checked": False})
    topk = getattr(core, "topk_indices_buffer", None)
    pool = getattr(core, "eager_scratch_pool", None)
    budget = config.scheduler_config.max_num_batched_tokens
    supported = topk is not None and type(pool).__name__ == "DeepseekV4EagerScratchPool"
    checks = {"selected_scratch_pool": supported}
    layouts = {}
    if supported:
        checks["topk"] = (list(topk.shape) == [budget, hf.index_topk]
                          and topk.dtype == torch.int32 and topk.device == device)
        layouts["topk"] = _tensor_layout(topk)
        heads = hf.num_attention_heads // config.parallel_config.tensor_parallel_size
        checks["q"] = (list(pool._q.shape) == [budget, 64 if heads <= 64 else 128, hf.head_dim]
                       and pool._q.dtype == torch.bfloat16 and pool._q.device == device)
        layouts["q"] = _tensor_layout(pool._q)
        checks["storage"] = (pool._storage.ndim == 1 and pool._storage.numel() > 0
                             and pool._storage.dtype == torch.uint8 and pool._storage.device == device)
        layouts["storage"] = _tensor_layout(pool._storage)
        variants = (("fp4", pool._fp4_template,
                     (([budget, hf.index_n_heads, hf.index_head_dim // 2], torch.uint8),
                      ([budget, hf.index_n_heads, hf.index_head_dim // 32], torch.uint8),
                      ([budget, hf.index_n_heads], torch.float32))),
                    ("global", pool._global_template,
                     (([budget, hf.index_topk], torch.int32), ([budget], torch.int32))),
                    ("compressor", (pool._compressor_template,),
                     (([budget, hf.head_dim], torch.float32),)))
        for label, tensors, specs in variants:
            offset, valid = 0, len(tensors) == len(specs)
            for index, (tensor, (shape, dtype)) in enumerate(zip(tensors, specs)):
                offset = math.ceil(offset / pool._ALIGNMENT) * pool._ALIGNMENT
                valid = valid and (list(tensor.shape) == shape and tensor.dtype == dtype
                    and tensor.device == device and tensor.is_contiguous()
                    and tensor.untyped_storage().data_ptr() == pool._storage.untyped_storage().data_ptr()
                    and tensor.storage_offset() * tensor.element_size() == offset)
                layouts[f"{label}.{index}"] = _tensor_layout(tensor)
                offset += math.prod(shape) * dtype.itemsize
            checks[label] = valid
    rows.append({"name": "model", "role": "v4_attention_scratch", "checks": checks,
                 "valid": all(checks.values()), "layouts": layouts,
                 "lifecycle": "forward_written_valid_region", "content_checked": False})
    for index in range(core.start_layer, core.end_layer):
        attn = getattr(core.layers[index], "attn", None)
        checks = {"backend": type(attn).__name__ == "DeepseekV4FlashMLAAttention"}
        if checks["backend"]:
            checks.update(cache_format=attn.kv_cache_dtype == "fp8_ds_mla"
                and attn.kv_cache_torch_dtype == torch.uint8,
                topk_owner=attn.topk_indices_buffer is topk,
                scratch_owner=attn.eager_scratch_pool is pool,
                compress_ratio=attn.compress_ratio == max(1, hf.compress_ratios[index]),
                compressor=(type(attn.compressor).__name__ == "DeepseekCompressor")
                    if attn.compress_ratio > 1 else attn.compressor is None,
                indexer=(type(attn.indexer).__name__ == "DeepseekV4Indexer")
                    if attn.compress_ratio == 4 else attn.indexer is None,
                bmm=attn.wo_a.is_bmm and attn.wo_a.bmm_batch_size == attn.n_local_groups)
            checks["sink"] = (list(attn.attn_sink.shape) == [attn.padded_heads]
                              and attn.attn_sink.dtype == torch.float32)
            if attn.indexer is not None:
                ix = attn.indexer
                checks["indexer_state"] = (ix.topk_indices_buffer is topk
                    and ix.eager_scratch_pool is pool and ix.scale_fmt == "ue8m0"
                    and ix.quant_block_size == 128 and not ix.use_fp4_kv
                    and ix.k_cache.head_dim == ix.head_dim + ix.head_dim // ix.quant_block_size * 4
                    and ix.indexer_op.k_cache is ix.k_cache)
        rows.append({"name": f"model.layers.{index}.attn", "role": "v4_attention_state",
                     "valid": all(checks.values()), "checks": checks})
    return rows


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
                if ((type(owner).__name__ == "DeepseekV4DecoderLayer"
                     and field in ("hc_attn_scale", "hc_ffn_scale"))
                        or (type(owner).__name__ == "DeepseekV4Model" and field == "hc_head_scale")):
                    row["scale_role"] = "mhc_native_coefficients"
                    row["scale_valid"] = (tensor.dtype == torch.float32 and row["finite"]
                                          and row["shape"] == ([1] if field == "hc_head_scale" else [3]))
                elif (type(method).__name__ == "Fp8MoEMethod"
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
                    expected = list(owner.weight.shape[:-2]) + [math.ceil(size / block)
                        for size, block in zip(owner.weight.shape[-2:], owner.weight_block_size)]
                    is_bmm = getattr(owner, "is_bmm", False)
                    layout_valid = (owner.weight.ndim == 3 and owner.weight.shape[0]
                        == owner.bmm_batch_size) if is_bmm else owner.weight.ndim == 2
                    row["scale_valid"] = (tensor.dtype == torch.float32 and row["finite"]
                        and owner.weight.dtype == torch.float8_e4m3fn
                        and layout_valid and row["shape"] == expected and row["positive"] and bool((values == torch.ldexp(
                            torch.ones_like(values), exponents - 1)).all().item()))
                elif (type(method).__name__ == "Mxfp4MoEMethod"
                      and method.mxfp4_backend.name == "MARLIN"
                      and field in ("w13_weight_scale", "w2_weight_scale")):
                    row["scale_role"] = "marlin_mxfp4_e8m0_multiplier"
                    row["scale_valid"] = (tensor.dtype == torch.float8_e8m0fnu
                        and row["finite"] and row["positive"] and tensor.is_contiguous())
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
    tensor_rows = {row["name"]: row for row in tensors}
    for name, module in modules.items():
        method = getattr(module, "quant_method", None)
        if type(module).__name__ == "DeepseekV4FlashMLAAttention":
            backends[name] = {"attention": type(module).__name__,
                "backend": module.backend_cls.__module__ + "." + module.backend_cls.__name__,
                "swa_backend": module.swa_backend_cls.__module__ + "." + module.swa_backend_cls.__name__,
                "cache_dtype": module.kv_cache_dtype, "einsum_recipe": module._einsum_recipe,
                "tma_aligned_scales": module._tma_aligned_scales}
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
        if type(method).__name__ == "Mxfp4MoEMethod":
            backends[name] = {"quant_method": type(method).__name__,
                              "backend": method.mxfp4_backend.name,
                              "experts": method.experts_cls.__name__ if method.experts_cls else None}
            inventory = {"name": name, "role": "mxfp4_quantization_inventory", "valid": False}
            if (method.mxfp4_backend.name == "MARLIN"
                    and all(hasattr(method, field) for field in
                            ("num_experts", "hidden_size", "intermediate_size", "moe_kernel", "moe_quant_config"))):
                e, n, k = method.num_experts, method.intermediate_size, method.hidden_size
                kernel, quant = method.moe_kernel, method.moe_quant_config
                experts = kernel.fused_experts if kernel is not None else None
                checks = {"dimensions": e > 0 and n > 0 and k > 0 and n % 32 == k % 32 == 0,
                          "consumer": type(experts).__name__ == "MarlinExperts"
                              and type(experts) is method.experts_cls
                              and experts.quant_config is quant and experts.input_dtype is None,
                          "weight_only": quant is not None and quant.use_mxfp4_w4a16
                              and quant.a1_scale is None and quant.a2_scale is None}
                for weight_field, scale_field, size_n, size_k, scale in (
                    ("w13_weight", "w13_weight_scale", 2 * n, k, getattr(quant, "w1_scale", None)),
                    ("w2_weight", "w2_weight_scale", k, n, getattr(quant, "w2_scale", None)),
                ):
                    weight, scales = getattr(module, weight_field, None), getattr(module, scale_field, None)
                    scale_row = tensor_rows.get(f"{name}.{scale_field}" if name else scale_field)
                    checks[weight_field] = (isinstance(weight, torch.Tensor)
                        and weight.dtype == torch.int32 and weight.is_contiguous()
                        and list(weight.shape) == [e, size_k // 16, size_n * 2])
                    checks[scale_field] = (isinstance(scales, torch.Tensor)
                        and scale_row is not None and scale_row["scale_valid"]
                        and list(scales.shape) == [e, size_k // 32, size_n]
                        and checks[weight_field] and scales.device == weight.device and scale is scales)
                    if scale_row is not None:
                        scale_row["scale_valid"] = checks[scale_field]
                inventory.update(valid=all(checks.values()), checks=checks,
                                 dimensions={"experts": e, "intermediate": n, "hidden": k})
                backends[name].update(prepare_finalize=type(kernel.prepare_finalize).__name__
                    if kernel is not None else None, input_dtype=str(getattr(experts, "input_dtype", None)))
            else:
                inventory["reason"] = "Backend-specific packed weights and scales are unverified"
                for field in ("w13_weight_scale", "w2_weight_scale"):
                    row = tensor_rows.get(f"{name}.{field}" if name else field)
                    if row is not None:
                        row["scale_valid"] = False
            auxiliary.append(inventory)
    if type(model).__name__ == "DeepseekV4ForCausalLM":
        auxiliary.extend(_v4_attention_inventory(model, config, modules))
        core = model.model
        hf = config.model_config.hf_config
        expected = [i for i in range(core.start_layer, core.end_layer)
                    if i < hf.num_hash_layers]
        tables = []
        for i in range(core.start_layer, core.end_layer):
            moe = core.layers[i].ffn
            tensor = moe.gate.tid2eid
            if tensor is not None:
                valid = (i in expected and isinstance(tensor, torch.Tensor)
                         and list(tensor.shape) == [hf.vocab_size, hf.num_experts_per_tok]
                         and tensor.dtype == (torch.int64 if moe.use_mega_moe else torch.int32)
                         and bool(((tensor >= 0) & (tensor < hf.n_routed_experts)).all().item()))
                tables.append({"name": f"model.layers.{i}.ffn.gate.tid2eid", "valid": valid})
        auxiliary.append({"name": "model.layers", "role": "v4_hash_tables",
                          "expected_count": len(expected), "observed_count": len(tables),
                          "valid": len(tables) == len(expected) and all(row["valid"] for row in tables),
                          "tables": tables})
        if core.start_layer == 0 and core.start_layer < core.end_layer:
            layer = core.layers[core.start_layer]
            tensor = layer.hc_attn_fn_broadcast
            shape = [(2 + layer.hc_mult) * layer.hc_mult, layer.hidden_size]
            valid = (isinstance(tensor, torch.Tensor) and list(tensor.shape) == shape
                     and tensor.dtype == layer.hc_attn_fn.dtype == torch.float32
                     and tensor.device == layer.hc_attn_fn.device
                     and not tensor.requires_grad and tensor.grad_fn is None
                     and bool(torch.isfinite(tensor).all().item()))
            if valid:
                valid = torch.equal(tensor, layer.hc_attn_fn.detach().view(
                    -1, layer.hc_mult, layer.hidden_size).sum(dim=1))
            auxiliary.append({"name": f"model.layers.{core.start_layer}.hc_attn_fn_broadcast",
                              "role": "mhc_first_layer_broadcast", "valid": valid,
                              "shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                              "dtype": str(tensor.dtype) if isinstance(tensor, torch.Tensor) else None})
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
        result["nvidia_smi_L"] = subprocess.run(
            ["nvidia-smi", "-L"], check=True, text=True, stdout=subprocess.PIPE).stdout
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
