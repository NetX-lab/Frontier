"""H4a-2 probe: run a tiny V4 PDD case on the branch code, without patches; measured mode binds the step-cost table."""
import csv, runpy, sys
from pathlib import Path

mode = sys.argv[1]  # dummy | measured
root = Path(sys.argv[2]); root.mkdir(parents=True, exist_ok=True)
with (root / "requests.csv").open("w", newline="") as s:
    w = csv.DictWriter(s, fieldnames=("arrived_at", "num_prefill_tokens", "num_decode_tokens"))
    w.writeheader()
    for i in range(3):
        w.writerow(dict(arrived_at=float(i), num_prefill_tokens=6144, num_decode_tokens=4))
args = ["--simulation_mode", "offline", "--sys_arch", "pd-disaggregation", "--no-enable_parallel_clusters",
    "--log_level", "warning", "--decode_cuda_graph_mode", "none", "--cc_backend_config_type", "analytical"]
for role, n in (("prefill", 1), ("decode", 1)):
    p = f"--cluster_config_{role}_"
    args += [f"{p}cluster_num_replicas", str(n)]
    for k, v in (("num_pipeline_stages", 1), ("attn_tensor_parallel_size", 8), ("moe_tensor_parallel_size", 1),
                 ("moe_expert_parallel_size", 16), ("total_expert_num", 256), ("router_topk", 6), ("device", "h800")):
        args += [f"{p}replica_config_{k}", str(v)]
args += ["--replica_config_model_name", "deepseek-ai/DeepSeek-V4-Flash", "--replica_config_network_device", "h800_dgx",
    "--replica_config_attn_dp", "2", "--replica_config_moe_routing_distribution_type", "balanced",
    "--replica_config_moe_routing_seed", "42", "--replica_config_attention_backend", "FLASHMLA_SPARSE_DSV4",
    "--replica_config_kv_cache_dtype", "fp8", "--replica_config_indexer_kv_cache_dtype", "fp8",
    "--replica_scheduler_config_type", "vllm_v1", "--vllm_v1_scheduler_config_batch_size_cap", "16",
    "--vllm_v1_scheduler_config_block_size", "256", "--vllm_v1_scheduler_config_num_blocks", "54740",
    "--vllm_v1_scheduler_config_num_blocks_mode", "explicit", "--vllm_v1_scheduler_config_max_model_len", "7169",
    "--vllm_v1_scheduler_config_max_tokens_in_batch", "8192",
    "--analytical_kv_cache_transfer_config_kv_connector", "nixl",
    "--request_generator_config_type", "trace_replay", "--trace_request_generator_config_trace_file", str(root / "requests.csv"),
    "--trace_request_generator_config_max_tokens", "7169", "--seed", "42",
    "--metrics_config_output_dir", str(root / "metrics"), "--metrics_config_cache_dir", str(root / "cache"),
    "--metrics_config_run_id", f"probe_{mode}", "--metrics_config_write_metrics", "--metrics_config_store_request_metrics",
    "--metrics_config_enable_metrics_ground_truth_trace",
    "--no-metrics_config_store_plots", "--no-metrics_config_enable_chrome_trace", "--no-metrics_config_write_json_trace"]
if mode == "dummy":
    args += ["--random_forrest_execution_time_predictor_config_enable_dummy_mode",
             "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "0.01"]
sys.argv = ["frontier.main", *args]
runpy.run_module("frontier.main", run_name="__main__")
