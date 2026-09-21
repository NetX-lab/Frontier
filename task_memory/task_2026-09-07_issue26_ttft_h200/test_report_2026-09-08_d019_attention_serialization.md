## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded failed attention export, direct serializer check, and isolated continuation. |

# Attention export recovery

Execution: original exact command is preserved in analysis/h200-d019-profiles-01/launch.sh and runtime/environment.log. Python /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python,3.10.16,approvedH200image. The wrapper completed measurements but json.dumps failed with `TypeError: Object of type AttentionBackend is not JSON serializable` before samples.json was written; communication did not run.

Cause: existing AttentionWrapper.profile returns the configured AttentionBackend enum. The new export helper had not converted that metadata field. Repair: convert row["attention_backend"] via its value; timings and inference code unchanged.

Direct check: parse tests/performance/issue26_attention_exact_profile.py with Python3.10 AST grammar; execute its actual For node against a row containing an Enum and representative numeric timing; json.dumps/json.loads then assert FLASHINFER string. PASS. Shellsyntax for recoveryphaseworker PASS. FreshGPU acceptance requires three complete rows with expected4096/KV0/TP4 identity and positive event medians; PENDING in analysis/h200-d019-profiles-02/runtime/attention/samples.json.

The continuation command is in analysis/h200-d019-profiles-02/launch.sh, phase attention_communication. Existing MoE/linear measurements are retained and not rerun.


Fresh GPU continuation PASS: three complete FLASHINFER/4096/KV0/TP4 rows exported; prefill medians0.146176/0.145088/0.143424ms; KVsave0.012320/0.018400/0.012192ms. Parent job completed0; subsequent8rankcommunication completed. Full P/S evidence: analysis/d019-linear-attention-comparison.md.
