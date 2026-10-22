# Controlled T2 inputs and execution proposal

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Root accepts constructor/CPU values and rebinds full integrated files at662849e; run approval remains pending. |
| 2026-10-23 | Prepare existing-schema inputs and exact commands; no run/resource approval or execution. |
| 2026-10-23 | Bind retained serving triple through both pre-construction paths; host flow PASS, root integration/image check pending. |

TaskID: D29-T2-ENGINE-LIMITS, following D29-T2-INPUT-PREPARATION.
Status: **PROPOSAL ONLY / NOT RUN**.
Root accepts this controlled-input/code bundle and still requires actual run/resource
authorization after remaining implementation readiness. Only root-directed
two-file limit binding is added; no new numerical state tolerance, mode or harness.

## Identity and controlled inputs

- Integrated code snapshot:662849e99223c98b9ea0d47d34783c3977cb3282, including
  native a5e5189/4a7adb7 and the selected helper/test from PRICING3f54bc6.
  Root verified selected-function AST equality,177 PASS/15 SKIP, actual installed
  image config values and independent No findings. The full integrated script/helper
  hashes in provenance.json replace the prepared mixed-file identity. The frozen
  source worktree is `/data/ycfeng/Frontier/.worktrees/verify-step5-t2-limits-01`.
  Do not change its code without producing a new binding and acceptance record.
- Accepted linux/amd64 image:
  `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`.
- Actual source export: `/data/ycfeng/tmp/d29_published_sources01/source`.
  It is retained provenance, **not** a worker PYTHONPATH/mount overlay.
- Installed native root: `/usr/local/lib/python3.12/dist-packages/vllm`.
- Unchanged complete metadata copy: `model_metadata/config.json`,78447 bytes,
  SHA256 `e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63`.
  Origin is retained root `reference_inputs/config.json`, official model
  `stepfun-ai/Step-5-Preview` revision
  `364b3b03fed2477aa0953fe855a2c4c8c48ee3cc`. Full outer/text/vision/quantization
  exclusions are preserved. T2 uses dummy loading, language_model_only and
  skip_tokenizer_init; no checkpoint, tokenizer or network download is needed.
- `cases.json` SHA256:
   `ea35310017b9057a305266a90988db72ab81aaeed0fcf5457ada401e8ec60002`.
  This is **controlled probe input**, not observed native/Frontier traces.
  Construction/provenance and source/file hashes are in `provenance.json`.

True mask entries denote padding. All physical counts are positive even when
live count is zero. Padding positions are deliberately stale, not real tokens.

| Input | DP0 physical/live | DP1 physical/live | Purpose |
| --- | --- | --- | --- |
| source6128+dummy |6129/6128|1/0| `[0,6128)` plus masked7167; idle DP masked43; includes real cycle edge43/44 |
| tail16+dummy |1/0|17/16| DP1 `[6128,6144)` plus masked0; opposite idle DP masked6144 |
| q1 padded4+empty DP |4/1|4/0| DP0 real6144, masked43/44/7167; DP1 all padding6147/0/43/44 |

Graph replays both have17 physical rows. Replay0 has real6128..6143 plus masked0;
replay1 has real6144..6147 plus masked stale6128..6140. The same addresses receive
new positions/masks. The existing script additionally captures and replays an
all-padding dummy in each mode before assertions/baseline; no extra dummy case
or mode is added. Routing rule remains `(position*top_k+slot)%num_experts`.
E352/K8, configured sigmoid, normalization and scale3.0 come from retained
metadata plus the existing Step5Config default sigmoid; no hardcoded replacement
formula/backend is introduced into production code.

**Required engine limits:** cases.json now supplies max_model_len7168,
max_num_batched_tokens8192 and max_num_seqs128 from retained
`/data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/engine.json`, SHA256
`7947f58bf37ab0256756faf1ec0613f4dc92101e94ad4a7f596c50ae4a11655e`.
Native fused_moe/layer.py:274,361 derives MoE max_num_tokens from the scheduler's
max_num_batched_tokens; positions alone cannot supply this capacity. Full metadata
max_position_embeddings1048576 remains unchanged. T2 passes the same required
mapping before helper/module and direct graph constructors, then records the final
post-with_hf_config values in rank_N.json.engine_limits.modules/graph. A mismatch
stays diagnostic-only UNVERIFIED; original construction/native errors propagate.
Host entrypoint/helper flow PASS, including final-value drift. Root actual-image
CPU config construction also yields7168/8192/128 for both final objects. This is
constructor-value acceptance, not GPU defaults, graph/backend execution or full
serving parity. Original G3/state criteria remain unchanged.

## Minimum topology and proposed ceiling

The existing T2.main fixes TP8/DP2/gpus_per_node8. Existing
`expected_topology`/`validate_launcher` require WORLD_SIZE16, LOCAL_WORLD_SIZE8,
two full nodes and node-local TP groups. Ranks0..7 are DP0/node0,8..15 DP1/node1;
DP group pairs are `[j,j+8]`, EP covers0..15, PP1. Therefore the minimum existing
script topology is **2 nodes ×8 H800 =16 GPUs**, not a1-GPU or8-GPU proof.
Step auto backend excludes mega_moe, so the inspected Step SP predicate is false;
no speculative_config is supplied, so MTP metadata is retained but MTP execution
is not activated. Do not force a backend, SP, MTP or alternative graph mode.

Root resource proposal: codesign, non-preemptible16 H800,32 CPUs/128GiB host RAM
per node, one attempt, no automatic retry,1800s maximum allocation wallclock:
**8 GPU-hours maximum**. CPU/RAM/time are proposed reserves, not measured minima
or fit/runtime guarantees. Root may place this finite unit inside its later
approved finite-collection allocation instead of submitting a separate job.
No authorization follows from this estimate or earlier32-GPU serving approval.

Six module calls/rank (2 native layers ×3 frames),96 across16 ranks. Thin graph
uses17 rows and2 live replays/mode for NONE/PW/FULL, plus existing dummy replay
and nested-context restoration. No full model/KV pool/reference service is loaded.

Source-derived raw weight lower bounds: EP16 gives22 complete experts/rank with
expert TP1 (do not divide them by TP8 again). H4096/I1536 and3 matrices' dimensions
give415236096 bytes FP8 routed weights,830472192 bytes BF16,101376 bytes FP32
block scales for FP8 before any rounding/format conversion. At maximum6130
physical gathered rows, existing expected native schemas imply26285440 bytes FP8
dispatch,50609280 bytes BF16 dispatch,50216960 bytes BF16 combine payload.
These are dimension calculations, not measured allocator/NIC/backend facts.
Peak also includes shared/gate weights, assignment overlap while layer88 loads,
cached allocations, native workspace/activations, NCCL/DeepGEMM/JIT and graph pools;
actual memory fit remains UNVERIFIED. See `resource_proposal.json` for formulas.

## Mount and import setup (root supplies later)

Use the published image above, not a mutable tag/local index digest. Mount this
accepted source worktree at its absolute local path, read-only, on both workers.
Root has bound native a5e5189, inputs and the integrated helper at662849e; actual
allocation mounts must preserve that binding. T2 sets existing routing_input_file=None to
preserve its controlled plan, not substitute observed profiling inputs.
The repository mount includes existing script/helper and controlled inputs. It
does not contain a top-level vllm package. Do not put source exports,
native_source snapshots, extra_site, a local conda or inherited PYTHONPATH first.

Root chooses a fresh local writable temporary directory mounted at `/work/t2`
on each worker, with neutral cwd `/work/t2/neutral`, output `/work/t2/output`, and
worker-local caches. Output rank filenames are disjoint; if per-node mounts are
separate, root retains all16 reports in its own selected durable evidence path.
No cloud path is read/written or created by this preparation. Platform/cloud
mounts and retention follow root's existing rules, without creating a new launcher.
Preserve all platform NCCL_* variables and existing rendezvous/NIC settings.

## Exact existing-script invocation (NOT EXECUTED)

Run the common shell setup on **both** approved workers before any Python/Torch
process starts. The driver branch is the existing run_on_worker.sh:19–26 recipe;
the wrapper itself is not invoked because it changes cwd and performs unrelated
dependency preparation. Existing T2/helper imports require only image native
dependencies plus the repository namespace, not profiler pandas/sklearn extras.

```bash
set -euo pipefail
REPO=/data/ycfeng/Frontier/.worktrees/verify-step5-t2-limits-01
INPUTS="$REPO/task_memory/task_2026-10-22_step5_d29_native_balanced/t2_preparation"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic VLLM_MOE_SKIP_PADDING=1
unset VLLM_TARGET_DEVICE VLLM_USE_BREAKABLE_CUDAGRAPH
export PYTHONPATH="$REPO"
export TMPDIR=/work/t2/tmp XDG_CACHE_HOME=/work/t2/cache
export TRITON_CACHE_DIR=/work/t2/cache/triton TORCHINDUCTOR_CACHE_DIR=/work/t2/cache/torchinductor
mkdir -p /work/t2/neutral /work/t2/output "$TMPDIR" "$XDG_CACHE_HOME"
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
if (( ${driver_version%%.*} < 580 )); then
  export VLLM_ENABLE_CUDA_COMPATIBILITY=1
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export VLLM_ENABLE_CUDA_COMPATIBILITY=0
fi
cd /work/t2/neutral
```

Root must ensure no inherited unsupported MoE/backend/compile/SP/MTP overrides
exist. Breakable Step graphs are auto-enabled by existing config; do not force
them to bypass a failing default. R570 uses the compat13-0 directory at **process
start**; setting only the vLLM switch after Python starts is known insufficient.
On R580+ do not prepend that older compatibility package. Actual mapped libcuda,
host driver, installed native origin and effective configuration are root runtime
evidence still required; the existing script's source-origin report does not
attest the image digest or all library mappings by itself.

Root supplies MASTER_ADDR (reachable node0 address), MASTER_PORT (unused port)
and T2_RUN_ID (new finite-unit rendezvous ID), identically on both nodes. They
are unresolved allocation inputs, not guessed values. Then execute exactly:

```bash
: "${MASTER_ADDR:?root must supply node0 address}"
: "${MASTER_PORT:?root must supply unused rendezvous port}"
: "${T2_RUN_ID:?root must supply sealed finite-unit ID}"
python3 -B -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$T2_RUN_ID" --max-restarts=0 \
  "$REPO/tests/integration/step5_environment/test_position_cyclic_graph.py" \
  --model-path "$INPUTS/model_metadata" --cases "$INPUTS/cases.json" \
  --installed-root /usr/local/lib/python3.12/dist-packages/vllm --output /work/t2/output
```

Node1 command is identical except `--node-rank=1`. This is the existing torchrun
entry point and T2 script, not a generated runner. Root's approved controller
enforces the1800s allocation ceiling and retains terminal status; no new timeout,
stop command, RJobBackend instantiation or cancellation policy is introduced here.

## Qualification and remaining prerequisites

Expected future reports:16 `rank_N.json` files with final modules/graph engine
limits7168/8192/128, real native FP8 layer3/BF16 layer88
module calls under explicit positions/masks, actual backend/expert/collective
observations, and thin graph NONE/PW/FULL address/counter checks. Any diagnostic
failure stays UNVERIFIED under the existing soft boundary; native errors remain
native errors. Even a future PASS_THIN_GRAPH_AND_EAGER_MODULES is **not** MRv2
serving inputs, attention/connector breaks, real backend graphs, G3/state/transfer,
original-router semantics, numerical fitting or op-gap/CPU overhead qualification.

Unfinished: remaining implementation readiness; explicit
resource/run approval and finite-collection coordination; real node/mount/output/
rendezvous/retention identities; actual
worker source/import/library/config/backend/memory/runtime receipts. No runtime
receipt is generated, no numerical state tolerance is selected, and no GPU action
is authorized or performed by this document.
