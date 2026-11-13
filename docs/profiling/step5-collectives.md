# Step-5-Preview collective measurements

The producer is `frontier.profiling.collectives.vllm_collective_benchmark`, not a
second benchmark implementation. It uses the locked Step image's Python, vLLM
GroupCoordinator, NCCL and other native all-reduce backends. The pure shape
module selects Step's non-sequence-parallel path from model metadata and the
native MoE backend predicate. The existing DeepSeek-family path is unchanged.

## Prepare S0 without GPUs

Run from the repository root with the CPU test environment:

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python \
  tests/integration/step5_comm/prepare.py \
  --stage-dir "$PWD/task_memory/step5_comm/s0_20261020_01" \
  --run-id s0_20261020_01
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python \
  -m frontier.profiling.collectives.vllm_collective_benchmark \
  --config "$PWD/task_memory/step5_comm/s0_20261020_01/benchmark.json" --list-cases
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/step5_comm/submit.py \
  --config "$PWD/task_memory/step5_comm/s0_20261020_01/job.json" --dry-run
```

Commit the source before preparing a submission package. The package records
source hashes and the commit; do not modify its source or reuse an attempted
package. Preparation and `--dry-run` construct no platform client. S0 has 10
scenarios, 100 callsite rows, 1 shuffled pass, 5 repeats and 5 calls per repeat.
It uses one node of eight H800 GPUs, TP4/DP2/EP8, for an estimated 20 minutes
(2.67 GPU-hours, excluding queue time). It is diagnostic, not target-topology
cost qualification.

Only the parent agent submits. Initialize personal `i-fengyicheng` auth and
`STEPMIND_BACKEND=rjob` according to the host handbook, then run with the
StepMind Python and keep that local launcher alive until terminal status:
Restore the repository working directory after sourcing an auth helper that
changes directories.

```bash
"$STEPMIND_PYTHON" -u tests/integration/step5_comm/submit.py \
  --config "$PWD/task_memory/step5_comm/s0_20261020_01/job.json"
```

The worker uses the image interpreter, sets the established CUDA 13 forward
compatibility environment, and installs no packages. It preserves platform
`NCCL_*` transport settings. The launcher only adds NCCL INFO/INIT,NET,GRAPH logging.
The SDK's RDMA resource defaults are retained. Durable outputs are confined to
`/mnt/codesign-exp/ycfeng/frontier/step5_comm/<run_id>/`.
The platform log prints `STEP5_COMM_WORKER_LOG` before workload stdout/stderr
are redirected to that cloud file; the full workload log is not the platform
log. The manifest labels the PyTorch NCCL version separately from each native
PyNccl communicator's version in `collectives/plan.json`. They can differ in
the same pinned image.

## S0 result checks

Platform success alone is insufficient. Inspect creator, pinned image, local
NFS source mount, GPU identities, container exit and restart count. Retain the
worker log and manifest, `collectives/{plan.json,measurements.jsonl,kernels.jsonl,
summary.csv,forward_totals.csv,all_reduce.csv}`, and the native dispatch probe
report. Expect 100 measurement and kernel rows, 10 totals, and explicit dtypes,
columns, group ranks, useful/padded tokens, communicator, timing domain and
concurrent rails. Native device DP groups are `[k,k+4]` on S0; the isolation
scenario uses only `[0,4]`. Host DP sync uses rail 0 and is kept out of device
totals. Idle ranks retain dummy or graph-padded participation but no logits.

The native probe must observe the unmodified Step router/expert path for one
FP8 and one excluded BF16 layer. Its actual gathered tensor schema, including
FP8 scales, is the authority: expected rows are 4288 and 8256 bytes. A mismatch
requires correcting the source-derived shape model before S1, not relabeling
the probe. Native probe timings are not cost rows.

For device timings require `queue_held_fraction=1`, or show a median change
below 1% after dropping unheld repeats. Unknown communicators remain
unverified. S0 TP4 communicator thresholds are not the target TP8 thresholds.
One-node S0 cannot demonstrate inter-node IB/GDRDMA or the target eight rails.

The sleep is calibrated from the maximum measured CPU enqueue window across
the participating group, with the existing 2x margin. The held predicate checks
that the CUDA begin event is still pending after every launch and the end event
have been enqueued. Raw measurements retain per-rank launch/queued enqueue
times and the sleep budget. Do not substitute GPU elapsed time for CPU enqueue
time or include the alignment collective's host cost in the held predicate.

The original `s0_20261020_01` run completed on eight H800 GPUs, and all sixteen
native layer/rank schema comparisons matched. It exposed the old clock-window
and calibration defects; use a new package/run ID for the corrected collector.
The original run's flags do not contain enough diagnostics to reconstruct the
correct predicate. CPU regression checks do not replace a GPU recheck of it.

The parent submitted corrected `s0_20261020_02` as `exp-1020-213124-403435`.
All 100 rows/groups/rails and all sixteen native rank/layer schemas passed.
All 90 device cells, 450 repeat positions and 3470 rank-repeat flags held;
the full-versus-held median difference is zero. Device sample CV median fell
from 9.29% to 4.43%, but five short repeat averages still do not qualify a
low-noise cost dataset. The worst cell has 52.05% CV despite every repeat
holding. No further timing-algorithm change or new threshold was applied.

## Prepare S1 and S2 after S0

Do not submit an old S0 package with a larger replica count. Prepare the
stage definitions with the multi-node worker and stage-supplied native probe:

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python tests/integration/step5_comm/prepare.py \
  --definition tests/integration/step5_comm/s1_run.json \
  --stage-dir "$PWD/task_memory/step5_comm/s1_new" --run-id s1_new
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python tests/integration/step5_comm/prepare.py \
  --definition tests/integration/step5_comm/s2_run.json \
  --stage-dir "$PWD/task_memory/step5_comm/s2_new" --run-id s2_new
bash "$PWD/task_memory/step5_comm/s1_new/submit_with_env.sh"
```

Only the parent runs the last command. Each generated wrapper loads the
credential environment before the submitter, restores the repository directory
and uses `exec` to keep the blocking launcher alive. Do not detach or terminate
it. Use a new ID for every attempt and recheck quota before submission.

The worker consumes platform-produced `NODE_RANK`, `NODE_COUNT`, `PROC_PER_NODE`
and `MASTER_ADDR`, requires eight processes per node, and uses distinct
benchmark/probe ports. All nodes use the same static torchrun rendezvous;
`--standalone` is used only for a one-node stage. Native failures retain their
exit status. Node lifecycle receipts coordinate phase completion through the
mounted local workspace, not an auxiliary quality verdict.

Each node owns `node_<rank>/worker.log`, `manifest.json`, source/transport
preflight and phase-exit receipts. Node zero alone creates the common root,
collective outputs, `transport.json`, `validation.json` and `result.json`.
The probe writes rank-specific diagnostics and only global rank zero writes
`native_dispatch/dispatch_probe.json`; it checks stage TP/DP/EP groups and
host placement, without substituting TP ranks for request-owner DP lanes.

Transport verification retains libibverbs device visibility, active port/link
type and actual inter-node `Channel ... via NET/IB/.../GDRDMA` lines from every
node. Merely finding `NET/IB` initialization is insufficient. Bootstrap sockets
are not data-plane fallback; selected Socket channels leave evidence unverified.
These are node-level channel observations, not inferred attribution of each
interleaved line to one DP group. Native DP group identity is checked separately.

All auxiliary source, inventory, artifact and timing validators are advisory:
exceptions or negative verdicts mark outputs unverified and print
`STEP5_COMM_UNVERIFIED` to the platform log, without aborting healthy native
phases. Native probe observation/diagnostic failures must likewise not suppress
the underlying native call. This is mandatory for 32-GPU S1, and is also used
for S2. Platform success alone never admits cost data.

| Setting | S0 | S1 | S2 |
| --- | --- | --- | --- |
| H800 nodes / GPUs | 1 / 8 | 4 / 32 | 2 / 16 |
| TP / DP / EP | 4 / 2 / 8 | 8 / 4 / 32 | 8 / 2 / 16 |
| Native device DP groups | `[k,k+4]` | `[k,k+8,k+16,k+24]` | `[k,k+8]` |
| Concurrent device DP rails | 4 | 8 | 8 |
| Host sync bytes | 48 | 96 | 48 |
| Local logits row bytes | 64448 | 32224 | 32224 |
| Repeats / passes | 5 / 1 | 30 / 3 | 30 / 3 |
| Expected wall / GPU-hours | 20 min / 2.67 | 25 min / 13.33 | 12 min / 3.2 |

S1 uses the full preregistered grid: graph batches
`1,2,4,8,16,32,64,128,256,512`; uniform eager tokens
`16,32,64,65,128,256,512,1024,2048,4096,8192,8193`; for each of
`T=256,512,2048,8192`, uneven patterns `[T,1,1,1]`, `[T,T,1,1]`,
`[T,T,T,1]`, `[T,T/2,T/4,1]`, `[T,64,64,64]`; graph-idle batches
`8,64,512`; isolated-rail graph batches `8,64,512` and eager sizes `512,4096`.
These are 50 scenarios and 500 callsite cases per pass (1500 rows for 3 passes),
not the design's approximate 300 distinct-payload rows. Five all-reduce
callsites share the same payload; their multiplicities sum to 273. Keep exact
dry-run counts authoritative and revise the wall estimate after S0.

Kernel profiling runs once per callsite, not once per pass: expect 500 kernel
rows for S1 and 460 for S2, versus 1500 and 1380 measurement/summary rows.
The wall times above are planning budgets, not measured multi-node runtimes.

S2 keeps TP8 and all payload boundaries, adapting each explicit scenario to
two DP owners. Use the distinct uneven patterns `[T,1]`, `[T,T/2]`,
`[T,T/4]`, `[T,64]` at the four eager sizes (16 uneven scenarios rather than
20). This gives 46 scenarios, 460 cases/pass and 1380 rows. Use a new run ID
and cloud directory for every stage. Keep hard anti-affinity, RDMA defaults,
the image digest, enabled-backend defaults, and native custom-all-reduce flag.

S1 must show NET/IB rather than NET/Socket on DP groups. TP8 communicator
boundaries at 64/65 and 8192/8193 tokens are source-derived expectations to
check against kernel names, never a substitute for observations. Independent
held-out shapes and native TP8/DP4 placement qualification remain required
before ticket 12 acceptance. The full grid's 8193 point is a deliberate
communicator-boundary probe, not proof that production uses that batch size.
Graph capture sizes and serving limits still need reconciliation with ticket
06. Any auxiliary S1 validator must be a non-blocking soft gate at 32 GPUs;
it must mark evidence unverified without stopping healthy measurements.

The optional S1R repeats S1 on a different host set. Ticket 13 owns Step NIXL
and shared-network measurements. The ported NIXL/overlap tools are retained,
but S0 does not run them. Simulator cost-table integration belongs to ticket
21; this preparation creates no measured `data/profiling/network/h800_dgx/`
dataset and makes no prediction-accuracy claim.

## Offline judging and candidate costs

Analyze local copies, not a live worker or an unmounted cloud path. The tools
accept `collectives/measurements.jsonl.gz` as well as uncompressed JSONL and
never access the platform. Keep large GPU traces in cloud storage.

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python \
  tests/integration/step5_comm/analyze_result.py "$RUN" --output-dir "$RUN/analysis"
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python \
  tests/integration/step5_comm/export_costs.py "$RUN" --output-dir "$RUN/costs" \
  --environment-note "Parent reports a nonexclusive cluster; causal effect unknown."
```

`observations.csv` preserves every pass/scenario/call, the critical-rank repeat
averages, queue verdict, physical and useful token vectors, typed payloads,
group placement, active rails and observed kernel/backend labels. `curves.csv`
adds unfiltered descriptive aggregates without combining graph modes,
backends, rail sets, uneven patterns or physical sizes. Only identical TP
all-reduce contexts merge their five call labels. Their unique multiplicities
sum to 273, not 819 over three passes. Each pass remains visible. A batch
average, another label and another pass are not independent native forwards.

Times are in **milliseconds**, matching `BaseCCBackend.predict_comm_cost`.
`api_operation` uses `allreduce`, `allgather` or `reduce_scatter` only as an
endpoint-byte projection. Uniform AR/AG use per-rank input bytes; RS uses the
full input, not its output shard. TP has eight participants; S1 DP collectives
have four, not EP32. Uneven/idle vectors and host/Gloo have no lossless scalar
projection. A scalar size does not encode the typed tensor list, graph mode,
backend or eight concurrent rails. Endpoint bytes are not measured wire traffic.

The samples are the maximum across participating ranks of the concurrent role,
including its multiple TP8 groups; they are not independent isolated TP groups.
The exporter retains CV, repeat range, all per-pass statistics and every failed
observation. No outlier is removed, no row is selected for training, and
`production_consumable` and `cost_data_qualified` are always false. Its manifest
records source hashes and reported environmental context. Output must not
overwrite the copied source worker manifest.

The legacy collector's `all_reduce.csv` already supplies the eager Vidur byte/ms
schema. Neither it nor these candidate sidecars is an admitted H800 dataset.
The current Vidur loader cannot represent the required context: its
`devices_per_node == world_size` filter excludes a real 8-GPU/node world of
32, and its precomputed element keys stop at 100000. All-gather/reduce-scatter
also lack measured curve consumers. Ticket 21 owns the consumption changes;
do not copy candidates into production profiling paths to bypass those limits.

The first S1/S2 jobs at `307467b` completed all containers, payload/group checks
and native module probes, with actual RoCE NET/IB/GDRDMA channels and no selected
Socket channels. Their strict queue check rejects 9/1350 and 15/1242 device
cells respectively, leaving the auxiliary result unverified without stopping
healthy work. The parent reports cluster co-tenancy; there is no isolated
baseline to quantify its effect. Stable curves, independent held-out shapes,
per-DP-communicator transport attribution and cost admission remain open. Native
module/group observations do not establish full-model/runner or PD load-balancer
behavior for tickets 06, 08 or 20.
