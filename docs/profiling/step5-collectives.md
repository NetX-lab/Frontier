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
`NCCL_*` transport settings. The launcher only adds NCCL INFO/INIT,NET logging.
The SDK's RDMA resource defaults are retained. Durable outputs are confined to
`/mnt/codesign-exp/ycfeng/frontier/step5_comm/<run_id>/`.

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

## Build S1 and S2 after S0

Do not submit S0's one-node package with a larger replica count. Multi-node
torchrun rendezvous and node rank assignment must be added to the worker using
the existing native launch patterns, then dry-run and record all node commands.

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
