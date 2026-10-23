# Main08 residual decode-wave diagnostic report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record bounded actual wave RED, isolated variables, profile and matching limits. |

Task ID: **D29-MAIN08-RESIDUAL-REPRO**. Python3.10.6 in frontier-py310,
frozen source79f1a3fd35d42b751cd1520c764ddac717b923d9, submodule
3dac53c544e546574cb6e8078cd14053d0916113. No pytest or full CLI was run.
Every command is bounded by60 seconds and scripts set RLIMIT_AS30GiB.
The requested systemd user MemoryMax scope failed; logs retain the failure.

## Actual symptom and bounded correspondence

Main08 offline execution.json records null returncode and timeout2400.175252s.
The scheduler is259555 bytes and has52 records; four completed DECODE batches
span214.734s,221.392s,220.158s,224.602s. Iteration4 begins but has no batch end.
No online artifact/process was inspected. The1.448GB trace was not loaded or
completely scanned: only4MiB head and8MiB tail samples were parsed.

Final loop source is the actual tail frame: DECODE layer61, source7052/global8/
group4, interval6147+1, peer7051/global9 FULL dummy live0, physical[1,1],
gathered2, all EP0..15. Real materializer, public stage predictor, phase accessors,
EP metadata emission and TraceStore flush run unchanged. Fixture paired pickles
are verified with their existing compatible bundle SHA; source byte hashes and
exact script hashes are in scratch/source_probe_receipt.json.

The existing test predictor construction seam avoids full initialization. It
still uses inherited production methods; final config/role/model/backend,
forward-start marker, family tables and two-stream classifier come from actual
CLI/construction facts. CPU overhead remains the existing skipped scope; no
attention/event-queue replay is included. All output/cache paths are relocated
before CLI parse. Pickles are read-only and registered through the real registry;
no lock/cache write is performed against retained fixture cache.

## Final deterministic real-path RED command

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-red02 > /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-red02.log 2>&1
```

Observed **exit1 / RED**, not an exception: cold/warm pricing3.387136/2.734742s,
full wave3.597119/2.926180s.208 events per wave. Cold/warm JSON parses28004/28003;
artifact reads80/80. The deterministic RED count signal is repeated unchanged
artifact reads plus>20000 real parses per wave. This is a diagnostic signal,
not a new product gate or evidence of original2400s completion. Timing alone
was the initial preliminary signal; the final script uses stable work counts.
Final counts end at flush completion, before diagnostic payload-hash verification;
the earlier actual02/final-red01 results included128 extra verification decodes.
Those earlier artifacts stay retained but do not define the final wave counts.
Fresh reruns must use a new output path (`mkdir(exist_ok=False)` protects records).

## Profile command and decisive source sites

Same environment/command with output `wave-actual-profile01` and `--profile`.
Profiled values are not uninstrumented timings; parser-site SHA accounting adds
overhead. Two waves: wave phase pricing8.591s, query recovery6.447s, real
json.loads3.103s, actual artifact reads1.452s, collective pricing1.584s,
trace flush0.832s. Group resolution occurs once only (the previous fix remains).

| Actual site | Evidence across two profiled waves |
| --- | --- |
| moe_routing_workload.py:156–160 | Routing input parse28390 calls,1027 unique texts; context/source checks stay live. |
| moe_routing_workload.py:168 | CORE reads and parses full collective artifact each lane. |
| moe_routing_workload.py:177–190 | Plan-key parse21504 calls,168 unique texts,163257088 bytes decoded;168 candidate keys rescanned per phase/lane. |
| collective_sim_cc_backend.py:812–817 |64 calls/wave still read exact artifact text; conditional JSON parsing fix works. |
| expert_parallel.py:109–139 |16 physical participant predictor calls per wave, not one source-only call. |
| sklearn_moe_execution_time_predictor.py:228–238 |Actual query/binding/proof/pricing entrypoint reached for each participant. |
| trace_store.py:43–46,119–121 |Recursive dataclass conversion dominates captured trace serialization. |

## Ranked hypotheses and isolated tests

Presented before variable tests:
1. Repeated row/plan text decoding: exact same-text decode reuse should reduce
   parsing work without changing proofs/full plans.
2. Repeated artifact I/O: retaining identical text should remove reads while
   keeping parsing/selection/validation live.
3. Trace conversion copying: avoiding extra recursive copy for captured
   JSON-only metadata should preserve byte output and reduce flush cost.
4. Actual estimator/numerical work dominates: if1/2 are disproved or small,
   residual profile should show estimator cost. Current evidence rejects it as
   the primary cost at this wave, not as universally irrelevant.

Each wave test changes one variable through a scratch-only runtime interceptor:

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-read02 --diagnostic-variable artifact-read > /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-read02.log 2>&1
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-json02 --diagnostic-variable exact-json-decode > /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-json02.log 2>&1
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-plan02 --diagnostic-variable plan-key-decode > /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-plan02.log 2>&1
```

| Variable | Cold/warm pricing seconds | Warm parses/reads | Payload |
| --- | --- | --- | --- |
| Original actual binding |3.387 /2.735 |28003 /80 |b2593e… |
| Same artifact text only |2.581 /1.887 |28003 /0 |identical |
| Exact same JSON text decode only |1.922 /1.321 |0 /80 |identical |
| CORE plan-key text decode only |2.806 /2.188 |17251 /80 |identical |

Full paired cost/proof/plan payload SHA256 is
`b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a`.
The same-text diagnostic interceptor is not a production/global-cache proposal:
it proves repetitive work only; bounded ownership, mutation and all actual
point/family/source checks require a separately authorized implementation.
No variables were combined and no cached predictions bypassed validation.

## Trace replay and byte preservation

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/trace_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/trace-baseline01 --profile
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/trace_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/trace-shallow01 --profile --diagnostic-shallow-fields > /data/ycfeng/tmp/d29_main08_residual_repro/trace-shallow01.log 2>&1
```

598 actual complete DECODE events,8353308 bytes per iteration. Original
profiled1.202172/1.174562s,453008 deepcopy calls, exit1 count-RED. Diagnostic
shallow fields0.081105/0.075974s,0 deepcopy; serialized bytes exactly identical,
SHA256 `3ea6239cea9295e5d532722200c78238c9e01ecb58b32383fe8c8aaa20b7e004`.
This sample contains JSON primitives. The differential does not qualify
arbitrary nested dataclass metadata, alias/mutation isolation, or production
trace changes. Full metadata and serialization identity were retained.

## Actual artifact match

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_main08_residual_repro/compare_actual_wave.py --replay wave-final-red02 > /data/ycfeng/tmp/d29_main08_residual_repro/wave-final-red02-comparison.log 2>&1
```

**PASS /exit0**,165 captured layer61(ep,name) pairs compared to208 replay pairs.
Durations, model/count proof hashes, full native context/source, collective
plan and cost equal;0 mismatches.43 replay pairs are not in the bounded tail
sample. Batch/request entity IDs are intentionally new diagnostic entities;
actual routing source IDs and all numerical point identity stay retained.

## Retained failed setup probes and verification limits

- wave-baseline01: argv slice error before config construction; not a bug RED.
- wave-baseline02: stale bundle SHA rejection before pricing; no cache bypass.
- wave-baseline03/original01/read01/json01: inherited test two-stream setup
  chose KERNEL_ONLY. Actual comparison found CUDA_EVENT model/proof mismatch.
  Those timings are superseded. Final actual02/final-red01 and variable tests
  use the corrected actual setup and PASS the artifact comparison.
- wave-actual01: missing loaded kernel-only family before public selection;
  corrected by loading both actual verified family bundles, not a fallback.
- Requested systemd user scope failed; timeout60s and RLIMIT_AS30GiB are used,
  which are not a resident-memory cgroup guarantee.
- No source/test/retained data edits, full CLI, exhaustive CSV/full1.448GB trace
  scan, GPU/network/platform or online access. No full timeout PASS is inferred.
