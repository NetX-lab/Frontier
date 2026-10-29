# Interval metrics verification — 2026-10-25

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Verified owner delivery, independent primary unit and one synthetic scale pair; archived exact decode control and prefill native SIGSEGV; source/control acceptance remains blocked. |

## Scope and frozen commands

Decision: `D-V4-P0-AUDIT-INTERVAL-REPRESENTATION-CORRECTION`. New source81/160 gross lines, tests243/260, owner source notes25/80; primary driver150/150. Source SHA256 `33fee81fbaad80f89f9c971acf81870c6c6ac78f8e1bbe0d81a6de361f1a17f6`; tests `642ab9a4c41c37a94dc3472401dd8fb312e81ada60a35c6a321747ffb601f00b`. Original accepted adapters, auditor, pipeline, reader, runtime, allocator environment and raw input bindings remain frozen.

Every fresh command is bound before execution in `primary/<mode>_allocation.json`:

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_correction03/primary/verify_interval.py unit
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_correction03/primary/verify_interval.py scale
```

Python3.10.6 (GCC11.3.0), `/usr/bin/python3`, no Conda; original selected allocator/GC environment equals the frozen allocation. Parent scopes each observe32212254720 bytes. Owner and primary unit and synthetic scale each use1/1; no automatic retry. The driver only imports the new/pinned adapters and AST-extracts pure frozen functions for unit/scale; no spent outer harness runs.

## Observed results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Owner delivery | Exact17 bound files,9 tests/90 internally consistent saved assertions, actual30GiB | All reproduced from retained files; owner invocation not rerun | PASS |
| Primary unit | Exact metrics including floathex/count/key-order, immutable inputs and original recurrence/composed AST outside metrics | Four fixtures;36/36 saved checks, exit0, internal0.168827s/enclosing0.241901s | PASS |
| Synthetic pair | Observed-scale counts and exact all group metric values | 1215919 events/240 groups/5331778 memberships;24/24 checks, exit0, enclosing27.758189s | PASS |

Synthetic accepted pure group function14.02723632287234s; candidate12.577456965576857s. Observed difference1.449779357295483s/10.335459700864391% for this one ordered pair. No repeated performance sample, raw trace or real P0 metric timing is inferred. Synthetic group distributions differ from the actual trace; quantities only match the measured scale.

## Pending complete controls and reviews

New decodeDP1rank0 and prefillDP1rank0 controls are separately frozen in `primary/control_decode_allocation.json` and `primary/control_prefill_allocation.json`; paths/hash equality prepared in `primary/control_bindings_preparation.json`. Decode control PASS:37saved checks,142567rows/10462613decompressed bytes/SHA4ea76cfb6c97cbf7f614dcf7165fa69f6cd0aafb1093caf65cbc842be47d93b0/cleanEOF, parent exit0/enclosing6.947991s. Prefill FAILED: child-11/SIGSEGV,parent1/enclosing20.366353s,actual parent/child30GiB;no finalJSON, no equality/EOF checks after returncode failure. Last of29 observations19.943kernel_progressidx1175000; heartbeat earlier samples JSON encoding and endsFatal Segmentation fault/noPythonframe. This index is partial/hash-only;no expected570964-count claim. Expected complete counts142567/570964 and useful61/4. Acceptance removes only `elapsed_seconds`, `maxrss_kib`, `kernel_sample_index.path`, `kernel_sample_index.sha256`; every remaining field/floathex and every decompressed index byte must match, with fresh excluded fields verified separately. Controls write fresh files under `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_interval_controls_20261025_01`. Each rank remains60s/actual30GiB/selected1/max2.

Independent Spec/Standards source reviews reportNo findings within source/evidence scope in `primary/review_spec.md` and `primary/review_standards.md`. Source acceptance remains pending. New affected/GPU allowance0; R5 remains24/32. Original native abort/censored diagnoses, old failed pilots, eight original timeouts and all spent controls remain preserved. Corruption writerUNKNOWN; no allocator cure, native ownership, dependency, timing/cost/reference or full R5/R6/R7 acceptance follows.

Native failure blockscomplete controls; new observation allocation `prefill_fault04/primary/allocation.json` does not grant a new control/affected/GPU run. `primary/control_archive.json` binds12smallbyte-exact copies and3largehash-only paths. Current core limit0, no core files found in checkout/thisscratch; `coredumpctl` unavailable. Native detection instruction/corruptionwriter remain UNKNOWN.
