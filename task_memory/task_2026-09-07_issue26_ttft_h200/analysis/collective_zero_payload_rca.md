## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the reproduced zero-payload contract failure, scoped correction, and runner cleanup/split analysis before repair. |

# Collective Zero-Payload RCA

## Observed Failure and Causal Chain

The fresh `runs/h200-fresh-frontier-02/runtime/frontier.log` reaches request 0's first decode iteration after its 4096-token prefill. The layer-0 EP conservation record reports one routing token, top-k 8, and routed assignments `{0: 8, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}`. Lane 0 completes prediction; the next participant fails with `htsim_runner failed`, return code 2, and `missing required fields: ['tensor_bytes']`.

`frontier/scheduler/utils/expert_parallel.py::predict_ep_wave_phase_times` predicts every EP participant, including empty lanes. `frontier/operators/families.py::_expert_parallel_payload_bytes` preserves the physical lane payload: hidden width 2048 times two bytes times zero routed tokens equals zero bytes. The failed participant is inferred to be EP lane 1 from the ordered participant loop and the preceding successful lane-0 record; the failed scenario's temporary file was not retained.

Both `CollectiveSimCCBackend._build_collective_spec` and `Scenario.to_runner_spec` explicitly serialize `tensor_bytes`, including zero. In `htsim_runner.py::_merge_spec_into_args`, this field incorrectly uses `set_if_none_or_zero`, which discards a spec zero and can replace an explicit CLI zero with a positive spec value. The required-field check then classifies zero together with missing values. This is parameter normalization failure, not a missing Frontier payload producer.

Source anchors before repair: Frontier backend lines 497 and 635; `python/collective_sim_core/schema.py` line 273; `htsim_runner.py` lines 1398, 1469, and 2349. Negative data sizes are rejected by Frontier's `BaseCCBackend._validate_data_size`, but direct runner arguments also need explicit nonnegative validation.

## Deterministic Reproduction

The read-only investigation used `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python 3.13.13, with `PYTHONDONTWRITEBYTECODE=1`. It constructed the real `collective_sim_core.schema.Scenario`, called its real `to_runner_spec()`, and passed serialized JSON to the actual runner with `subprocess.run([sys.executable, str(backend / 'htsim_runner.py'), '--spec', '/dev/stdin'], input=json.dumps(spec), text=True, capture_output=True, timeout=15)`.

The scenario used one server, eight GPUs, TP1/CP1/DP1/EP8, participant ranks 0 through 7, EP domain, placement order TP/CP/DP/EP, `exclude_intra_server=True`, and `alltoall_model='pairwise_steps'`. Its tensor bytes were zero, channels 8, chunk bytes 8000000, and in-flight chunks per peer 2. Intra-server settings were `nvlink_analytic`, one-way bandwidth 450 GB/s, efficiency 0.8, and latency 0.5 microseconds.

Observed results:

- The serialized scenario retained `collective.tensor_bytes=0`.
- Explicit zero and an otherwise identical runner spec with `tensor_bytes` removed both returned code 2 and the same missing-field error.
- The actual `estimate_intra_server_ms(scenario)` returned 0.0035 ms: zero transfer bytes, seven modeled steps, 0.5 microseconds per step, and zero all-to-all launch overhead.
- The reproduction stopped at runner argument validation and did not invoke the htsim binary.

Root subsequently reported a real-binary regression in `tests/unit/test_collective_sim_zero_payload.py`: 3 FAIL / 2 PASS in 1.17 seconds. The failures were explicit zero rejected as missing, negative payload incorrectly accepted, and CLI zero replaced by positive spec 32768. Positive-payload model behavior passed. This paragraph records root's reported result; the independent investigation did not rerun that test file.

## Preserved Semantics and Minimal Repair

EP all-to-all has `ZeroPayloadPolicy.PREDICT`, not `EXACT_NOOP`. `tests/unit/test_comm_operator_families.py::test_zero_payload_policy_preserves_other_collective_modeling` explicitly preserves this distinction. The current NVLink model charges its collective latency even when payload transfer bytes are zero. Returning total time zero, changing operator policy, or removing empty EP participants would change the established model and scheduler contracts.

The bounded correction is confined to the actual runner:

1. Merge `tensor_bytes` through the existing `set_if_none` helper, preserving spec zero and explicit CLI precedence.
2. Permit zero for this required field while continuing to reject an absent value; retain the existing requirements for other fields.
3. Reject negative tensor bytes explicitly before execution.

The one-server H200 scenario filters all participant pairs from network traffic because intra-server traffic is handled analytically. Positive payload already exercised this empty-network-flow path. A real zero-payload run must verify that the same path completes and that the core result retains its 0.0035 ms analytic contribution.

Required direct checks are explicit zero, a positive payload, missing payload, negative payload, and CLI-zero precedence over a positive spec. No timing calibration or bandwidth fitting is part of this repair.

## Cleanup and Functional Split Analysis

The root `htsim_runner.py` has 2691 lines before repair. The applicable root `AGENTS.md` Development Gates require cleanup first for a critical module above 2000 lines, then a concrete boundary explanation and functional split analysis when it remains above the soft limit.

Whole-submodule Python reference searches found `_as_str` and `_as_int_nonneg` only at their private definitions, with no callers. Removing these two unused converters is a safe local cleanup of approximately 16 lines. The existing validation helpers, active topology checks, and environment filtering have live callers and should remain intact.

The remaining size comes from several independent responsibilities: traffic construction beginning at `generate_tm_from_spec` (roughly 565 lines); topology/schedule generation; process launch and output parsing; and CLI/default normalization plus orchestration in `main` (roughly 627 lines). The current repair keeps those execution boundaries intact and changes only the meaning of explicit zero at the parameter boundary. This permits a small direct regression to establish the correction without changing topology generation, traffic algorithms, cache semantics, model latency, or scheduler participation.

Proposed future split, recorded for design review rather than implemented here:

`baseline real-CLI behavior -> topology module -> traffic module -> CLI normalization module -> retained root entry point`

- `python/collective_sim_core/topology.py`: topology and schedule validation/generation, preserving current generated file content and paths.
- `python/collective_sim_core/traffic.py`: collective traffic construction and its FlowDef/TriggerDef/CollectiveSpec dependencies; keep layout conventions explicit and avoid a cycle with schema/layout.
- `python/collective_sim_core/runner_cli.py`: argument parsing, spec/default precedence, validation, and orchestration. The root script remains the established external CLI entry point.

Each extraction should preserve real runner output and traffic-matrix behavior before the next module moves. A broader extraction is outside this zero-payload correction and has not been authorized or implemented by this investigation.

## Applied correction and observed verification

Root applied the scoped runner correction and removed the two unused helpers after recording the cleanup/split analysis.26 real backend and communication-policy tests passed in27.14s. Commits:Frontier69764e50,collective-sime564935. Fresh H200 generation03 was submitted03:06:35HK with new numerical caches and outputs; full-case result pending. See test_report_2026-09-08_collective_zero_payload.md.
