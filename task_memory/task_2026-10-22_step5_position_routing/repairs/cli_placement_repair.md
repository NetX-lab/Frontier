# D29 actual-main placement repair

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record both final-code CLI failures and begin a real-constructor short-feedback reproduction before choosing a repair. |

## Requirement and observed failure

Existing D29 acceptance requires full6144/1024 offline and online CPU-fixture
execution within2400s per mode,including the request/output and proof checks.

Frozen source1be91bad,fixture03 and main05 produced two failures:

| Mode | Wall seconds | Return code | Timed out |
| --- | ---: | ---: | --- |
| offline | 1580.9479549312964 | 1 | false |
| online | 1560.1816781787202 | 1 | false |

Both logs end with `ValueError: Step plans require explicitly configured TP8/DP2 placement`.
The actual call chain reaches the position operator consumer and
`collective_sim_cc_backend.py:809` from `moe_operator_times.py:145` during the
first prefill EP wave. The runner's final assertion is a consequence,not the cause.
Raw source/runner/fixture identities and exact commands remain in
`/data/ycfeng/tmp/d29-parent-position-main05/{offline,online}/execution.json`
and `command.json`; mode logs and summary are preserved unchanged.

## Diagnosis frontier

D29-CLI-PLACEMENT-REPRO builds a seconds-scale command using the real CLI
configuration and actual backend factory/consumer boundary. It must reproduce
this precise failure without the expensive full training/request run or manually
substituting a convenient backend placement. The scout reproducer takes1.5s.
Parent runs the original command in1.609s (both roles RED) and one-variable global
placement probe in1.749s (both roles fully priced at0.03ms with matching keys).
Receipt:../final_parent/cli_placement_parent_repro.json. Scripts and logs:
/data/ycfeng/tmp/opencode-v2-tmp/opencode/d29_cli_placement_repro/.

Parent presented3 hypotheses before its probes: missing selected-model placement;
independent dimension mismatch; native-path qualification still failing after the
placement correction. Results support the first: real dimensions already equal
TP8/CP1/DP2/EP1,while placement is the genericTP,CP,DP,EP default. Passing only the
existing global `--collective_sim_cc_backend_config_placement_order TP,DP` yields
actual CPU plan costs,not merely passage through the guard. No error-as-green
result from the scout's permissive diagnostic script was accepted by parent.

Root assigns the actual-main runner's position-only command branch and its existing
integration test to CORE. Delta<=65 gross inside CORE4404,via the zero-sum
cli_placement_scope_amendment.json. Emit the already-approved finite model's
explicit placement alongside its existing profiling-data selector. Keep global
defaults,other modes,Cluster/factory/backend behavior and the placement gate intact.
The regression must exercise actual runner command generation and real config/
Cluster/backend construction with an exact CPU plan cost,not only inspect argv.

## Evidence incident and deferred scope

The scout's first CLI parse rewrote one retained main05 offline metrics config.json
before output redirection was added. Its original bytes were not captured,so that
file is no longer original-run evidence. Exact affected path:
/data/ycfeng/tmp/d29-parent-position-main05/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json.
Raw command,execution,run logs and inputs
remain available. Subsequent and parent probes redirect metrics into scratch.
Do not silently restore or claim original config byte identity.

Role-scoped placement overrides were also observed to be inert. The approved
shared-placement path works through the existing global option,so correcting that
independent legacy override mechanism is not required here; defer it in future.md.

## Acceptance and boundaries

Keep the placement check,topology,physical source/collective accounting,typed
identities and per-mode2400s bound. No fallback,fabricated placement or timeout
increase. The real constructor regression must turn green and independent review
must close before the next final code freeze and full actual-main replay.

Strict77 and the explicit cross-environment unit comparison passed on1be91bad;
they do not close this workflow failure. Hold main code integration and final T2
staging/seal until this blocker is resolved. T2's static63-checksum/26-file result
remains tied to1be91bad and requires alignment to any later repaired source.
