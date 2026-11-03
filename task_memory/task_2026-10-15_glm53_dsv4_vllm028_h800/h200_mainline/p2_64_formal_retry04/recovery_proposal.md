# Fourth P2-64 submission decision

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | Prepared a verified package and concrete resource options after the lead-caused third-job stop; decision remains pending. |

## Evidence and responsibility

The lead stopped Pi labelp64-formal3-1103b to replace a broken monitor, mistakenly assuming the detached launcher was isolated. The stop script selects inherited DS4_SESSION_ID rather than a process group, so launcher907414 receivedSIGINT and stopped exp-1103-092042-217677 at2026-11-03T01:26:58Z. Platform stop audit and all eight replica owner/state/event chains agree. No replica was scheduled and no container/native request started. Inferred container usage is0GPUh, separately from the generic exact-null clock receipt. This interruption is the lead's error, not a platform preemption.

All three GPU submissions permitted by requirements D-P2-64-FORMAL-B/D-P2-64-STARTUP-RECOVERY have been used. Ordinary debug approval does not change this numerical cap. The next GPU submission needs YC's new decision.

## Prepared option and alternatives

| Option | Resources/cost | Outcome |
| --- | --- | --- |
| Recommended: authorize exactly one fourth submission |64H200=8x8,all preemptible,step_main,timeout3600s,ceiling64container GPUh | Obtain a new promotion-bound, pair-warmed reference; no further automatic retry |
| Defer resource allocation |0new GPUh | Retain prepared package and all evidence;P2-64formal remains INSUFFICIENT_EVIDENCE |
| Explicitly waive the pre-launch receipt requirement for the old October25 reference |0new GPUh | Requires a separate acceptance-policy change; old reference lacks pair coverage before all formal first contacts and retains paired exclusions; not recommended |

Package `package_01`, manifest `d75b0277a3c08592359002d256ac8115844f50fa051efe422815cc5b2d19f081`, has47verified sealed files and independent check-onlyPASS.41files equal generation03;only6identity/provenance files differ. Nativea06a72a4/harnessf33306d4,workload,topology,promotion receipt,R5and completed Frontier replay stay frozen. Wrapper `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry04.sh` passes bash-n and has not been executed. No new quota observation is represented as capacity reservation.24.3GPUh is a historical estimate, not a validated forecast for the pair-warmup candidate; the request is bounded by64GPUh.

## Execution after approval

1. Read current quota and submit this exact candidate once through ds4-exec/StepMind personal Python, preserving localNFS and required cloud mount. Keep the detached launcher alive throughout queueing and execution. The submitting Pi turn ends normally after a finite receipt checkpoint.
2. Monitor with a distinct Pi label, using get_rjob_infos plus exact replica objects and untruncated outputs. Never call ds4 --stop for a label owning a live StepMind launcher. An auxiliary monitor failure returns evidence and does not trigger workload control. No tracking-marker removal is needed.
3. Verify actual8nodes/64H200UUIDs/source hashes/API readiness/request progression. Auxiliary coverage/readers remain soft gates. No replacement GPU job is automatic if this fourth job fails.
4. On terminal: capture clocks/events immediately,copy and independently rehash produced artifacts,then pair coverage,R5,all-request and warm readings using the frozen completed Frontier replay. Keep every repetition and four metrics; report missing evidence without inventing a PASS.

The local CPU lifecycle check already verified that a detached child survives normal parent exit, retains its environment marker, and remains selectable by the stop mechanism. The operational fix is to preserve the launcher-owning label and separate monitoring; it changes no native serving behavior.
