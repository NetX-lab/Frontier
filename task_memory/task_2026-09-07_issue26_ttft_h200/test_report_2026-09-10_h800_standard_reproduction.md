## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded successful H800 standard first-forward reproduction. |

# H800 standard replay reproduction

Full script paths, commands, environment, criteria and evidence: [run report](analysis/h800-standard-replay-176000-repro-01/report.md).

PASS: RJob Succeeded, clean400/batch400completed clients,3x100drained warmups per mode,100formal requests per mode,8worker identity validator PASS. Fresh DP0 batch4711 median77.483665466ms,P90 77.486233521ms,max77.486846924ms,spread0.026206970ms. Historical H800 median80.654880524ms,max80.687454224ms,spread0.091041565ms. Median difference-3.171215057ms(-3.931833%). Shutdown TCPStore warnings retained; all completion gates pass. This closes standard forward-scale reproduction only.
