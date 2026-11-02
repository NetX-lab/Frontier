# Profile02 shutdown observation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the source/evidence investigation and separated receipt failure from an unproved cross-host ordering inference. |

Investigator: `ses_e96938101ffe4dtAkwixeNCe1s`; root checked all four downloaded `result.json` files. Frozen worker source is `05952396`. Evidence root is `/data/ycfeng/tmp/glm53_dsv4/profile02_results_readonly_7ba56n6z/artifacts`.

Only decodeDP1/node0266 records `head_stopped_before_headless_stop=false` and `Head shutdown marker missing; native stop order unverified`. PrefillDP1/node0213 records true and no observation error. Both heads retain the four-node drain quorum; all four workload and stop return codes are0.

Retained UTC wall-clock timestamps: decode head drained14:56:06, began shutdown14:58:07 and wrote its stop marker14:58:23; decode headless drained14:56:06 and wrote its stop marker14:57:50. These timestamps are consistent with an inversion, but cross-host clock alignment and receipt visibility were not measured. The accepted finding is failed shutdown-order evidence, not a proved cross-host happens-before relation. The earlier two-node report and definitive inversion wording were too strong.

`native_pd_worker.py:369–373` writes drain then polls a directory-glob quorum. `:387–401` waits for the head marker using `stop_timeout_s=90`, then stops the headless server even after timeout. The head writes its marker only after `stop_server` returns; that routine can spend90seconds on SIGINT and another90 on SIGTERM (`vllm_replay.py:606–615`). The same90-second marker budget cannot cover every supported head-stop path.

The head's roughly121-second pre-shutdown interval lacks internal boundary timestamps. Delayed directory enumeration/receipt visibility and process/IO scheduling remain hypotheses; this run does not distinguish them. No filesystem-cache change is justified by the current evidence.

Next diagnostic step: reproduce the too-short marker budget through the existing worker lifecycle test seam, using delayed quorum/head-stop completion; compare the existing peer-wait budget and explicit known-peer paths. Record bounded lifecycle timing if needed. Preserve auxiliary soft failure and original native stop escalation. No source or package repair is included in this receipt.
