# Root profile02 diagnostic acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Bound full root reproduction, cloud archival, independent review and the open qualification checks. |

Root reran every original per-trace command from `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/commands.json`, changing only the output destination to `/data/ycfeng/tmp/glm53_dsv4/profile02_root_rerun_nakiix_p`. Python3.10.6, `PYTHONDONTWRITEBYTECODE=1`,60-second timeout and30-GiB system scope; at most two traces concurrently. Adapter SHA256 `7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8` matched before and after. Exact argv, times, output hashes and comparisons are in `root_verification.json`.

Expected:32exit0 and exact receipt equality apart from elapsed/RSS and raw-index output paths/compressed-file headers. Actual:32/32PASS,326,328 matched annotations,7,600,008 kernels. Every decompressed kernel-index byte stream also matched independently. Per-lane useful-step counts23/2/61/61 remain unchanged; this reproduces the insufficient prefillDP1 coverage rather than making it pass.

Independent review found no audit correctness defect. Its two upstream documentation findings are closed in the analysis plan/current progress. The full53,441,663-byte audit and raw-index evidence were archived as46 hash-verified files/86,211,102bytes under the personal cloud path in `archive_receipt.json`. Initial archive creation failed on a root-owned parent; the original error is retained there. Privileged creation of the fresh analysis sibling then passed; original run files and permissions were unchanged.

`primary_acceptance.json` accepts this diagnostic audit only.192missing CUPTI stream mappings, captured per-op ownership, cross-DP identity and GPU dependency evidence remain unresolved. Shutdown evidence fails for decodeDP1 only; unaligned wall-clock timestamps do not prove cross-host order. Native06R4 remains the accepted reference. No new source/GPU run, serving cost, CPU-gap assignment or E2E gate change is included.
