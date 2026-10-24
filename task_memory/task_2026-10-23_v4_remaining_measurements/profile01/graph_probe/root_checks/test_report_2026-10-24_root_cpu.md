# Graph payload correction: root source and CPU acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted source1aa1e4a5 after independent source reviews, root regression rerun and actual-image ABI/entry/bytecode checks. |

## Source and reviews

Source1aa1e4a5, owner recordsa2c4ad87; probe SHA256 `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`. Delta26/80gross lines; CPU regression controls159/160. Existing event/DAG ownership and strict9-node/5-kernel/4-event criteria remain unchanged. The correction dereferences the official ResourceData envelope, checks descriptor presence, then copies GraphData scalar fields and queries IDs in the original order.

Independent Standards (`ses_e93abff63ffd1uuwupd9kIy3Ew`) and Spec (`ses_e93abff63ffeaLmGwRc633Vskw`) reviews found no actionable defect. They inspected full source, diff, real allocated-envelope tests, official sample/header, old mapper and receipts, without executing tests or rehashing files. Root performed72 separate source/record/frozen-input/matrix/result checks, including37 historical byte pins, four referenced frozen inputs, exact Git bytes, and the complete historical77-case comparison. No new matrix capture was needed.

## Root executions

- Local Python3.12.13, `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, separate processes with MemoryMax30G/timeout60s and CUDA hidden:16 new cases PASS; two safe unchanged-source RED cases fail as expected;71 inherited cases PASS with71 actual candidate loader redirects. Five old bare-payload/ABI tests are explicitly replaced by the new envelope cases, not silently counted as passes.
- Initial root inherited invocation accidentally omitted `-k`; pytest treated the expression as a filename and ran0tests. The failure is retained in `local_cpu/`. Only the argv was corrected and the inherited71cases rerun (`corrected_inherited/`); source and tests were unchanged.
- Actual locked image2286e853, network disabled,8GiB/2CPU, Python3.12.3/GCC13.3.0: official header hash matches; the same five-line C++ ABI control passes; Python ctypes ABI matches; real `inspect-api` CLI executes through diagnostic initialization, imports and output creation, with all U1–U4 NOT_RUN and CUDA uninitialized. Exact commands/input/output are under `image/`.
- Actual image bytecode distinguishes old CALL offset516/line196 (node) and offset574/line197 (original node). The retained crash names line197. This identifies the source call site; it does not prove the original native crash cause.
- The initial evidence copy encountered root-owned0600 diagnostic files. Read-only `sudo cat` of those two exact files completed the byte-identical archive; original contents and permissions were preserved.
- Raw pytest RED log/XML contain assertion-output trailing spaces. The initial staged whitespace check reported them; their bytes are retained, with only these two raw files excluded from the documentation/script whitespace check.

## Acceptance boundary

`primary_acceptance.json`: ACCEPTED_CPU_GRAPH_PAYLOAD_CORRECTION. Actual resource callback delivery, captured-graph clone completeness, strict U1–U4 outcomes and the old crash cause remain UNVERIFIED. The independent package, actual SDK, current platform preflight, seal and one1H800/600s run are later gates. No native/C26/reference/operator/serving/E2E qualification follows.
