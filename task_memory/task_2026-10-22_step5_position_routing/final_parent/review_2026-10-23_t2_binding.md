# Independent T2 source-binding verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Accept pointer/source-identity corrections after parent63-checksum/26-file verification; physical staging remains pending CLI completion. |
| 2026-10-23 | Persist read-only independent verification and assign two target-document corrections to the native owner. |

TaskID D29-T2-PARENT-BINDING-CHECK; reviewer session
ses_e99139b55ffef68CRUUdbMqLtN. Delivery43c2947c/2adbf9f2 in
`/data/ycfeng/Frontier/.worktrees/step5-d29-t2-final`; source identity60ba4c75.
No suite rerun, file edit, network or GPU operation by the verifier.

## Observed checks

- `git diff --name-only 60ba4c75 2adbf9f2`:11 record paths only; source diff empty.
- In t2_preparation, `sha256sum -c hashes.sha256`:63/63 OK. Index SHA256
  a4bed10cd7b0189153f9bf5d1c1886dafd6d64fff4867612adf9cf9d1dd20e15.
- Git blob comparison of26 bound files at60ba,ca135d89 and disk:all equal.
  Twenty repository-relative hash-index entries also equal at both source targets
  and2adbf9f2. The set includes script/supporting files, publication and parent
  receipts, native4 snapshots, cases and official model metadata.
- Rehash actual published local source inventory:2907 files,exact path set,zero
  hash differences;12 backend modules,53 serving pins and32 native mechanism
  sources agree with disk and receipts. Native4 derived bytes and archive agree.
- Published index and platform payload hashes reproduce10cdd7af… andf037d958…;
  archive9,585,732 bytes matches receipt. No remote operation was performed.
- Cases/model metadata unchanged,positions below7168; supplied7168/8192/128
  agree with retained engine.json. Native MoE capacity still derives from scheduler
  max_num_batched_tokens. Native1350/1582 is unchanged.

## Controlled environment instruction

execution_proposal.md unsets inherited FRONTIER_STEP5_PROFILE_MANIFEST before
Python. With no routing_input_file, the controlled T2 observer does not install
runtime_policy; an inherited profiler manifest would otherwise enter the collector
branch and record observer_errors. The unset retains the original operation path,
route/weight/padding assertions and source checks. It supplies no profiling or
runtime qualification. This is a controlled-input proposal, not an approved run.

## Non-blocking documentation corrections

1. Proposal mount/REPO pointers name integrated-02 while the new sealed packet is
   in step5-d29-t2-final. Bind the actual intended staged tree and verify its26
   source bytes. Do not claim amended records already exist in integrated-02.
2. Explain85 incremental gross versus81 retained-baseline growth from the exact
   original two-file diff; keep native1350/1582 distinct from a step's gross delta.

Assigned to original native owner as D29-T2-TARGET-POINTERS,record-only. The
intended staging tree is integrated-04; code review/final freeze remain pending.
Root may integrate the verified records, then repeat final-target byte comparison
and index checks. No blocking source finding; final package acceptance is still
pending those pointer corrections and actual staging. The16 H800/two-node/
  30-minute proposal remains unapproved.

## Final target correction and parent verification

Owner delivery8c57df44/174e58e0/2c0f9bd7 resolves both record issues. Runtime
identity is now1be91bad431d3bd886de7f5d80c19bd7e6c54a25 with Git tree
a5cb27c3e26943f0dfa638301b3564fb2427d56d;60ba is retained only as history.
The record explicitly identifies the production quantization_manager change
outside the26 bound files,so it does not call the final revision records-only.
All staging/mount/REPO pointers name integrated-04 as the intended future target,
with updated_records_staged=false. Native step arithmetic is83 added+2 deleted=85
incremental gross;83-2=81 retained-baseline growth. Native total remains1350/1582.

Parent directly verifies final Git tree and changed-production SHA,all26 complete
files against both final Git blobs and actual integrated-04 disk,all script hashes,
engine limits7168/8192/128,and all63 entries with `sha256sum -c hashes.sha256` in
the isolated packet directory. Only11 native task-record paths differ from60ba;
there is no source/test edit. Index SHA256:
1e36d25e3fe8c41a4f2e7e04f2df7d352fd9b474d5aa34c3bda6bc92de5c97ca.

PASS_STATIC_BINDING_PENDING_STAGING. Receipt:t2_final_target_receipt.json;
log:/data/ycfeng/tmp/d29_t2_final_parent02.log. The active CLI tree stays immutable.
After CPU runs finish,root will stage the delivered records there and repeat
the index/source checks before package seal. No GPU launch is authorized.
