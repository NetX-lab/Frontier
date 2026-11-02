## Modification History

| Date (UTC) | Change |
| --- | --- |
| 2026-10-24 | Reviewed frozen runtime evidence and completed primary audit. |

Reviewer: Codex, independent runtime-evidence reviewer. Phase: S5, `exp-1025-021228-212042`; primary owns GPU/platform/acceptance.

Paths are scope-relative; `task:` means task root, `repo:` means checkout. Hashes identify inspected snapshots.

| Artifact | SHA256 |
| --- | --- |
| `repo:AGENTS.md` | `82354337fec8131194f31ad8f09d298d2ef6e3ad48282b0b4bd46e87b5a57249` |
| `task:requirements.md` | `c229e702243f2093e323aa37af75f2b7e44feb2eace9a6c19868d3ea563b4c3f` |
| `task:plan.md` | `46d8b409189c02c19858132c6c50d6f8a38db8327c2f369b073173b5fe979591` |
| `task:progress.md` | `d2c2d6054973b9f698b1aec0485c17689343f4658eb9a4cb5ed94b87e8a06278` |
| `requirements.md` | `faedecc3a642c5a8d487981d389b836e536f929938a35c81866bed1917bbd209` |
| `plan.md` | `9ce664a8abe2a6f2fd032bc78e9dbfa053f3c01e1f9b2c4a707c124ab057ebdd` |
| `progress.md` | `d7ebe8761ca10ee813b03050bd7194126f2b550c80d35625c90c6a623fb32606` |
| `scope_lifetime_design.md` | `e8469d17ea4ffed882523ebcc50f2614dafd411554b019bb3d3713099bf7b3bd` |
| `probe_scope_lifetime.py` | `bc638fbdf0111a8ce78383d182d41e19a5799bacc6cc5a657aa79b1a34672685` |
| `source_binding.json` | `89232a9e1ee60e5d57859dd707796a85b3d99ba89ad309d98aab3c93b2ed6b0b` |
| `primary_acceptance.json` | `f639ab3351d436e32a6e2233f0aea5a18f989bb3927e1292a1ff062a7b1562bc` |
| `../graph_probe/probe_diagnostic.py` | `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0` |
| `repo:tests/comparison/calibration/native_serve_profile.py` | `c8f712177fcd3e8ef6fbbfe2ceb62a30e06d298d432da48bd7790ed214669f58` |
| `h800_scope_lifetime_probe_01/manifest.json` | `9ff1abec5fa2451baa2a285bdf081011eba9a2741448ad39a64b9a27aae4198e` |
| `results/cloud_inventory.json` | `9eb11fbd97ecce24b0b7c7e41219fd2c10be4a8a6c4757abd7f630778cf99809` |
| `results/download_receipt.json` | `3be010ccd8cb5664f8b1802a18ef4443e9e6e593cca2fdb50095de730d6f38c4` |
| `results/raw/diagnostics/stages.jsonl` | `ee426997af0349e1510d3113525f1e6700781da4c536385e64744136aca6818e` |
| `results/raw/diagnostics/fault.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `results/raw/probe/probe_report.json` | `e988ee758ae7e08a52ecd996277b8b574f7113511f58d3ef9d4ddad77812e586` |
| `results/raw/probe/observe_trace.json` | `fe8c03c857ec412fbb5dd02335b2454e4ac3b25d3fa88b72d504c0786a6fbee8` |
| `results/raw/probe/control_trace.json` | `a3815882be30a2ac83b50577b508e74cb5211cc06e66fb81327cf60d5c8f32bc` |
| `package_primary/execution/platform_02.json` | `f036a63b0f2dad146c58a9d2c5ce8c1ceb558ba0305c90d8bb2a5a7a06c67416` |
| `package_primary/execution/platform_02.exp-1025-021228-212042-cfcd2084.log` | `fc18505c0803a545303d87e1d205da21e9390d80ba95d74b235a7a3980417d1c` |
| `package_primary/execution/launcher_01.raw.log` | `515f434a7b77af8970a55b17161dd4a7661bd0e062ac61ed2cabac85f406cc59` |
| `results/audit_allocation.json` | `8904fd340b7203c0afe2196854e97d6d8c51695330905efd4fb987b5ffdc9407` |
| `repo:tests/analysis/v4_scope_lifetime_audit.py` | `0cec6d93c28408c46c07078c40a5293ffd9974470428841bad368a0a0a1f7414` |
| `results/primary_raw_audit.json` | `f5a88af4f1afa68228e76d399b063fc35cd05e7702d048c50c42f17620c7b294` |
| `results/raw_audit_receipt.json` | `febd5a16084521928989be2ac67f55484900b9e83d31b27ff39c3b5c033a34e8` |
| `results/execution_source_binding.json` | `a8bcd4e22bb5816c1d244947040f37b19fdce6ae5759fe2aa641c88eb37cad6c` |

Verification: Python 3.10.6, `python3 -B` small-file checks 63/63 PASS; primary audit 53/53 PASS, execution binding 20/20 PASS. Five downloads and 18 frozen package inputs match hashes. Latest logs support Succeeded/exit0/restart0 and the recorded H800 UUID; allocation spent1/1.

- P1: Payload copying and node/graph/exec queries stay inside callbacks 13/20/18. Eleven created rows include five selected originals and six unmatched rows; their cause remains unproved.
- P2/P3: Native/Python thread equalities are observations. Freshness uses exactly one monotonic op window and exact scope equality for each of five sources.
- P4: Default `keep_graph=False`; one `raw_cuda_graph` expected-raise boolean per captured graph after capture, no source-handle dereference. Copied exec handle matches; close precedes Kineto.
- P5/P6: Five complete `created(node,fullID)`↔`clone(originalNode,originalNodeID)` joins transfer directly to five unique replay fullIDs under one target launch.
- Control proves matching five-kernel topology and exact outputs only. Preserve `control.verdict=UNVERIFIED`; no timing/non-perturbation.

Audit limitation: whole 142-line script reviewed. Lines 43, 61–64, 116–117 omit explicit `observe.errors`, `op.start/end` pairing, and successful unique close checks. Current raw records satisfy these independently. Primary acceptance must cite combined source/raw review, not the audit JSON alone.

Permitted conclusion: this sealed image/1024-element H800 probe supports the frozen predicates. Host driver release is `UNRECORDED`; API13000/CUPTI130001 are telemetry. Native serving, general ownership strategies, dependencies, costs and reference admission remain excluded; primary decides acceptance.
