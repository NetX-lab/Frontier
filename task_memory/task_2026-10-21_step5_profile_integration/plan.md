# Integration plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Isolate the full profiling merge for independent review and verification. |

Dependency: provisional merge -> {whole-branch source review, affected CPU tests,77-case fidelity} -> parent integration and documentation sync.

The provisional merge is f0ab2eb, combining parent b0018c4 and profiling5fa1926. Scope37 files, including earlier attention/linear implementation and the14-file D18 correction. Review additional files against current requirements, not only the latest three commits. Require unchanged legacy outputs/cache identities and affected test verdicts. Reuse the freshly completed clean b1f032d baseline; b0018c4 changes only public documentation. Keep new candidate/results/report in a separate output root. GPU timing and the unresolved CPU performance budget are separate gates.
