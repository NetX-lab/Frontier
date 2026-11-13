# D23 native startup census correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record the observed startup contamination and scope of the CPU observer correction. |

## Original request and inherited authority

[Original Request] Continue Step P0 implementation and calibration under D20/D21. Diagnose observed failures before making in-scope fixes. G3/G4 capture01 has completed; its native judgment and all original evidence remain immutable.

The original MRv2 census includes 192 source-defined startup calls (six per physical rank) as scheduled useful work. Business request counts reconcile, but the original census gate is FAIL. Correct the existing observation boundary so future captures distinguish native compilation/kernel warmup from serving execution, without filtering business requests by their names or changing native execution.

Acceptance: reproduce contamination through the real observation wrappers/native host timing seam; exclude startup at its existing native lifecycle boundary; preserve subsequent useful and idle-DP forwards, exceptions, exactly-once native calls and soft observation failure behavior. Original native FAILs remain unchanged. This CPU correction alone does not qualify G3, profiles, clean timing or accuracy.

No GPU/platform action or package submission belongs to this slice. The source-defined retained-record census remains an explicitly diagnostic projection, not a replacement native verdict.
