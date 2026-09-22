# Issue 26 Correctness PR — Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Landed the execution specification verbatim (Section "Execution Specification" below) and recorded the amendments agreed with the user before Step 0. |

## Amendments (authoritative where they differ from the specification below)

Facts were verified against `origin/main` at `1f694f7c549aa3aeeb7c5bbae04e119c09167a77`, `origin/bug/ttft-check` at `a7b3320fe9b8b083ee86b91dae3d6838f4443d91`, and the host `kun-workspace-vgen2` on 2026-09-21. Each row records the specification clause, the verified fact or user decision, and the resulting rule.

| # | Spec clause | Verified fact / decision | Amended rule |
| --- | --- | --- | --- |
| A1 | §3.1, §3.2 `docs/development/issue26-correctness-pr/` | User decision: task records live in `task_memory/` of the PR worktree; no new `docs/` tree. `task_memory/` is ignored by `.gitignore`, and Git cannot re-include a child of an excluded directory. | Records live in `task_memory/task_2026-09-21_issue26_correctness_pr/`. `.gitignore` uses `task_memory/*` plus a narrow `!task_memory/<task-dir>/` exception for this directory and the refactor directory in A3. Only Markdown and small text evidence are placed there; generated JSON/CSV stay in the scratch root. The CLAUDE.md files `requirements.md`, `plan.md`, `progress.md`, `summary.md` are kept; the specification's `review.md` and `validation.md` are supporting files in the same directory. |
| A2 | §5 execution order, §4.1 oversized modules | User decision (Q4=c, Q8=a, Q9=b, Q10=b): the four touched modules above 2,000 lines (`frontier/config/config.py` 5720, `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` 5138, `frontier/execution_time_predictor/shared_prediction_model_manager.py` 4614, `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` 3539) receive a full cleanup-first and functional split. | Stacked PRs. `refactor/oversized-module-split` branches from main and carries only behavior-preserving cleanup and splits. `fix/issue26-correctness-pr` branches from the refactor branch. Both draft PRs are opened; the correctness PR's base is the refactor branch and is retargeted to main after the refactor PR merges. Steps 2–7 of this specification start only after the refactor branch's fidelity matrix passes. The other five modules above 2,000 lines are out of scope. |
| A3 | §3.2 four documents | Decision Q12=b. | The refactor PR has its own records in `task_memory/task_2026-09-21_oversized_module_split/`. This plan references that directory instead of duplicating its content. |
| A4 | §6 baseline file `tests/unit/test_open_source_release_arch_guard.py` | The file does not exist on main (`git ls-files` returns nothing). `AGENTS.md` still references it; recorded as a deferred documentation drift, not fixed here. | Baseline selection: `tests/unit/test_cluster_scheduler_dp_lanes.py`, `tests/unit/test_colocation_release_review_contracts.py`, `tests/unit/test_config_owned_contracts.py`, `tests/unit/test_stage_execution_time.py`, `tests/unit/test_stage_finalized_contract.py`, `tests/unit/test_moe_routing_runtime.py`, plus the co-location and PDD dense offline example wrappers. |
| A5 | §6 "use the existing CPU environment" | No conda and no Frontier-capable Python environment existed on this host. `uv` is available at `/data/ycfeng/toolchain/bin/uv`. | Task environment: `/data/ycfeng/envs/frontier-py310` created with `uv venv --python 3.10` and `uv pip install -e ".[test]"` from the refactor worktree. `PYTHONPATH` points at the active worktree. Scratch root: `FRONTIER_TMP_ROOT=/data/ycfeng/tmp/issue26-correctness-pr`. |
| A6 | §7.2 separate read-only vLLM checkout | No local vLLM-BS checkout existed. User decision: clone under `.real-engine/`. | Reference checkout: `<repo>/.real-engine/vLLM-BS` detached at `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1`; `.real-engine/` is listed in `.git/info/exclude` (not in the tracked `.gitignore`). Read-only. |
| A7 | §13 Step 7 | `fwyc0573/frontier-htsim` exposes only `main` at `b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04`; commit `e564935d…` returns HTTP 422 (not found). No local clone containing it exists on this host (the historical calibration worktree is absent). User decision Q2=c: decide at Step 7. | Step 7 starts from these facts. Reconstruction in a companion branch requires the user's authorization at that point; otherwise the package is `EXCLUDED`. |
| A8 | §12 Step 6 native validation, §1 execution environment | `/data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md` exists and StepMind `RJobBackend` GPU workers are available. User rule: simulator runs stay on the local CPU; GPU workers are used only for the Step 6 native numerical check and for any additional profiling CSVs. | Unchanged Step 6 requirements; the GPU path is StepMind per the runbook. |
| A9 | §3.3 publication, §3.5 approvals | User authorized (Q5) pushing both branches to `origin` (`NetX-lab/Frontier`) and creating/updating the two draft PRs for the whole task. Not authorized: merge, force-push, history rewrite, closing Issue 26. | Checkpoint pushes need no further approval. Q6=a: this session stops after the Step 0 push for user review. |
| A10 | §2.1 candidate snapshot | Verified: merge-base is `d71ad80b…`; the final candidate commit `a7b3320` touches only `task_memory/` (1006 files, no source); the candidate gitlink is `e564935d…`; donor design documents exist at the archive top level; `tests/unit/test_moe_routing_runtime.py` already exists on main and is modified by the candidate. | No change to the specification; facts recorded for Step 1. |
| A11 | §7.1 candidate scope | The candidate adds ~80 `tests/e2e|integration|performance/issue26_*` experiment scripts. | Default disposition `DROP` unless Step 1 finds a specific reusable helper (the specification already names `tests/integration/issue26_dp_coordinator_reference.py` for review). |

## Execution Specification (verbatim copy of `.local-draft/Frontier_Issue26_Correctness_PR_Execution_Spec_2026-09-21.md`)

# Frontier Issue 26: Correctness Fixes for Main

## Execution specification for Claude Code

**Prepared:** 2026-09-21  
**Task type:** A new, independently reviewable pull request against `NetX-lab/Frontier:main`  
**Execution environment:** CPU master by default; narrowly scoped GPU-worker tests only when native kernel correctness requires them  
**Status:** Ready to begin implementation. This document records a source-reviewed plan, not completed implementation or newly executed tests.

> Start a new worktree and branch from freshly fetched `origin/main`. Extract and improve the justified fixes from `bug/ttft-check`; do not merge that branch wholesale. Read the relevant Frontier and vLLM 0.10.2 code before editing. Complete the steps below in order, publish code, tests, and task documentation after each meaningful checkpoint, and provide a clear user-facing report. Do not resume TTFT calibration or run a vLLM serving benchmark.

## 1. Objective and boundaries

Deliver a small, readable set of correctness fixes that stands on its own without a Frontier-versus-vLLM latency comparison. The source branch is evidence and a source of candidate changes, not an implementation to copy without review.

The intended fixes are:

| Work package | Intended outcome | Inclusion rule |
| --- | --- | --- |
| Round-robin DP placement | Preserve DP rotation across separate scheduling calls. | Core scope. |
| Shared monolithic forward execution | Allow prefill, decode, and mixed source batches to complete one shared EP forward without splitting synchronization identity or borrowing another source's timing. | Core scope; port as one coherent behavior change. |
| vLLM-style DP request placement | Offer an opt-in, explicitly bounded implementation of vLLM 0.10.2 request-count selection and delayed count publication. | Include after source and event-integration checks. Keep existing defaults. |
| Routing implementation identity | Keep expert-load distribution separate from the implementation used to predict routing cost; prevent model/cache sharing across incompatible implementations. | Core scope; carry the identity through the complete selection chain. |
| Legacy fused-MoE profiling | Execute the missing gated activation and local output reduction, with accurate measurement ownership. | Include after CPU checks and required native numerical checks; do not overwrite main's newer backend support. |
| Optional collective backend input handling | Distinguish zero payload from missing payload; reject negative input; preserve explicit CLI precedence. | Conditional on a reviewable, remotely fetchable submodule fix and real CPU runner tests. |

**Success is semantic correctness, regression safety, maintainability, and reviewability. A TTFT error threshold is not an acceptance criterion.**

### Explicitly excluded

Do not launch vLLM servers, replay the historical 100-request calibration workload, compare serving TTFT/TPOT/E2E results, or conduct Nsight/Kineto investigations. Do not retune collective latency, bandwidth, CPU overhead, or compute scaling. Do not add an empirical residual to make simulated latency match a measurement.

Do not implement a new physical communication execution model, change the default ideal EP accounting, adopt task-local profiling CSVs as official datasets, or force-add the old task archive. Do not rewrite unrelated scheduler, profiling, or training infrastructure. Do not change the vLLM reference to make a test pass. Do not merge the final PR or close Issue 26: this PR addresses selected correctness defects, not the entire calibration issue.

A complete **Frontier-only CPU simulation** is allowed and required for relevant lifecycle checks. That is different from a vLLM serving comparison.

## 2. Source snapshot and recovered context

### 2.1 Revisions reviewed when this specification was prepared

| Repository / reference | Reviewed revision | Role |
| --- | --- | --- |
| `NetX-lab/Frontier`, `main` | `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` | Integration baseline at specification time. |
| `NetX-lab/Frontier`, `bug/ttft-check` | `a7b3320fe9b8b083ee86b91dae3d6838f4443d91` | Candidate fixes and historical evidence. |
| Candidate's parent before the large evidence import | `b7f8d055461d9208b8ae57eceeb6c0246cbc8d3c` | Convenient source-code comparison point; verify rather than assume that the final commit is documentation-only. |
| Reviewed common ancestor | `d71ad80b0800880808a0857fd30477e6d96592c6` | Separates candidate changes from subsequent main development. |
| `fwyc0573/vLLM-BS`, `feature/frontier-comparison-instrumentation` | `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` | User-designated vLLM reference. |
| Frontier optional backend in reviewed main | `b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04` | Existing `collective-sim` gitlink. |
| Optional backend in candidate | `e564935d3874d8c71b52a554ab7c9a72e5e19f68` | Candidate gitlink; a fresh remote API lookup could not resolve this commit. Recheck in Step 7. |

Fetch again at execution start. Use the latest fetched main as the new branch base and record its actual SHA. Pin the candidate and reference revisions used in the audit. If either has advanced, inspect the changes; do not silently move a reference during an implementation step.

Main and the candidate have diverged. Main contains newer execution-time reporting, model support, profiling API selection, and accelerator handling. A change appearing in the candidate does not justify reverting those main changes. Use both a common-ancestor diff and a direct main-to-candidate comparison. [R1–R3]

### 2.2 What is already known, and what is not

The old task found useful independent defects: DP placement could restart at lane zero; shared monolithic work could split by local request phase; a legacy MoE profiling path omitted real arithmetic; routing runtime identity was insufficiently separated from load distribution. The task also accumulated timing experiments whose measurement conditions differ. Numerical calibration remains unresolved. [R2, R4]

Do not copy historical `PASS` counts into this task as fresh validation. Old tests are design references. Re-run or improve them on the new branch. In particular, some shared-forward tests replace ownership or wave behavior with stubs; those are not sufficient evidence for the real event lifecycle. The old MoE native test covers a limited BF16 case, not every dtype or backend. [R6, R7]

The old experiment checkout also used local vLLM overlays beyond the remote reference. They are not prerequisites for this PR. Only recover an overlay when a narrowly scoped native test genuinely needs a compatibility fix, and record exactly what it changes. Never use an undocumented overlay as the reference implementation.

## 3. Working agreement

### 3.1 New worktree, unchanged existing work

Create a **new named branch in a new worktree**. Do not switch, reset, clean, rebase, or reuse the calibration worktree. Do not delete another agent's files or modify its environment. A dirty existing checkout is a reason to preserve its state, not permission to clean it.

Suggested names:

```text
Branch:   fix/issue26-correctness-pr
Worktree: <Frontier checkout>/.worktrees/issue26-correctness-pr
Docs:     docs/development/issue26-correctness-pr/
```

These are task setup names, not paths to embed in production code. If a name already exists, inspect it. Resume it only when it is demonstrably this task's already-created worktree; otherwise choose a new descriptive suffix. Do not overwrite it.

### 3.2 Four tracked task documents

Keep the task's authoritative, GitHub-reviewable state in four files:

| File | Contents |
| --- | --- |
| `plan.md` | This specification, copied in full, with a short amendment history when decisions change. |
| `progress.md` | Current branch/base, step status, most recent checkpoint, blockers, exact next action, and concise chronological updates. |
| `review.md` | Source references, candidate-change dispositions, design decisions, and the final code-review findings. |
| `validation.md` | Commands, environments, test selections, results, baseline failures, native-test evidence, and validation limits. |

Main ignores `task_memory/` and `repairs/`. Do not rely on those paths for reviewable progress or alter their global ignore policy. Put this task's records under `docs/` and verify that they are tracked. Do not create an additional reporting framework or a separate document for every tiny action. [R5]

Store reproducible test inputs in the existing test layout. The repository also broadly ignores generated JSON/CSV files under `tests/`; verify that any intentionally committed small fixture is actually tracked. Prefer in-test data or a narrowly scoped ignore exception over force-adding generated output directories.

### 3.3 Checkpoint = implementation + validation + documentation + publication

At every completed key step:

1. Review the actual diff, run the required checks, and distinguish new failures from baseline failures.
2. Update the four documents as applicable. Include what changed, why, what was observed, and remaining limits.
3. Commit the implementation, tests, and documentation together, or as a small adjacent set of coherent commits.
4. Push **all those commits to the code repository branch**, not merely documentation or a separate notes repository.
5. Verify the remote branch points to the pushed local HEAD and provide GitHub commit/branch links to the user.

The initial setup checkpoint and final documentation-only corrections may naturally have no production changes. An implementation checkpoint must not be described as published when its code is still uncommitted or local-only.

Do not repeatedly amend or force-push commits the user may already be reviewing. Keep published history additive. A final history rewrite or PR merge requires explicit user approval.

A commit cannot contain its own final SHA in a tracked receipt. Report the new SHA in the user-facing update, and reference it in subsequent records when useful. Do not generate recursive receipt-only commits merely to embed a document's own commit hash.

### 3.4 Required user-facing update

After each key step, after a blocker, and before ending a work session, report:

```text
Completed: <the code/docs/tests changed in this round>
Reason: <the defect or maintainability problem being addressed>
Evidence: <tests run, counts, important observations, and limits>
Conclusion: <what is established and what remains unverified>
Published: <repository, branch, commit(s), push verification, GitHub links>
Next: <the next concrete action, or the decision needed>
```

Use complete sentences and actual findings, not a command transcript or a generic statement that progress was made. During longer work, give short updates at meaningful boundaries. A failed or skipped test is not a pass; a successful local commit is not a successful push.

If push fails, retain the local changes, record the error without exposing credentials, and report `LOCAL_ONLY` publication state. Resolve repository access before treating the checkpoint as delivered. Do not accumulate several completed but unpublished work packages without informing the user.

### 3.5 High-value decisions: question the user when evidence cannot decide

Do not ask the user to resolve facts available in source, tests, or existing records. Investigate those first. Do not ask routine permission to perform the work already authorized here.

When a consequential choice remains unresolved, ask **one focused question at a time**. State the relevant evidence, the alternatives, the effect on correctness/scope/compatibility, your recommended choice, and what remains blocked. Continue independent work without silently implementing the disputed choice.

Examples that justify a decision checkpoint:

| Trigger | Decision to put to the user |
| --- | --- |
| Correcting the MoE measurement changes accepted dataset semantics, and existing metadata cannot distinguish obsolete rows. | Approve a narrowly scoped data-contract change and migration, or defer that work package? Do not silently reinterpret old data. |
| The new DP strategy advertises configurations for which the available step identity is not valid. | Support those configurations with a small source-backed change, or explicitly narrow the opt-in capability? Do not quietly make it Qwen/DP2-specific. |
| Required native validation is unavailable or exposes a precision/backend incompatibility. | Keep the affected package blocked in a draft PR, or split it into a later PR? Do not waive numerical validation implicitly. |
| The optional submodule fix cannot be published to its configured remote. | Omit that conditional package, or authorize the necessary companion-repository work? Do not publish an unreachable gitlink. |
| A safe fix requires substantially widening the agreed public API or changing defaults. | Present the smallest viable alternatives before proceeding. |

Do not stop the whole task for a local naming or formatting choice. Do not turn this mechanism into approval requests after every step.

## 4. Implementation standards

### 4.1 Read the owning code before extracting a helper

For every changed runtime path, inspect its producer, consumer, lifecycle owner, existing helper, and focused tests. In `review.md`, briefly explain why the selected ownership is correct and which candidate mechanisms were removed or simplified.

Use existing configuration types, event handling, request transitions, predictor contracts, metrics ownership, cache utilities, and temporary-output helpers. Do not introduce another scheduler framework, duplicate model registry, generic compatibility layer, or task-specific runtime switch.

Respect repository `AGENTS.md` and applicable nested guidance. For a touched oversized module, identify the relevant responsibility and an existing or sensible extraction boundary. Prefer a small cohesive extraction when it simplifies the fix. Do not split unrelated code solely to hit a line-count target. [R1]

### 4.2 Validate at boundaries; trust established internal state

Validate external configuration, loaded artifacts, native backend capabilities, and untrusted event inputs at their owning boundary. Once required state is established, access it directly.

Avoid repeated `getattr(..., None)`, `hasattr`, fallback assignments, and `if x is None` ladders for fields that must exist. Do not turn a missing required execution record into zero work, an idle batch, an empty list, or a default routing implementation.

Retain legitimate optional states: an absent previous count snapshot, an optional expert map, or a genuinely disabled reporting path can be meaningful. Do not mechanically delete `None` handling or assert optional state always exists. Document the invariant rather than adding defensive checks at every call site.

Do not modify production behavior to accommodate incomplete test stubs. Build an appropriate test fixture or use the real object.

### 4.3 Names and structure

Use names that describe requests, DP lanes, shared forward steps, load reports, expert outputs, and measurement identity. Reuse existing terminology where it is part of Frontier's API. Prefer short functions with one state transition or computation.

Examples of suitable names are `RequestLoad`, `select_dp_lane`, `publish_request_counts`, `complete_shared_forward`, and `routing_runtime_path`; these are examples, not instructions to rename established APIs unnecessarily.

Do not introduce opaque labels such as `arm` or `protocol` for new functions, variables, flags, or work packages. Do not use task numbers, dates, model names, or calibration experiment names in production identifiers. Avoid vague suffixes such as `manager`, `adapter`, or `context` unless the object genuinely owns that responsibility.

### 4.4 Constants and supported scope

Derive DP/TP/EP sizes, replica IDs, layer bounds, tensor shapes, and dtype from existing contracts. Do not embed H200, Qwen, DP2, TP4, EP8, 4096 tokens, or historical request IDs in production logic.

A version-defined behavior is different from a calibration constant. The reference's waiting-count weight and publication intervals may appear as clearly named source-backed constants. Do not expose tuning flags just to eliminate a literal. Explicit restrictions such as one frontend and PP1 must be honest capability boundaries, not hidden assumptions.

### 4.5 Preserve current-main behavior outside each fix

Retain main's supported co-location, sequential PDD, and sequential PD-AF behavior; supported model families; existing accelerator selection; functional and legacy fused-MoE entry points; and reporting/measurement families.

In particular, do not restore an obsolete execution-time builder from the candidate over main's current `StageExecutionTime` and demand-driven reporting logic. Keep reporting disabled paths cheap. Do not add per-layer rescans or redundant predictor queries when existing owned state is sufficient.

Use a small paired CPU-runtime smoke as a regression signal after scheduler changes. Record event counts and reproducible conditions; investigate a clear regression. This is not a simulator-throughput benchmark campaign or a fixed percentage performance gate.

## 5. Execution order

| Step | Work | Main execution surface | Publication milestone |
| --- | --- | --- | --- |
| 0 | Create worktree, establish references and baseline | CPU / Git | New branch and tracked plan pushed. |
| 1 | Audit candidate changes and vLLM semantics | CPU / source | Reviewed scope and draft PR published. |
| 2 | Fix RR DP rotation | CPU | Source, tests, and results pushed. |
| 3 | Fix shared monolithic forward lifecycle | CPU | Full coherent runtime fix pushed. |
| 4 | Add bounded opt-in vLLM DP placement | CPU | Selection and event-integration evidence pushed. |
| 5 | Separate routing implementation identity | CPU | Config/data/model/cache chain pushed. |
| 6 | Repair legacy fused-MoE profiling | CPU + focused GPU worker | Code and CPU results pushed; native result published separately when complete. |
| 7 | Resolve optional zero-payload backend fix | CPU / companion repository | Included and remotely fetchable, or explicitly excluded. |
| 8 | Run combined regression, simplify the diff, hand off PR | CPU; native results reused only for unchanged code | Complete review-ready draft and final status pushed. |

Steps 2–4 come first because they affect runtime execution. Do not block them on optional backend availability or GPU allocation. Do not call Step 6 complete until its native acceptance requirements are satisfied, or the user explicitly removes it from this PR.

## 6. Step 0 — Create the worktree and establish a reproducible baseline

### Actions

Inspect the existing repository, remotes, worktrees, and local instructions. Confirm `origin` is the intended Frontier repository. Do not initialize candidate submodules merely to read source: the candidate contains an unresolved gitlink.

The following is a starting sequence from an existing Frontier checkout. Resolve name collisions before running `git worktree add`.

```bash
ROOT="$(git rev-parse --show-toplevel)"
git -C "$ROOT" remote -v
git -C "$ROOT" worktree list --porcelain
git -C "$ROOT" status --short

git -C "$ROOT" fetch --no-recurse-submodules origin
BASE="$(git -C "$ROOT" rev-parse 'origin/main^{commit}')"
CANDIDATE="$(git -C "$ROOT" rev-parse 'origin/bug/ttft-check^{commit}')"
MERGE_BASE="$(git -C "$ROOT" merge-base "$BASE" "$CANDIDATE")"
BRANCH="fix/issue26-correctness-pr"
WORKTREE="$ROOT/.worktrees/issue26-correctness-pr"

# Inspect any pre-existing branch or directory; never overwrite either.
git -C "$ROOT" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
cd "$WORKTREE"
mkdir -p docs/development/issue26-correctness-pr
```

Copy this document into `plan.md`; create the other three task records. Record full SHAs, repository URLs, branch/worktree paths, the chosen Python executable, and the first planned validation commands.

Use the existing CPU environment when suitable. Keep GPU dependencies out of the minimal simulator environment. Avoid changing a shared editable installation to point at this worktree: use a task-specific environment or an explicit worktree `PYTHONPATH`. Record and verify the actual imported `frontier` path.

Use Frontier's existing scratch-root support for test output. If `FRONTIER_TMP_ROOT` is used, point it to a new task-owned directory and record it. Do not reuse, delete, or overwrite old calibration caches.

### Baseline validation

Read `AGENTS.md`, applicable nested instructions, `pyproject.toml`, existing test configuration, and the relevant test helpers before selecting commands. Run a small existing CPU suite for the areas this PR will touch. Candidate baseline files include:

```text
tests/unit/test_open_source_release_arch_guard.py
tests/unit/test_cluster_scheduler_dp_lanes.py
tests/unit/test_colocation_release_review_contracts.py
tests/unit/test_config_owned_contracts.py
tests/unit/test_stage_execution_time.py
tests/unit/test_stage_finalized_contract.py
```

Check their existence and dependency requirements at the fetched revision. Use an explicit file selection, not an unreviewed repository-wide command that launches native dependencies. Record any import, collection, or baseline failure with its actual exception. Do not modify production code to conceal a pre-existing environment problem.

Run an existing small Frontier-only CPU smoke with synthetic/dummy execution times and the built-in analytical communication backend. Record the command and expected completion criteria. No trained historical profiles or GPU service are needed for this baseline.

### Exit and publication

The new worktree is based on the recorded main SHA; existing worktrees are unchanged; task docs are tracked; the baseline result and limitations are recorded. Commit and push the setup checkpoint. Verify that GitHub shows the branch and the full plan. A baseline failure must be explicit; it does not automatically prevent a source audit.

## 7. Step 1 — Audit candidate changes and the reference behavior

### 7.1 Three-way source review

Use all three views below, then read complete functions and their call sites rather than relying on patch context alone:

```bash
git diff --name-status "$MERGE_BASE" "$CANDIDATE" -- frontier tests docs
git diff "$MERGE_BASE" "$CANDIDATE" -- <relevant-paths>
git diff "$MERGE_BASE" "$BASE" -- <relevant-paths>
git diff "$BASE" "$CANDIDATE" -- <relevant-paths>
```

Read the donor task's `design_dp_load_balancing.md`, `design_shared_forward_sync.md`, `review_shared_forward_sync.md`, and relevant focused test reports. Consult `summary.md` or `handoff.md` only to resolve provenance or an implementation decision; do not restart the calibration investigation. The complete historical archive is not required reading for this PR.

For each meaningful candidate hunk, record one disposition in `review.md`:

| Disposition | Meaning |
| --- | --- |
| `PORT` | The behavior is justified and still missing from main. |
| `ADAPT` | Preserve the intent but implement it using main's current ownership or API. |
| `ALREADY_PRESENT` | Main already has an equivalent correction; add a missing regression only when useful. |
| `DROP` | Redundant, experimental, overcomplicated, unrelated, or incorrect. |
| `BLOCKED` | An evidence or user decision prevents inclusion. |

Record the old defect, main behavior, reference behavior when applicable, chosen owner, planned test, and donor commit/path. Do not cherry-pick a large commit solely because its message says it is a fix.

### 7.2 Required vLLM 0.10.2 source reading

Create or use a separate read-only reference checkout pinned to the designated vLLM revision. Establish its relationship to vLLM 0.10.2 from version/release history and relevant source differences. The fork branch name or installed package version alone is not proof. When comparison to the upstream tag is necessary, pin that tag's resolved commit as well and record meaningful fork differences; do not substitute a newer vLLM release.

| Reference source | Questions the audit must answer |
| --- | --- |
| `vllm/v1/engine/core_client.py` | How are DP engines ordered and scored? What does one frontend reserve locally? How are count updates applied? What changes with multiple frontends or explicit DP rank selection? |
| `vllm/v1/engine/coordinator.py` | When are changed counts published? What do the previous-step snapshot, minimum collection wait, unchanged heartbeat, and wave/step ordering mean? |
| `vllm/v1/engine/core.py` and scheduler stats producers | Which events cause a report, at what point relative to request state updates, and which reports are suppressed? |
| `vllm/v1/core/sched/scheduler.py` | What belongs to waiting versus running, including admitted-but-unscheduled work, preemption, and completion? |
| `vllm/v1/worker/gpu_model_runner.py`, `vllm/forward_context.py` | Why can DP source batches have different local phases while taking part in shared expert work? How are token populations and dummy participants represented? |
| `vllm/model_executor/models/qwen3_moe.py` | What is the model's gated expert computation and reduction path? Use this as a concrete case, not a production model-name condition. |
| `vllm/model_executor/layers/fused_moe/fused_moe.py`, `layer.py`, related activation/output helpers | What is the exact low-level arithmetic, routing-weight placement, workspace use, expert-map behavior, and local output reduction? |
| `vllm/distributed/device_communicators/all2all.py` and group helpers | Which reductions are local expert aggregation versus distributed communication? Avoid accounting for either twice. |

Produce a concise reference-behavior table with pinned file/symbol references. Explain why Frontier represents the relevant behavior the way it does. No live vLLM server is needed.

### 7.3 Main integration points to inspect

Inspect request admission, `GlobalBatchEndEvent`, shared forward identity, waiting rooms, stage ownership, layer advancement, `Batch`/`Request` completion, `StageExecutionTime`, and metrics reporting before the scheduler fixes.

Inspect `ReplicaConfig` creation/copying, routing-runtime resolution, model training signatures, model registries, family/precision selection, persistent cache loading, and current profiling backend selection before predictor/profiler edits.

Prefer existing test utilities such as the repository's scratch-root and predictor-cache fixtures. Inspect any donor reference-comparison helper before adopting it. Do not create a second general testing harness.

### Validation and publication

Finalize the work-package dispositions and exact CPU test selections. Identify any decision checkpoint rather than inventing an answer. Create a **draft PR** after the branch and scope audit have been pushed, using the existing repository template when present.

Suggested title: `Fix DP request placement, shared forwards, and MoE profiling contracts`.

Use `Related to #26`, not `Fixes #26`. State that latency calibration and vLLM E2E comparison are out of scope. Keep the draft open and update it at subsequent checkpoints. Do not request merging before the user has reviewed it.

## 8. Step 2 — Preserve RR DP rotation across scheduling calls

### Source scope

Primary implementation: `frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py::_schedule_batch_mode`. Start from main's implementation and the donor's small correction. Existing regression starting point: `tests/unit/test_cluster_scheduler_dp_lanes.py`.

### Required behavior

Request assignment must not depend on how an identical ordered request stream is divided across calls to `schedule()`.

The donor's intended order is:

```python
ordinal = completed_assignments + request_index
replica_index = ordinal % num_replicas
dp_lane = (ordinal // num_replicas) % dp_size
```

Reuse the existing persistent counter if its meaning matches this calculation. Do not create a second counter or cache without a demonstrated need. Preserve the existing order across replicas and the return structure. Use actual replica IDs, not an assumption that IDs equal list indices.

Keep dedicated decode/PD-AF allocation logic unchanged unless the same defect is independently demonstrated there. This is an RR correction, not vLLM load-balancer emulation.

### Required CPU tests

Test the production scheduling path with one request at a time, a single burst, and different batch boundaries for the same sequence. Cover DP1, more than two DP lanes, multiple replicas, non-contiguous replica IDs, an empty scheduling call, and continuation after that empty call. A small parameterized table is sufficient; no arbitrary scenario count is required.

A minimal regression must fail on the old implementation for the intended reason: repeatedly starting lane assignment from zero. Compare placement by request identity and preserve order within each destination lane, with every request assigned exactly once. Do not require the flattened return-list order of one burst to equal that of many calls: the existing implementation may group each call's results by replica.

Run the existing DP-lane and RR tests plus the small CPU smoke. Compare unaffected DP1 behavior against the recorded baseline.

### Exit and publication

The same request sequence has the same assignment regardless of call partitioning; the new test is demonstrably sensitive to the old defect; unrelated allocation behavior is unchanged. Update docs, commit source and tests, push, and report the result and next step.

## 9. Step 3 — Repair shared monolithic forward completion

### Source scope

Review these donor paths together, then adapt them to main:

```text
frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py
frontier/scheduler/utils/forward_sync_state.py
frontier/scheduler/utils/sync_state.py
frontier/scheduler/utils/forward_collective.py                 # donor addition
frontier/scheduler/utils/ep_wave_inputs.py
frontier/scheduler/utils/ep_wave_schedule.py
frontier/scheduler/utils/prefill_collective.py
frontier/scheduler/utils/decode_collective.py
frontier/events/replica_stage_schedule_event.py
```

Also read the current sync-entry and stage-ownership helpers, `PrefillSyncEvent`, `DecodeSyncEvent`, `BatchStageEndEvent`, and request completion consumers. The listed paths are an audit map, not permission to overwrite each file.

### Invariants to implement

**One shared forward identity.** In a monolithic model requiring shared expert execution, participating source lanes join the same forward step regardless of local prefill/decode classification. Do not add a parallel hierarchy of event classes. Preserve existing separation for disaggregated execution where it has real meaning.

**One source owner per request.** Preserve original source batches and their request/token vectors. A real request cannot appear in two source lanes of the same shared step. Validate ownership at the appropriate group-formation boundary rather than repeatedly scanning unchanged membership at every operator.

**One completion of the shared expert work.** The completed shared work must be consumed once, and full-stage ownership must be restored once for the participating group. Understand what the restore helper's return value means. Do not conflate “not attempted” with “not applicable” through fallback calls that restore owners twice.

**Source-local continuation.** Each source uses its own next-attention inputs, context lengths, stage tail, and request phase. Do not choose a single sample batch and apply its predicted duration to all source lanes. An idle participant may be required for synchronization but must not become a completed user request.

**Correct layer advancement.** In a mixed source, advance the decode subset exactly once per completed layer. Pure-decode sources follow their existing completion path; prefill requests must not gain a duplicate decode-layer increment. Respect pipeline stage bounds and supported speculative-decoding state.

**Single accounting owner.** Reuse main's current execution-time and reporting ownership. Shared waiting contributes to elapsed stage time; it must not also be invented as CPU work or charged twice as an operator. Include shared expert work in the appropriate source model-time accounting without fabricating a second metrics system. Preserve demand-driven reporting when it is disabled.

Do not change same-layer admission timing for dense layers within a MoE model merely because it can be refactored at the same time. Fix source-local continuation and bookkeeping without expanding the scheduling policy.

### Required CPU tests

Use the donor's `test_monolithic_mixed_forward_sync.py` as a starting point, not the full acceptance suite. Add behavior tests covering:

| Case | Required assertion |
| --- | --- |
| Prefill/prefill, decode/decode, prefill/decode, and mixed-source combinations | Each admitted group reaches one shared completion without deadlock. |
| Reversed source arrival order | The same shared identity and correct outcome; no dependence on which phase arrives first. |
| Unequal source tokens/context lengths | Each lane queries/uses its own continuation; a deliberate unequal-duration fixture detects borrowed timing. |
| A real source plus idle participants | Idle participation does not create fake requests, progress, or terminal callbacks. |
| Duplicate request ownership | A clear failure at the owning boundary, before producing inconsistent state. |
| Multiple layers and successive forward steps | Owners, open-step bindings, waiting rooms, and pending events are consumed or released correctly. |
| Mixed-batch decode requests | One layer increment and one final token credit where appropriate; prior TTFT remains unchanged. |
| Supported dense-layer transitions within a MoE model | No invented EP collective for dense work; local continuation remains correct. |
| Reporting on versus off | Same simulated execution outcome; reporting does not introduce extra mandatory runtime work. |

At least one integration test must run the **real Frontier event loop and real admission/ownership/completion code** through a small multi-request case with prefill overlapping decode. Inject deterministic execution times only at the predictor/backend boundary. Do not replace the synchronization, ownership restoration, or terminal callbacks being tested.

Assert exact request/token conservation, unique completion, no stranded source batch, no live ownership record after completion, and no unfinished request when the event queue drains. A nonempty metrics file is not sufficient.

Run relevant existing tests for forward identity, EP-wave materialization, stage ownership, stage execution reporting, sequential PDD/PD-AF, and supported prefix/speculative paths affected by the call chain. Resolve concrete test file names in Step 1 and record the executed list. No GPU is needed.

### Exit and publication

Publish the shared lifecycle fix as a coherent unit, not an intermediate state where waiting rooms are shared but completion ownership is still phase-specific. Tests must distinguish the old mixed-phase defect and detect a deliberately wrong source-timing implementation. Record any unsupported path explicitly. Commit, push, and provide the runtime-invariant evidence.

## 10. Step 4 — Add bounded, opt-in vLLM-style DP placement

### Source scope

Candidate components to review:

```text
frontier/config/cluster_scheduler_config.py                  # donor addition
frontier/config/config.py
frontier/types/cluster_scheduler_type.py
frontier/scheduler/request_load.py                          # donor addition
frontier/scheduler/utils/vllm_dp_load_balancer.py             # donor addition
frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py
frontier/scheduler/cluster_scheduler/cluster_scheduler_registry.py
frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py
frontier/scheduler/replica_scheduler/base_replica_scheduler.py
frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py
frontier/events/cluster_schedule_event.py
frontier/events/global_batch_end_event.py
```

The config-family extraction is optional unless needed to keep the change cohesive. Preserve existing imports, config discovery, CLI flattening, and serialization. Do not add a broad config migration merely to register a new strategy.

### Required behavior and boundaries

Preserve the reference's separation between engine counts, the last publishable snapshot, and frontend estimates. For the intended one-frontend mode, select the first engine with minimum `4 * waiting + running`, and reserve one local waiting request. A fresh published snapshot replaces the frontend estimates; it is not an incremental adjustment to their locally reserved counts. [R3]

Model the changed-count publication, previous-step snapshot, minimum collection wait, and unchanged heartbeat according to the pinned reference. The expected source constants are 4, 50 ms, 100 ms, and 5000 ms; confirm them in the actual source before using them. Explain the initialization epoch and equal-time ordering in code comments or the reference table, not as hidden timing adjustments.

Use one small state owner for the count-publication/selection behavior and the existing cluster scheduler for DES integration. Pass simulation time explicitly through a clean interface; avoid a mutable “last routing time” bridge if the existing scheduling API can carry time directly without a broad change. If an adapter is necessary for older policies, keep it thin and document its contract.

Report load at the correct post-step point, after the lane's real request-state transition. A load accessor must count admitted-but-unscheduled running requests and preempted waiting work correctly. Reuse that accessor in existing diagnostics rather than maintain two definitions.

Lazy timer advancement is acceptable when it preserves all observable publication/selection behavior. Do not create perpetual heartbeat events that keep a drained simulation alive. Verify suppressed reports and same-time routing/report events, not just standalone timeout arithmetic.

Initial support remains one serving Replica, one modeled frontend, co-location/MONOLITHIC, V1, and PP1. Keep the strategy opt-in and retain all existing defaults. Do not claim support for multi-frontend delivery, IPC timing, elastic scaling, or exact warm-start publication phase.

The donor uses `ForwardSyncState.get_step_id(batch)` as a report-order key. **Do not assume this is valid for every configuration accepted by the new strategy.** Check that its order/equality properties match the reference's relevant wave/step semantics for the admitted paths. Check dense models, DP1, idle participants, successive request waves, and real batch-end hooks. Provide a correct existing identity, or propose a clearly bounded capability restriction to the user; do not manufacture a guessed step number or silently use a request ID.

### Required CPU tests

Start from `test_vllm_dp_load_balancer.py`, then add the missing integration coverage:

| Layer | Cases and assertions |
| --- | --- |
| Pure count selection | Weighted counts, deterministic ties, local reservations, replacement by a new snapshot, empty initial counts. |
| Publication state | Same-step reports, a newer step while changes are pending, unchanged/suppressed reports, heartbeat replacement, delayed frontend observation. |
| Ordering | Report at a deadline, select at a deadline, both at the same timestamp, multiple events within the same millisecond, monotonic time validation. |
| Request populations | Waiting admission, running but not scheduled, preemption, completion, and the resulting reports from actual lane state. |
| Real DES integration | `ClusterScheduleEvent` supplies time; `GlobalBatchEndEvent` observes post-step state; the strategy does not keep an otherwise finished run alive. |
| Capability and defaults | Allowed topology works; unsupported topology fails before simulation; RR remains the default where it was previously selected. |
| Configuration | Enum/registry discovery, old imports, generated CLI selection, config copying, and round-trip representation. |

Use a compact reference-driven trace test with explicit times and expected publications/selections. Prefer exercising relevant pinned reference methods through a minimal test-only fake clock/poller when practical. Otherwise derive the expected trace independently from the source and document the mapping. Inspect the existing donor `tests/integration/issue26_dp_coordinator_reference.py` before deciding whether it is reusable.

Do not simply duplicate the new balancer into the expected-result generator. Source-string assertions and tests that assign final state directly cannot replace behavioral tests.

### Exit and publication

The selection and publication rules are source-backed; actual Frontier events exercise them; supported scope is truthful; current defaults and unrelated strategies are unchanged. Publish the implementation and tests with an explicit statement that source-level behavior is validated but no vLLM serving-placement or timing equivalence is claimed.

## 11. Step 5 — Separate routing load distribution from implementation identity

> **Closed 2026-09-22 without source changes (user decision; see `review.md` W5 "Final disposition" and `requirements.md`).** The premise check found the collision this step fixes unreachable on main: the routing distribution has one global field and no per-role override, so every cluster resolves one routing path per run. The override that would have made it reachable was judged not worth its configuration surface, and the drafted implementation is archived as `w5_reverted_moe_routing_runtime_path.patch`. The specification below is kept as written for the record.

### Source scope

```text
frontier/config/config.py
frontier/moe_routing_runtime.py
frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py
frontier/execution_time_predictor/shared_prediction_model_manager.py
```

Inspect current model-cache helpers and any query caches reached by these paths. Existing donor tests include `test_moe_routing_runtime.py` and `test_moe_routing_runtime_model_sharing.py`; reuse main's predictor-cache fixtures where applicable.

### Required behavior

The configured expert-load distribution and the routing implementation used for timing prediction are separate dimensions. Retain existing default resolution, but support the explicit validated runtime override using Frontier's normal config mechanisms.

Follow the value through:

```text
Configuration creation / copying
    -> resolved routing runtime
    -> dataset row selection and validation
    -> training identity
    -> trained-model registry and precision/measurement-family selection
    -> persistent cache load
    -> runtime model lookup / query cache, where relevant
```

Do not implement only the CLI flag. Two routing implementations with the same layer shape must not select each other's routing-cost model. Conversely, equivalent resolved configurations should retain sharing; do not separate models merely because one configuration used an explicit default and another left it implicit.

Only include runtime identity where it changes the measured semantics. Avoid unnecessarily retraining or duplicating unrelated expert-compute models when their inputs and contracts are identical. Keep current layer, precision, and measurement-family identity intact, including main's device-event support.

Resolve required configuration once at its owner. Avoid nested `getattr` fallbacks on every query or reattaching mutable identity to a cached estimator without provenance. An artifact with unknown or conflicting routing identity must not be silently relabeled as the caller's requested runtime. Any recovery of missing metadata needs proof from the existing cache key and source dataset; otherwise reject it clearly or use an approved migration.

### Required CPU tests

Use tiny synthetic rows and existing training/cache paths, not historical profiling data or a large fitting run.

Verify defaults and explicit overrides; invalid runtime rejection; config reconstruction/copying; dataset filtering; rejection of incompatible or ambiguous rows; two runtimes with otherwise identical features; equivalent explicit/implicit configurations; precision/family separation; cache miss and cache hit; serialized reload in a fresh manager instance; and rejection of conflicting cached metadata.

Include at least one test that goes through actual minimal training or the existing trainer's real storage path and then a cache reload. Registry-only fake-estimator tests do not demonstrate persistent-cache isolation. Test unchanged ordinary model sharing as well as the newly separated routing models.

### Exit and publication

The override reaches the actual model selection; mismatched runtimes cannot collide; existing compatible sharing remains intact; fresh and cached paths agree. Publish source, tests, and the identity rules. Do not claim that timing prediction has become more accurate without a separate calibration task.

## 12. Step 6 — Repair legacy fused-MoE profiling without regressing main

### Source scope and implementation

Primary source: `frontier/profiling/moe/moe_vllm_kernel.py`. Read its callers, measurement export, training targets, and current backend selection before editing.

The legacy path must perform the actual supported gated expert computation:

```text
First expert matrix multiplication
    -> gated SiLU activation
    -> activation quantization, when the selected supported path needs it
    -> second expert matrix multiplication with the correct routing weights
    -> local reduction of the top-k expert outputs
```

The donor adds `silu_and_mul`, local `moe_sum`, an activation buffer, and shared output workspace. Port the justified arithmetic and layout into main's current legacy entry point. Do not replace the file or revert functional fused-experts, ROCm/MXFP4 handling, profile-method validation, or shared timer statistics. Main already tests population statistics and finite single-sample behavior; preserve them. [R7, R8]

Local top-k output reduction is not a distributed collective. Do not add or remove DP/TP/EP communication costs as part of this arithmetic repair.

Use explicit existing backend selection. Avoid broad import/exception fallback paths that silently switch the implementation. Keep optional native dependencies out of CPU simulator imports. Do not change a public profiling return contract just to expose an intermediate tensor to a test when the existing low-level helper can be tested directly.

### Measurement ownership is part of the fix

Specify exactly which operations belong to the corrected measured target. In particular, identify whether local output reduction is contained in the grouped-expert measurement or separately represented. The full local computation must be counted exactly once in training and runtime queries.

Do not silently keep using old rows or cached models as if their measured scope had changed. Prefer the repository's existing artifact/measurement identity mechanism for a narrowly scoped distinction. Preserve unaffected measurement families and data. Do not delete shared caches or require regeneration of all profiling data.

If existing metadata cannot safely distinguish the old incomplete target from the corrected target, stop that part for a user decision. Present the smallest metadata/migration change and the alternative of deferring the profiling package. “The old cache probably will not be used” is not an acceptable compatibility rule.

Do not generalize from the Qwen BF16 case to arbitrary activation or quantization. Follow the actual model contract. If an advertised path is unsupported, expose that fact and discuss the compatibility impact before narrowing it; do not silently compute gated SiLU for a non-gated model.

### CPU validation

Extend the existing test layout rather than duplicating the native test framework. Starting points include:

```text
tests/unit/test_moe_fused_event_contract.py
tests/unit/test_moe_native_admission.py
tests/unit/test_moe_mxfp4_increment10.py
tests/unit/test_device_timer_contract.py
tests/unit/test_timer_owner_lifecycle.py
```

Some profiling-boundary tests require CPU-importable Torch/native-related Python packages even though no GPU kernel runs. Inspect their import requirements and use the existing suitable test environment. Do not make those optional packages new mandatory dependencies of the simulator.

Required checks include the supported call order, activation/output dimensions, routing-weight placement, expert-map propagation, workspace alias lifetime, measurement ownership, old-target/cache admission, and unchanged functional-backend dispatch. Add a small explicit CPU arithmetic example that distinguishes a gated activation from merely slicing the first matrix output. Mark mock-based dispatch tests as boundary validation, not native numerical parity.

Maintain existing event-method/platform checks and statistics tests. Do not restore the donor's older timing/statistics implementation while moving its arithmetic fix.

### Native numerical validation: narrow but required

Run a **local-expert numerical test on a GPU worker**, not a serving or timing benchmark. The CPU master must not execute CUDA work. Reuse the approved job mechanism and existing suitable image/environment; record actual source/import paths, binary versions, device, command, and results. Do not invent a new image, download model weights, allocate an eight-GPU server, or build a full serving environment unless the minimal test genuinely requires it and the user authorizes that expansion.

Use identical generated inputs, weights, routing weights/IDs, expert map, dtype, and kernel configuration for the repaired Frontier helper and the pinned vLLM reference. Compare real outputs, including local output reduction. A single compatible GPU can ordinarily test different local EP partitions sequentially; the logical EP size is not a requirement for a distributed serving run.

Minimum BF16 acceptance:

| Case | Purpose |
| --- | --- |
| Existing Qwen-shaped 4096 and 4097 token cases, using the checked-in model config and local expert-map layout | Preserve the donor test's important production-shaped coverage. |
| More than one local expert partition | Detect incorrect global/local expert handling without launching an EP cluster. |
| A smaller boundary case with a different valid top-k and uneven expert occupancy | Detect shape-specific correctness and missing-local-expert output errors. |

The donor test is `tests/unit/test_moe_fused_expert_numerical_parity.py`. Adapt its useful checks into the repository's appropriate native-test location and capability gating. Do not import the entire GPU stack into the ordinary CPU unit suite merely to preserve its old filename.

For identical BF16 kernels and operation order, retain the donor's zero-tolerance comparison unless an identified supported implementation difference justifies a documented tolerance. Do not loosen tolerance merely to make a failure disappear. Check finite outputs and repeated invocation with different inputs so workspace reuse cannot leak stale results.

Also validate any precision/backend path whose advertised execution is changed by this patch. A CPU shape test does not prove FP8 arithmetic. Keep the GPU matrix limited to affected paths, but do not claim an untested changed native path is proven correct. Resolve an unavailable required path with the user rather than launching an open-ended GPU campaign.

### Exit and publication

Push the code, CPU tests, and documentation before waiting on GPU availability; mark native validation `NOT_RUN` or `BLOCKED` explicitly. Once native checks finish, publish their concise results and any required fixes. Step 6 is `PASS` only when arithmetic, measurement ownership, artifact compatibility, and the required native checks are all satisfied.

The unrelated runtime fixes may continue while this step is blocked. The draft PR must not be presented as ready to merge with a required numerical check silently skipped.

## 13. Step 7 — Resolve the conditional zero-payload backend change

### Decision boundary

Inspect `.gitmodules` and the actual gitlink in the new branch. The configured optional backend repository is `fwyc0573/frontier-htsim`. At specification time, the candidate's `e564935d...` commit could not be resolved through the configured remote API. Recheck repository access and commit reachability; an access problem and an unpublished commit are different findings. Do not repeatedly try guessed repositories. [R9]

If the fix is available, inspect its diff against main's pinned backend and ensure the parent gitlink does not pull unrelated changes. If it exists only locally, verify that exact local source and tests before publishing it. If necessary, reconstruct the small input-handling fix from source and failing tests; do not treat a report as a substitute for code review.

Keep the backend package conditional. Its absence must not block RR, shared-forward, routing, or other analytical-backend CPU tests.

### Required behavior

Explicit zero is valid input where the existing backend accepts an empty transfer. Missing required payload is still an error. Negative payload is rejected. An explicit CLI zero overrides a positive JSON value. Positive-payload behavior remains unchanged.

Do not equate zero payload with zero total collective time. Preserve the backend's existing synchronization/latency semantics for the selected collective. Do not fit or change its timing parameters.

### Required CPU validation and publication

Start from donor `tests/unit/test_collective_sim_zero_payload.py`. Test the real scenario serialization, CLI runner, and built CPU simulation executable with zero, missing, negative, and positive input, plus CLI-versus-JSON precedence. A mocked subprocess result is insufficient.

If a new companion-repository change must be created or published, obtain the user's decision on that additional repository scope first; continue independent Frontier work while it is pending. Once authorized, create a dedicated backend branch, commit tests and source, push to the correct permitted remote, and open a companion draft PR when appropriate. Do not push to its main branch. Publish the backend commit **before** updating Frontier's gitlink.

Validate the resulting Frontier checkout with a clean temporary checkout/clone and `git submodule update --init` for this backend. This must fetch the exact intended SHA without relying on another worktree's local Git object store. Build and execute the CPU runner tests in the recorded environment.

Then commit and push Frontier's gitlink, parent-side tests, and docs. Report both repositories and their commits/dependency status.

If the package cannot safely be included, leave Frontier on its current-main backend revision and record `EXCLUDED` with the reason. Do not vendor the backend, alter `.gitmodules` to an unapproved destination, or publish an unreachable gitlink. Ask the user when companion-repository scope or access needs a decision.

## 14. Step 8 — Combined regression and final PR review

### 14.1 Validate the integrated branch

Run the selected focused suites together, not only each package in isolation. Use fresh task-owned outputs and caches where the test concerns first-load behavior. Also exercise cache-hit paths intentionally rather than accidentally inheriting a donor cache.

The combined CPU selection must cover:

| Area | Required result |
| --- | --- |
| RR and opt-in DP placement | Persistent assignment, reference count semantics, real event hooks, and unchanged defaults. |
| Shared forwards | Correct mixed-source progress, source-local timing, ownership release, no dangling waiting rooms, and request/token conservation. |
| Existing architectures | Supported co-location, sequential PDD, and sequential PD-AF CPU regressions remain valid. Preserve their existing unsupported-mode guards. |
| Stage reporting | Current `StageExecutionTime`/metrics behavior survives; enabling reporting does not change simulated completion. |
| Routing models | End-to-end config-to-query identity, compatible sharing, and persistent-cache isolation. |
| Profiling boundaries | Correct legacy arithmetic structure and measurement ownership; main's functional/accelerator/timer dispatch tests remain intact. |
| Optional backend, when included | A fresh remotely fetchable submodule checkout passes the actual CPU runner tests. |
| Public interfaces | CLI/config registration, imports, supported backend selection, and relevant docs are consistent. |

Run small deterministic Frontier-only simulations with real event processing: an RR arrival stream, an opt-in DP-placement stream, and a stream that creates overlapping prefill/decode source batches. Reuse existing fixtures. Their purpose is lifecycle validation, not matching historical H200 latency. Include heterogeneous/dense-layer or pipeline cases where the changed shared code applies, even though the new DP strategy itself remains PP1-only.

Before handoff, fetch main again. If it advanced, inspect overlap and assess whether integration validation is needed. Prefer additive integration into the published branch when necessary; do not rebase or force-push away the user's review anchors. Rerun the affected tests after any integration change.

Do not run native profiling suites on the CPU master. Do not launch vLLM E2E comparisons as a final confidence check; they are not part of this task.

### 14.2 Review the complete diff against the actual PR base

Perform a final line-by-line review of changed code and meaningful surrounding context. Record the reviewed revision and actual review method; do not describe a self-review as independent review. Use these questions:

- Is every change linked to a demonstrated defect, an essential regression test, or a directly related simplification?
- Are state ownership and initialization explicit? Are repeated fallback checks or parallel state representations still present without a reason?
- Do source batches keep their own shape and progress? Are terminal events and ownership transitions unique?
- Are config, layer, routing-runtime, precision, and measurement identities preserved through both fresh and cached paths?
- Is any operation omitted, counted twice, or relabeled without compatible metadata?
- Did the patch preserve current-main model/backend support and demand-driven reporting?
- Are tests checking production behavior rather than copying the implementation or replacing the behavior under test with a stub?
- Are any workstation paths, credentials, datasets, weights, generated traces, caches, or unreachable submodule references staged?
- Can a reader understand the names and functions without the historical calibration conversation?

Remove candidate scaffolding that no longer has a purpose. Do not add speculative abstractions in the cleanup pass. Re-run the affected tests after cleanup.

A broader CPU test selection may expose pre-existing failures. Reproduce them on the recorded base with the same environment where practical; report them as baseline failures with evidence. Do not relabel an unexplained failure as pre-existing or expand into unrelated repairs without a decision.

### 14.3 Draft PR contents

Update the existing draft PR rather than creating duplicates. Its description should contain the problem statement, included fixes and deliberate exclusions, important design choices, test commands/results, native-test status, compatibility/data-scope implications, source reference revision, and any companion PR dependency.

Link the tracked `plan.md`, `progress.md`, `review.md`, and `validation.md`, plus the relevant implementation commits. Explain that no vLLM serving/TTFT comparison was performed or required. Do not paste large logs into the PR body.

Keep the PR draft until the user reviews it and decides the next GitHub action. Report whether technical acceptance is complete independently of GitHub's draft status. Do not merge, squash, delete branches, or close Issue 26.

## 15. Validation records and completion rules

### 15.1 Keep four independent statuses

Do not collapse code status, test status, publication status, and user approval into one `PASS` label.

```text
Work package: NOT_STARTED | IN_PROGRESS | PASS | FAIL | BLOCKED | EXCLUDED
Test result:  PASS | FAIL | SKIPPED | NOT_RUN
Publication:  LOCAL_ONLY | PUSHED_VERIFIED
User review:  NOT_REVIEWED | CHANGES_REQUESTED | APPROVED
```

`EXCLUDED` must state whether exclusion was permitted by this specification or explicitly approved by the user. `SKIPPED` is never evidence that a native check passed. A job's platform status is not a substitute for test output and exit status.

### 15.2 Minimal per-checkpoint evidence

Use concise tables in `validation.md`:

| Field | Record |
| --- | --- |
| Step and purpose | Which correctness claim the test supports. |
| Source under test | Base/code revision and the exact candidate diff tested, or the committed implementation revision. |
| Environment | Python executable/version; relevant package versions; working directory and import path; native details only when relevant. |
| Command | Executable command with the selected test files and options. |
| Outcome | Pass/fail/skip counts, process exit code, and important assertions or numerical result. |
| Baseline comparison | Whether the defect test fails before the fix and how unrelated baseline failures were handled. |
| Limits | What was mocked, not run, unsupported, or not established by the result. |

Record tested code accurately. If testing precedes the checkpoint commit, keep the production/test diff unchanged between that run and commit; describe it as the tested checkpoint diff rather than claim the unmodified parent SHA was tested. Rerun affected checks after additional code edits. The user-facing update supplies the resulting commit SHA.

Keep enough small evidence in the tracked documents to review results in GitHub without access to local logs. Large raw output remains in the task-owned scratch location and is referenced as supplementary material, not the sole proof of completion. Redact secrets and avoid committing private environment dumps.

### 15.3 Final acceptance checklist

- [ ] A new worktree and branch were created from a freshly fetched and recorded main revision.
- [ ] The candidate/main/source audit records which changes were ported, adapted, already present, dropped, or blocked.
- [ ] Relevant vLLM 0.10.2 behavior is explained from pinned source, not assumed from branch names or old reports.
- [ ] RR DP assignment is independent of scheduling-call partitioning.
- [ ] Real shared-forward event processing preserves ownership, source-local timing, progress, and unique completion.
- [ ] The opt-in DP strategy is correctly integrated, explicitly bounded, and does not change defaults.
- [ ] Routing-runtime identity is correct through config, data, training, registries, persistent caches, and lookup.
- [ ] Any included legacy MoE change has correct arithmetic, single measurement ownership, compatible artifact handling, and required native numerical evidence.
- [ ] Any included backend gitlink is reachable from the configured remote and tested from a fresh checkout; otherwise the conditional package is explicitly excluded.
- [ ] Relevant current-main architecture, reporting, model, and accelerator regressions are preserved.
- [ ] Required tests are actually executed, with skipped/unrun work visible rather than converted to a pass.
- [ ] The final diff contains no calibration constants, unsupported accuracy claim, task archive, generated cache, or unrelated rewrite.
- [ ] Source, tests, and task docs are committed and pushed, and remote HEAD verification succeeds.
- [ ] The draft PR links all review material and distinguishes technical completion from user approval.

No hidden GPU or vLLM E2E requirement may appear at the end. The only native work required here is the explicitly described numerical validation for included changed native execution paths.

## 16. Publication commands and interruption recovery

### 16.1 Normal checkpoint publication

Use explicit paths when staging. Do not use an unreviewed `git add -A` in a worktree containing experiment outputs.

```bash
git diff --check
git status --short
# Replace the placeholders with the reviewed source, tests, and docs paths.
git add -- <source-paths> <test-paths> docs/development/issue26-correctness-pr/
git diff --cached --stat
git diff --cached --check
git commit -m "<specific behavior fixed>"

BRANCH="$(git branch --show-current)"
git push --set-upstream origin "$BRANCH"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)"
test "$LOCAL_HEAD" = "$REMOTE_HEAD"
git status --short
```

The sequence is a template, not a substitute for reviewing the staged diff. Inspect the code repository's GitHub commit/files view after pushing. For companion-repository changes, repeat the corresponding checks in that repository and identify both SHAs in the report.

### 16.2 Resume after a terminal failure or a new session

First read `progress.md`, then the relevant part of `plan.md`, `review.md`, and `validation.md`. Confirm the current worktree, branch, recorded base, local status, recent commits, upstream branch, and remote HEAD. Inspect pending changes before deciding whether a step completed.

Treat code edits, tests, commits, and pushes as distinct stages. A completed test with no commit requires reviewing and committing the tested diff; a local commit with a failed push requires publication, not reimplementation. A test result from different code or a changed native environment cannot be reused without checking the difference.

Do not delete partial work, submit duplicate GPU jobs, recreate the branch, or rerun a large task solely because the terminal disconnected. Inspect the recorded native job and existing artifacts first. Keep the next action specific, for example: “Run the real mixed-source lifecycle test after the ownership change, then commit and push Step 3.”

Before stopping, leave the exact current state and next command/action in `progress.md`, publish the checkpoint when possible, and send the required user-facing update. Do not promise unattended work after the session ends.

## 17. Source index

These are reading anchors for the execution agent. They establish the reviewed snapshot and candidate intent; they are not test results for the future PR branch. Use pinned file/symbol links in the new task's review records after rechecking references.

**[R1] Current-main repository and contributor guidance**

- `https://github.com/NetX-lab/Frontier/tree/1f694f7c549aa3aeeb7c5bbae04e119c09167a77`
- `https://github.com/NetX-lab/Frontier/blob/1f694f7c549aa3aeeb7c5bbae04e119c09167a77/AGENTS.md`
- `https://github.com/NetX-lab/Frontier/blob/1f694f7c549aa3aeeb7c5bbae04e119c09167a77/pyproject.toml`

**[R2] Candidate branch and task archive**

- `https://github.com/NetX-lab/Frontier/tree/a7b3320fe9b8b083ee86b91dae3d6838f4443d91`
- `https://github.com/NetX-lab/Frontier/tree/a7b3320fe9b8b083ee86b91dae3d6838f4443d91/task_memory/task_2026-09-07_issue26_ttft_h200`

**[R3] User-designated vLLM reference**

- Branch: `https://github.com/fwyc0573/vLLM-BS/tree/feature/frontier-comparison-instrumentation`
- Reviewed source: `https://github.com/fwyc0573/vLLM-BS/tree/ea95f571e20937c7c908c6d59ddd1cd6bf9268f1`
- Read the files listed in Section 7.2 at this pinned revision. Record the verified 0.10.2 relationship and any material fork changes.

**[R4] Candidate design and recovery documents**

Under the pinned task archive in [R2]: `design_dp_load_balancing.md`, `design_shared_forward_sync.md`, `review_shared_forward_sync.md`, `summary.md`, and `handoff.md`. Prefer the focused design documents over the chronological experiment history.

**[R5] Ignore policy affecting reviewable records**

- `https://github.com/NetX-lab/Frontier/blob/1f694f7c549aa3aeeb7c5bbae04e119c09167a77/.gitignore`

**[R6] Candidate shared-forward and DP tests**

Under `tests/unit/` at [R2]: `test_cluster_scheduler_dp_lanes.py`, `test_monolithic_mixed_forward_sync.py`, and `test_vllm_dp_load_balancer.py`. The donor's `tests/integration/issue26_dp_coordinator_reference.py` is a possible reference-test helper, subject to review.

**[R7] Candidate MoE and routing tests**

Under `tests/unit/` at [R2]: `test_moe_fused_expert_numerical_parity.py`, `test_moe_routing_runtime.py`, and `test_moe_routing_runtime_model_sharing.py`.

**[R8] Main profiling and reporting integration anchors**

Under [R1]: `frontier/profiling/moe/moe_vllm_kernel.py`, `frontier/entities/stage_execution_time.py`, `frontier/scheduler/utils/prefill_collective.py`, `frontier/scheduler/utils/decode_collective.py`, and `tests/unit/test_moe_fused_event_contract.py`.

**[R9] Optional backend and historical zero-payload evidence**

- Parent configuration: `https://github.com/NetX-lab/Frontier/blob/1f694f7c549aa3aeeb7c5bbae04e119c09167a77/.gitmodules`
- Configured repository: `https://github.com/fwyc0573/frontier-htsim`
- Candidate report: `task_memory/task_2026-09-07_issue26_ttft_h200/test_report_2026-09-08_collective_zero_payload.md` at [R2].
- Candidate parent regression: `tests/unit/test_collective_sim_zero_payload.py` at [R2].

---

**First action for a fresh Claude Code session:** perform Step 0, publish the new worktree branch and tracked plan, then complete the source audit in Step 1. Do not begin by cherry-picking the calibration branch or launching a GPU job.
