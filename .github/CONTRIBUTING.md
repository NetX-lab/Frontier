# Contributing to Frontier

## Questions and proposals

Search [existing issues](https://github.com/NetX-lab/Frontier/issues), then
[open an issue](https://github.com/NetX-lab/Frontier/issues/new/choose) using the
bug report, feature request, or usage question form. All three belong in Issues.
If a question turns out to be a bug, continue tracking it in the same issue.
Close the issue when it is resolved, with a brief answer or a link to the fix.

For usage guidance, see the [CLI](../docs/cli/README.md),
[profiling](../docs/profiling/README.md), and
[predictor training](../docs/training/README.md) guides.

## Pull requests

Small fixes can go directly to a pull request; linking an issue is optional.
For substantial behavior changes, open an issue first to agree on scope and
design. Keep each PR focused on one purpose.

Use the PR template to explain the problem, the changes, and validation.
Include exact commands and observed results. State any checks you did not run
and why. Update relevant documentation and provide a runnable test scenario for
new behavior, following the [project guidance](../AGENTS.md).

If simulation behavior or predicted values change, provide the scenario and
configuration, before/after results with units, and the reason for the change.
When comparing against measurements, identify their source and distinguish
measured values from estimates. Passing basic tests does not establish
prediction accuracy.

## Manual validation

Choose checks that cover the change. The [test guide](../AGENTS.md#tests)
describes available suites and environment requirements. After installing the
project's test dependencies, these are starting points from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py tests/e2e/test_colocation_examples_layout.py -q
bash examples/architecture/run_dummy_smoke_matrix.sh
```

The first test command checks public examples and documentation contracts;
the dummy smoke matrix exercises simulation paths without measured execution
times. Neither replaces checks specific to a scheduler, predictor, or numerical
change. These commands are run manually by contributors or maintainers.

## Review and merge

External contributions require maintainer review before merging. Maintainers
may self-review and merge their own PRs after recording validation. Independent
review is recommended for high-impact changes, especially changes to simulation
results or public behavior. Resolve open design questions before merging.

These are contribution conventions; they do not configure branch protection
or automated checks.
