"""Distributed GDN examples must launch one process per tensor-parallel rank."""

import pytest

from tests.unit.test_rocm_documented_commands import ROOT, commands


@pytest.mark.parametrize("document", ["docs/profiling/ROCM_MI355X.md", "docs/profiling/README.md", "docs/cli/README.md"])
def test_gdn_distributed_example_matches_tp_ranks(document):
    selected = [cmd for cmd in commands(ROOT / document) if "frontier.profiling.gdn.main" in cmd]
    assert selected
    for command in selected:
        tp = int(command[command.index("--tensor-parallel-size") + 1])
        if tp > 1:
            assert command[0] == "torchrun"
            assert f"--nproc-per-node={tp}" in command
            assert "--standalone" in command
        else:
            assert command[0] in {"python", "python3", "torchrun"}
