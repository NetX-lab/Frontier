"""Regression guard: sibling BaseRequestGeneratorConfig subclasses that reuse the
same nested BasePolyConfig field name collide in SimulationConfig's flattened CLI
namespace (create_flat_dataclass keys a poly field's --<field>_type flag by bare
field name, not by owning subclass) and crash at CLI-parser construction time for
every run, not just the offending generator's. Nothing else in the suite catches
this class of bug."""
from __future__ import annotations

from frontier.config.config import SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass


def test_simulation_config_flattens_without_field_collisions() -> None:
    create_flat_dataclass(SimulationConfig)
