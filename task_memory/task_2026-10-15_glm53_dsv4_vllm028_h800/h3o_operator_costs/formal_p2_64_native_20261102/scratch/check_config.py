"""CPU environment/schema probe; config writes mocked; not R3 admission."""
from pathlib import Path
import json
from unittest.mock import patch
from frontier.config import SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass

try:
    flat = json.loads(Path('/data/ycfeng/tmp/codex-implement/54cbb041-c825-4706-b25e-75530b995d23/formal_dry_final/case/inputs/frontier.json').read_text())
    with patch.object(SimulationConfig, 'write_config_to_file', return_value=None):
        resolved = create_flat_dataclass(SimulationConfig)(**flat).reconstruct_original_dataclass()
    print('EXISTING_PY312_FRONTIER_CONFIG_IMPORT_AND_RECONSTRUCTION_PASS')
except Exception as error:
    print(f"UNVERIFIED {type(error).__name__}: {error}")
