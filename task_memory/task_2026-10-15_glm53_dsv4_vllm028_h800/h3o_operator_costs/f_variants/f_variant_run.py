"""Run one A10 replay with extra flat fields, for the F1-F4 comparison variants.

Usage: f_variant_run.py <case> <output root> [field=value ...]   # case p2_32 or p2_64

Each field=value sets one flat SimulationConfig field (JSON, else a string),
parsed by h4a4.parse_field; a10_driver.main records them in receipt.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a10"))
import a10_driver  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in a10_driver.CASES or not all("=" in a for a in sys.argv[3:]):
        raise SystemExit(__doc__)
    fields = dict(a10_driver.h4a4.parse_field(argument) for argument in sys.argv[3:])
    a10_driver.main(sys.argv[1], Path(sys.argv[2]), fields=fields)
