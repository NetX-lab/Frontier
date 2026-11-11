"""Put the shared test helpers on the import path before collection.

Several unit modules reuse the helpers under ``tests/analysis`` and
``tests/comparison/calibration`` by name, the same way the analysis and
calibration scripts import each other. Resolving both directories here keeps
every module collectible on its own, instead of depending on whichever test
module ran first and inserted the path as a side effect of its own import.
"""

from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parents[1]
for helpers in (TESTS / "analysis", TESTS / "comparison/calibration"):
    if str(helpers) not in sys.path:
        sys.path.insert(0, str(helpers))
