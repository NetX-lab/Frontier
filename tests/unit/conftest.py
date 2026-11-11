"""Put the analysis helpers under ``tests/analysis`` on the import path.

Several unit modules reuse those helpers by name, the same way the analysis
scripts import each other. Resolving the directory here keeps every module
collectible on its own, instead of depending on whichever test module ran first
and inserted the path as a side effect of its own import.
"""

from pathlib import Path
import sys

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
