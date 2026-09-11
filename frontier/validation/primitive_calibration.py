"""Report held-out primitive calibration without using full-forward timings."""

import argparse
import json
from pathlib import Path

from frontier.runtime_cost.primitives import PrimitiveCalibration
from frontier.runtime_cost.sglang import RuntimeIdentity


def read_profiles(path):
    return [json.loads(p.read_text()) for p in sorted(Path(path).glob("rank*.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = read_profiles(args.calibration_dir)
    if not payloads:
        raise ValueError("No calibration rank profiles")
    identity = RuntimeIdentity(**payloads[0]["identity"])
    calibration = PrimitiveCalibration(payloads, identity=identity)
    report = calibration.validate(read_profiles(args.validation_dir))
    report["calibration_rows"] = calibration.rows
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("all_primitives_passed", "calibration_quality", "validation")}, indent=2))


if __name__ == "__main__":
    main()
