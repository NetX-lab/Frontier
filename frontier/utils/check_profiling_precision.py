"""CLI utility for validating profiling precision metadata in CSV files.

Example:
    $ python -m frontier.utils.check_profiling_precision \\
        --data_path data/profiling/compute/a800
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Any

from frontier.logger import init_logger

logger = init_logger(__name__)


def scan_profiling_files(data_path: str) -> Dict[str, Any]:
    """Scan a profiling directory for CSV files with precision metadata.

    Args:
        data_path: Root directory containing profiling CSV files.

    Returns:
        A structured report with per-file precision metadata and recommendations.

    Raises:
        FileNotFoundError: If data_path does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Profiling data path does not exist: {data_path}")

    profiling_files: List[str] = []
    for root, _, files in os.walk(data_path):
        for name in files:
            if name.endswith(".csv"):
                profiling_files.append(os.path.join(root, name))

    report = {
        "data_path": data_path,
        "total_files_scanned": len(profiling_files),
        "files_with_precision": 0,
        "files_without_precision": 0,
        "files": [],
        "recommendations": [],
    }

    for file_path in profiling_files:
        with open(file_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            has_precision = "profiling_precision" in header
            precision_values = set()
            if has_precision:
                precision_index = header.index("profiling_precision")
                for row in reader:
                    if len(row) <= precision_index:
                        continue
                    value = row[precision_index].strip()
                    if value:
                        precision_values.add(value)
            report["files"].append(
                {
                    "path": file_path,
                    "has_precision_column": has_precision,
                    "precision_values": sorted(precision_values),
                }
            )

        if has_precision:
            report["files_with_precision"] += 1
        else:
            report["files_without_precision"] += 1

    if report["files_without_precision"] > 0:
        report["recommendations"].append(
            "Add profiling_precision column to missing CSV files or re-run profiling with --precision."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check profiling CSV files for precision metadata."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to profiling data directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON report.",
    )
    args = parser.parse_args()

    report = scan_profiling_files(args.data_path)

    logger.info(
        "Scanned %d files: %d with precision, %d without precision",
        report["total_files_scanned"],
        report["files_with_precision"],
        report["files_without_precision"],
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Wrote report to %s", args.output)

    if report["files_without_precision"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
