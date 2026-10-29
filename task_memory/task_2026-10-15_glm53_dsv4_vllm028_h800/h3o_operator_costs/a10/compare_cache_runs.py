"""A9: exact artifacts and cache identities for a cold/warm pair of A10 runs.

Usage: compare_cache_runs.py <cold run root> <warm run root>
"""

import json
import sys
from pathlib import Path

from a10_common import REPO, cache_inventory, load_module

compare = load_module("refactor_fidelity_compare", REPO / "tests/e2e/refactor_fidelity/compare.py")


def run_paths(root):
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt["status"] != "FINISHED":
        raise ValueError(f"A9 needs a finished run: {root} status={receipt['status']}")
    # Rebase paths to the supplied root, allowing scratch copies as well as
    # original runs. MetricsConfig nests metrics below its initial root.
    original = Path(receipt["run_root"])
    metrics = root / Path(receipt["metrics_dir"]).relative_to(original)
    cache = root / Path(receipt["cache_dir"]).relative_to(original)
    return receipt, metrics, cache


def run_substitutions(root, receipt):
    # A scratch copy still embeds the original absolute root in config.json.
    substitutions = [*compare.substitutions_for(REPO, root, "metrics"),
                     *compare.substitutions_for(REPO, Path(receipt["run_root"]), "metrics")]
    return sorted(substitutions, key=lambda item: len(item.literal), reverse=True)


def differences(cold_root, warm_root):
    cold, cold_metrics, cold_cache = run_paths(cold_root)
    warm, warm_metrics, warm_cache = run_paths(warm_root)
    found = [f"metrics/{item.artifact}: {item.kind}: {item.detail}" for item in compare.compare_artifact_directories(
        cold_metrics, warm_metrics,
        run_substitutions(cold_root, cold), run_substitutions(warm_root, warm),
    )]
    inventories = []
    for label, directory, receipt in (("cold", cold_cache, cold), ("warm", warm_cache, warm)):
        if not directory.is_dir():
            found.append(f"{label}/cache: missing directory {directory}")
        inventory = cache_inventory(directory)
        inventories.append(inventory)
        expected = {entry["name"]: entry for entry in receipt["cache_files"]}
        actual = {entry["name"]: entry for entry in inventory}
        for name in sorted(set(expected) | set(actual)):
            if expected.get(name) != actual.get(name):
                found.append(f"{label}/cache/{name}: differs from the end-of-run receipt")
    cold_names, warm_names = ([entry["name"] for entry in inventory] for inventory in inventories)
    found.extend(f"cache/{item.stem}: {item.kind}: {item.detail}"
                 for item in compare.classify_cache_differences(cold_names, warm_names))
    # Report each changed name even when the shared classifier groups by stem.
    found.extend(f"cache/{name}: present only in cold" for name in sorted(set(cold_names) - set(warm_names)))
    found.extend(f"cache/{name}: present only in warm (warm cache added an artifact)"
                 for name in sorted(set(warm_names) - set(cold_names)))
    cold_hashes, warm_hashes = ({entry["name"]: entry.get("sha256") for entry in inventory}
                              for inventory in inventories)
    found.extend(f"cache/{name}: SHA-256 differs" for name in sorted(set(cold_names) & set(warm_names))
                 if cold_hashes[name] != warm_hashes[name])
    return found


def main(cold_root, warm_root):
    found = differences(cold_root.resolve(), warm_root.resolve())
    if found:
        print("\n".join(found))
        return 1
    print("EQUAL: metrics and cache identities match; warm cache file set equals cold at run end.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
