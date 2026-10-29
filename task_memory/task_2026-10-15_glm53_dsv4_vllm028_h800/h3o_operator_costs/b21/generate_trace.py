"""Write the agreed eight-request observer identity workload in a run root."""
import argparse
import csv
import hashlib
from pathlib import Path


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "trace.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        writer.writerows((f"{i * .05:.2f}", 6144, 48) for i in range(8))
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(generate(parser.parse_args().root))
