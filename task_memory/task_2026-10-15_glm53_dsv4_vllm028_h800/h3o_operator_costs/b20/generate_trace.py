"""Generate the pinned eight-request B20 identity workload."""
import hashlib
from pathlib import Path
import sys


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "trace.csv"
    path.write_text("arrived_at,num_prefill_tokens,num_decode_tokens\n" +
                    "".join(f"{i * 0.05:.2f},6144,48\n" for i in range(8)))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    path, digest = generate(Path(sys.argv[1]))
    print(path, digest)
