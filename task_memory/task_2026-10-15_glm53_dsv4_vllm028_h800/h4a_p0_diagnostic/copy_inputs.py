"""Copy the small H4a native inputs out of /data/ycfeng/tmp (D-INPUT-PERSISTENCE).

Copies, never moves, and never overwrites: an existing copy must match its source. shutil.copy2 keeps each file's mtime, which
client_metrics.native_client_rows checks (run files must be newer than run_manifest.json).
Writes inputs/manifest.json with each file's source path, cloud origin, bytes and SHA-256.
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = Path("/data/ycfeng/tmp/glm53_dsv4")
RUN_FILES = ["run_manifest.json", "replay_finished.json", "run/canonical_repetitions.json",
             "run/client_requests.jsonl", "run/proxy.log", "run/proxy_requests.jsonl", "run/replay_summary.json"]
# The TP0 rank of each decode DP lane; the step-cost builder reads only these traces.
PROFILE02_DECODE_TRACES = [
    "torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792766820606124725.pt.trace.json.gz",
    "torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792766798163291423.pt.trace.json.gz",
]
SOURCES = {
    "native06": dict(
        root=TMP / "native06_case_20261023/runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06",
        files=RUN_FILES + ["launcher_receipt.json", "platform_terminal_receipt.json"],
        extra={"run_check_status.groundtruth_clean.json":
               TMP / "native06_case_20261023/analysis/run_check_status.groundtruth_clean.json"},
        cloud_origin=None),
    "profile02": dict(
        root=TMP / "profile02_results_readonly_7ba56n6z/artifacts",
        files=RUN_FILES + PROFILE02_DECODE_TRACES,
        extra={"download_receipt.json": TMP / "profile02_results_readonly_7ba56n6z/receipt.json",
               "cloud_inventory.json": TMP / "profile02_results_readonly_7ba56n6z/remote_inventory.json"},
        cloud_origin="/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261023/"
                     "case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261023-02"),
    "profile04": dict(
        root=TMP / "profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04",
        files=RUN_FILES,
        extra={"download_receipt.json": TMP / "profile04_results_1u00cy05/receipt.json",
               "cloud_inventory.json": TMP / "profile04_results_1u00cy05/inventory.json"},
        cloud_origin="/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261024_04"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = {}
    for name, source in SOURCES.items():
        pairs = [(source["root"] / f, f) for f in source["files"]] + [(p, f) for f, p in source["extra"].items()]
        entries = []
        for src, relative in pairs:
            dst = HERE / "inputs" / name / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            digest = sha256(dst)
            if digest != sha256(src) or dst.stat().st_mtime != src.stat().st_mtime:
                raise SystemExit(f"Copy mismatch: {src}")
            entries.append(dict(path=relative, source=str(src), bytes=dst.stat().st_size, sha256=digest))
        manifest[name] = dict(source_root=str(source["root"]), cloud_origin=source["cloud_origin"], files=entries)
    (HERE / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print({name: len(value["files"]) for name, value in manifest.items()})


if __name__ == "__main__":
    main()
