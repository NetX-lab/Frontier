"""Copy the small native inputs of one input set out of /data/ycfeng/tmp (D-INPUT-PERSISTENCE).

    python copy_inputs.py h800_p0    # H4a: native06, profile02, profile04 -> inputs/
    python copy_inputs.py h200_p2    # H200 mainline: P2-32/P2-64 references and profiles, P2-32 lifecycle -> ../h200_mainline/diagnostic/inputs/

Copies, never moves, and never overwrites: an existing copy must match its source. shutil.copy2 keeps each file's mtime, which
client_metrics.native_client_rows checks (run files must be newer than run_manifest.json).
Writes <set dir>/manifest.json with each file's source path, cloud origin, bytes and SHA-256.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = Path("/data/ycfeng/tmp/glm53_dsv4")
H200 = TMP / "h200"
T18 = "/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18"
RUN_FILES = ["run_manifest.json", "replay_finished.json", "run/canonical_repetitions.json",
             "run/client_requests.jsonl", "run/proxy.log", "run/proxy_requests.jsonl", "run/replay_summary.json"]
# The TP0 rank of each decode DP lane; the step-cost builder reads only these traces.
PROFILE02_DECODE_TRACES = [
    "torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792766820606124725.pt.trace.json.gz",
    "torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792766798163291423.pt.trace.json.gz",
]
H800_P0 = {
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



def h200_reference(case, tag, node_files=()):
    root = H200 / "cases" / case / "runs/groundtruth_clean" / tag
    return dict(root=root, files=RUN_FILES + ["launcher_receipt.json", "platform_terminal_receipt.json"] + list(node_files),
                extra={"run_check_status.groundtruth_clean.json":
                       H200 / "cases" / case / "analysis/run_check_status.groundtruth_clean.json",
                       "download_receipt.json": H200 / "downloads" / tag / "download_receipt.json",
                       "cloud_inventory.json": H200 / "downloads" / tag / "cloud_inventory.json"},
                cloud_origin=f"{T18}/{tag.replace('-', '_')}/case/runs/groundtruth_clean/{tag}")


def h200_profile(tag, decode_traces):
    return dict(root=H200 / "downloads" / tag / "artifacts", files=RUN_FILES + decode_traces,
                extra={"download_receipt.json": H200 / "downloads" / tag / "download_receipt.json",
                       "cloud_inventory.json": H200 / "downloads" / tag / "cloud_inventory.json"},
                cloud_origin=f"{T18}/{tag.replace('-', '_')}/case/runs/groundtruth_clean/{tag}")


# Reference run files come from the R4 evaluation cases, whose run_manifest.json is the package copy with its original mtime
# (the SSH tar stream keeps only whole-second mtimes); the other run files are byte-identical to the hash-verified downloads.
H200_P2 = {
    "p2_32r": h200_reference("p2_32r_case_20261025", "dsv4-h200-p2-canonical-20261025-01"),
    "p2_32p": h200_profile("dsv4-h200-p2-profile-20261025-01", [
        "torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942701901162313.pt.trace.json.gz",
        "torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792942680280151453.pt.trace.json.gz",
    ]),
    "p2_64r": h200_reference("p2_64r_case_20261025", "dsv4-h200-p2-64-canonical-20261025-01"),
    # H4b-(a) on H200: the 4 engine-core lifecycle files, and the stop markers and the decode headless node's result,
    # which hold the shutdown-order evidence behind its R4 verdict.
    "p2_32l": h200_reference("p2_32l_case_20261025", "dsv4-h200-p2-lifecycle-20261025-01", [
        "nodes/gpu-h200-0043.lgcm.sh.istep.fun/scheduler_lifecycle/dp0_pid1418.jsonl",
        "nodes/gpu-h200-0628.lgcm.sh.istep.fun/scheduler_lifecycle/dp1_pid1277.jsonl",
        "nodes/gpu-h200-0705.lgcm.sh.istep.fun/scheduler_lifecycle/dp0_pid1419.jsonl",
        "nodes/gpu-h200-1191.lgcm.sh.istep.fun/scheduler_lifecycle/dp1_pid1358.jsonl",
        *(f"nodes/gpu-h200-{host}.lgcm.sh.istep.fun/native_stopped.json" for host in ("0043", "0628", "0705", "1191")),
        "nodes/gpu-h200-1191.lgcm.sh.istep.fun/result.json",
    ]),
    "p2_64p": h200_profile("dsv4-h200-p2-64-profile-20261025-01", [
        "torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792946239761662380.pt.trace.json.gz",
        "torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792946280760554510.pt.trace.json.gz",
        "torch_traces/decode/dp2/dp2_pp0_tp0_dcp0_ep16_rank0.1792946183741136595.pt.trace.json.gz",
        "torch_traces/decode/dp3/dp3_pp0_tp0_dcp0_ep24_rank0.1792946205265552185.pt.trace.json.gz",
    ]),
}
INPUT_SETS = {"h800_p0": (HERE / "inputs", H800_P0),
              "h200_p2": (HERE.parent / "h200_mainline" / "diagnostic" / "inputs", H200_P2)}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(input_set: str) -> None:
    target, sources = INPUT_SETS[input_set]
    manifest = {}
    for name, source in sources.items():
        pairs = [(source["root"] / f, f) for f in source["files"]] + [(p, f) for f, p in source["extra"].items()]
        entries = []
        for src, relative in pairs:
            dst = target / name / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            digest = sha256(dst)
            if digest != sha256(src) or dst.stat().st_mtime != src.stat().st_mtime:
                raise SystemExit(f"Copy mismatch: {src}")
            entries.append(dict(path=relative, source=str(src), bytes=dst.stat().st_size, sha256=digest))
        manifest[name] = dict(source_root=str(source["root"]), cloud_origin=source["cloud_origin"], files=entries)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print({name: len(value["files"]) for name, value in manifest.items()})


if __name__ == "__main__":
    if sys.argv[1:] not in [[name] for name in INPUT_SETS]:
        raise SystemExit(__doc__)
    main(sys.argv[1])
