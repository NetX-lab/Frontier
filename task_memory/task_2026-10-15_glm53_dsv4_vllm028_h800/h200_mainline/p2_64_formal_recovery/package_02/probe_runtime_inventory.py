"""Record the runtime inventory of the selected image on each node of a multi-node job.

Per node: dependency and system inventory, offline config/tokenizer resolution of
both models behind a network guard, an intra-node NCCL all-reduce, a cross-node
NCCL all-reduce, a cross-node NIXL READ of GPU memory, and a weight-file audit.
Each phase records its own result, so one failing phase does not hide the others.
The shared cloud directory carries barriers and retained evidence; NIXL agent
metadata is exchanged over the already validated TCP side channel.
"""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng")
SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "AUTH", "CREDENTIAL", "COOKIE")
RECORDED_ENV = re.compile(
    r"^(NCCL_|UCX_|NVIDIA_|CUDA_|VLLM_|HF_|TRANSFORMERS_|TORCH|PYTORCH_|GLOO_|OMPI_|PMIX_|"
    r"MASTER_|NODE_|LD_LIBRARY_PATH$|PATH$|RANK$|WORLD_SIZE$|LOCAL_RANK$|HOSTNAME$)")
NCCL_DEBUG_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH,ENV"}

NETWORK_GUARD = '''
import json, os, socket, time
_log = os.environ["T01_NETWORK_GUARD_LOG"]
_local_hosts = {"127.0.0.1", "::1", "localhost", "", None}

def _record(kind, target):
    with open(_log, "a") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "kind": kind, "target": repr(target),
                                 "time": time.time()}) + "\\n")

def _is_local(host):
    return host in _local_hosts or str(host).startswith("127.")

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo

def _is_remote(sock, address):
    return sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0])

def connect(self, address):
    if _is_remote(self, address):
        _record("connect", address)
        raise OSError("offline-resolution guard blocked a network connection")
    return _connect(self, address)

def connect_ex(self, address):
    if _is_remote(self, address):
        _record("connect_ex", address)
        raise OSError("offline-resolution guard blocked a network connection")
    return _connect_ex(self, address)

def getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        _record("getaddrinfo", host)
        raise socket.gaierror("offline-resolution guard blocked a name lookup")
    return _getaddrinfo(host, *args, **kwargs)

socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def run(argv, timeout=120):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"argv": argv, "returncode": None, "error": "command not found"}
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def mapped_libraries(fragment):
    return sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if fragment in line and "/" in line})


def weight_candidates(roots, suffixes):
    def scan_error(error):
        raise error

    candidates = {}
    for root in roots:
        if not Path(root).exists():
            continue
        for directory, _, files in os.walk(root, onerror=scan_error):
            for name in files:
                if name.lower().endswith(suffixes):
                    path = Path(directory) / name
                    stat = path.stat()
                    candidates[str(path)] = {
                        "resolved_path": str(path.resolve()), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                        "inode": stat.st_ino}
    return dict(sorted(candidates.items()))


def wait_for(condition, seconds, interval=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(interval)
    return condition()


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def primary_ipv4():
    # A UDP connect selects the outgoing interface without sending a packet.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]


# ---------------------------------------------------------------- child roles

def role_imports(args):
    result = {"distributions": sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())}
    sitecustomize = importlib.util.find_spec("sitecustomize")
    result["sitecustomize"] = sitecustomize.origin if sitecustomize else None
    vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
    third_party = vllm_root / "third_party"
    result["vllm_files"] = {
        "root": str(vllm_root),
        "shared_objects": sorted(str(p.relative_to(vllm_root)) for p in vllm_root.rglob("*.so")),
        "third_party": sorted(p.name for p in third_party.iterdir()) if third_party.is_dir() else None,
        "version_file": read_text(vllm_root / "_version.py"),
    }
    import torch

    result["libnccl_after_torch_import"] = mapped_libraries("libnccl")
    nccl_paths = [path for path in result["libnccl_after_torch_import"]
                  if re.fullmatch(r"libnccl(?:-[0-9a-f]+)?\.so(?:\..+)?", Path(path).name)]
    if len(nccl_paths) != 1:
        raise RuntimeError(f"Expected one torch-mapped NCCL library, found {nccl_paths}")
    result["nccl_runtime_library"] = nccl_paths[0]
    nccl_get_version = ctypes.CDLL(nccl_paths[0]).ncclGetVersion
    nccl_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    nccl_get_version.restype = ctypes.c_int
    nccl_raw = ctypes.c_int()
    result["nccl_runtime_returncode"] = nccl_get_version(ctypes.byref(nccl_raw))
    result["nccl_runtime_raw_version"] = nccl_raw.value
    modules = {}
    # vLLM resolves DeepGEMM as `deep_gemm` first, then `vllm.third_party.deep_gemm`.
    for name in ("torch", "triton", "flashinfer", "tilelang", "transformers", "nixl",
                 "deep_ep", "deep_gemm", "vllm.third_party.deep_gemm", "vllm"):
        try:
            module = importlib.import_module(name)
            modules[name] = {key: str(getattr(module, key, None))
                             for key in ("__file__", "__version__", "git_version", "__commit__")}
        except Exception as error:  # noqa: BLE001 - record every import failure
            modules[name] = {"import_error": repr(error)}
    if "import_error" not in modules["deep_ep"]:
        import deep_ep

        package_parent = Path(deep_ep.__file__).parent.parent
        modules["deep_ep"]["has_ElasticBuffer"] = hasattr(deep_ep, "ElasticBuffer")
        modules["deep_ep"]["installed_files"] = sorted(
            p.name for p in package_parent.iterdir() if p.name.lower().startswith("deep_ep"))
    result["modules"] = modules
    result["torch"] = {"cuda": torch.version.cuda, "nccl_compiled": torch.cuda.nccl.version(),
                       "device_count": torch.cuda.device_count(),
                       "device_name": torch.cuda.get_device_name(0)}
    squares = torch.arange(16, device="cuda", dtype=torch.float32)
    result["cuda_sum_of_squares"] = (squares * squares).sum().item()
    ones = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
    result["cublas_bf16_matmul"] = bool(((ones @ ones) == 64).all().item())
    result["libcuda_loaded"] = mapped_libraries("libcuda.so")
    driver_api = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(driver_api))
    result["cuda_driver_api_version"] = driver_api.value
    result["libnccl_loaded"] = mapped_libraries("libnccl")
    result["libnvshmem_loaded"] = mapped_libraries("libnvshmem")
    result["binary_sha256"] = {
        path: sha256(path) for path in sorted(set(
            result["libcuda_loaded"] + result["libnccl_loaded"]
            + result["libnvshmem_loaded"]
            + [str(vllm_root / name) for name in result["vllm_files"]["shared_objects"]]))}
    write_json(args.out, result)


def role_resolve(args):
    import huggingface_hub.constants

    result = {"label": args.label, "model_dir": args.model_dir, "trust_remote_code": False,
              "hf_hub_offline_constant": huggingface_hub.constants.HF_HUB_OFFLINE}
    from vllm.engine.arg_utils import EngineArgs
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.transformers_utils.config import get_config

    hf_config = get_config(args.model_dir, trust_remote_code=False)
    model_config = EngineArgs(model=args.model_dir, trust_remote_code=False, dtype="bfloat16",
                              load_format="dummy").create_model_config()
    tokenizer = cached_tokenizer_from_config(model_config)
    sample_ids = list(tokenizer.encode("Hello, world."))
    result["resolved"] = True
    result["hf_config"] = attributes(hf_config, ("model_type", "architectures"))
    result["hf_config"]["class"] = type(hf_config).__name__
    result["model_config"] = attributes(model_config, (
        "architecture", "architectures", "dtype", "quantization", "max_model_len",
        "tokenizer_mode", "use_mla", "is_deepseek_mla"))
    result["model_config"]["num_hidden_layers"] = attributes(
        model_config.hf_text_config, ("num_hidden_layers",))["num_hidden_layers"]
    result["tokenizer"] = {"class": type(tokenizer).__name__, "length": len(tokenizer),
                           "sample_ids": sample_ids}
    write_json(args.out, result)


def attributes(value, names):
    # Introspection failures are recorded per field; they do not undo a completed resolution.
    recorded = {}
    for name in names:
        try:
            recorded[name] = getattr(value, name)
        except Exception as error:  # noqa: BLE001
            recorded[name] = f"<error {error!r}>"
    return recorded


def role_nccl(args):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group("nccl", init_method=args.init_method, rank=args.rank,
                            world_size=args.world, device_id=device)
    elements = args.mib * 1024 * 1024 // 4
    values = torch.full((elements,), float(args.rank + 1), device=device)
    dist.all_reduce(values)
    expected = args.world * (args.world + 1) / 2
    correct = bool(torch.all(values == expected).item())
    timing = torch.zeros(elements, device=device)
    for _ in range(3):
        dist.all_reduce(timing)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        dist.all_reduce(timing)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / args.iterations
    algorithm_bandwidth = elements * 4 / seconds / 1e9
    write_json(args.out, {
        "rank": args.rank, "world": args.world, "local_rank": args.local_rank,
        "correct": correct, "expected": expected, "message_mib": args.mib,
        "seconds_per_iteration": seconds, "algorithm_gb_per_s": algorithm_bandwidth,
        "bus_gb_per_s": algorithm_bandwidth * 2 * (args.world - 1) / args.world,
        "libnccl_loaded": mapped_libraries("libnccl"),
        "nccl_compiled": torch.cuda.nccl.version()})
    dist.destroy_process_group()


def role_nixl(args):
    # Reuse the causally tested READ probe, including its TCP metadata handoff.
    from probe_nixl_read import run as run_nixl_read

    config = json.loads(args.config.read_text())
    output = Path(config["output_dir"]) / "nixl_exchange"
    read_config = {**config, "output_dir": str(output), "device": 0,
                   "transfer_timeout_seconds": config["phase_timeout_seconds"]}
    code = run_nixl_read(read_config)
    result = json.loads((output / "nodes" / socket.gethostname() / "result.json").read_text())
    write_json(args.out, result)
    return code


# ------------------------------------------------------------ node orchestration

class Node:
    def __init__(self, config_path, config):
        self.config_path = config_path
        self.config = config
        self.output = Path(config["output_dir"])
        if not self.output.is_relative_to(CLOUD_ROOT):
            raise ValueError("Probe output must be inside the personal cloud directory")
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob("nodes/*/receipt.json")):
            raise RuntimeError(f"{self.output} already holds receipts from an earlier job")
        self.host = socket.gethostname()
        self.node_dir = self.output / "nodes" / self.host
        self.node_dir.mkdir(parents=True, exist_ok=False)
        self.work = Path(tempfile.mkdtemp(prefix="t01_inventory_"))
        self.metadata_root = Path(config["metadata_root"])
        self.receipt = {"host": self.host, "image_requested": config["image"],
                        "python": sys.version, "started": time.time(),
                        "checks": {}, "phases": {}}
        self.node_rank = None
        self.master_ip = None
        manifest = json.loads((self.metadata_root / "metadata_manifest.json").read_text())
        self.weight_suffixes = tuple(manifest["weight_suffixes_refused"])
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        hf_home = Path(os.environ.get("HF_HOME", cache_home / "huggingface"))
        torch_home = Path(os.environ.get("TORCH_HOME", cache_home / "torch"))
        self.cache_roots = sorted({str(p) for p in (hf_home, torch_home / "hub")} | {
            os.environ[key] for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                                       "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                                       "PYTORCH_PRETRAINED_BERT_CACHE") if os.environ.get(key)})
        self.cache_candidates_before = weight_candidates(self.cache_roots, self.weight_suffixes)
        write_json(self.node_dir / "download_audit_before.json",
                   {"roots": self.cache_roots, "candidates": self.cache_candidates_before})

    def child(self, role_args, log_name, env=None):
        argv = [sys.executable, str(Path(__file__).resolve()), "--config", str(self.config_path),
                *role_args]
        with open(self.node_dir / log_name, "w") as log:
            return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **(env or {})})

    def finish(self, processes, label):
        timeout = self.config["phase_timeout_seconds"]
        deadline = time.monotonic() + timeout
        codes = []
        for process in processes:
            try:
                codes.append(process.wait(timeout=max(1.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                process.kill()
                codes.append(f"killed after {timeout} s")
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{label} child exit codes {codes}")
        return codes

    def phase(self, name, function):
        start = time.time()
        try:
            self.receipt["phases"][name] = {"result": function()}
        except Exception:  # noqa: BLE001 - each phase records its own failure
            self.receipt["phases"][name] = {"exception": traceback.format_exc()}
        self.receipt["phases"][name]["seconds"] = time.time() - start
        write_json(self.node_dir / "receipt.partial.json", self.receipt)

    def check(self, name, value):
        self.receipt["checks"][name] = bool(value)

    # -- phases

    def source_and_metadata(self):
        source = Path(self.config["workspace"]) / "frontier" / "main.py"
        source_sha = sha256(source)
        self.check("source_mount_readable", source_sha == self.config["expected_source_sha256"])
        manifest = json.loads((self.metadata_root / "metadata_manifest.json").read_text())
        mismatches, extra = [], []
        for repo, entry in manifest["models"].items():
            model_dir = self.metadata_root / repo.replace("/", "__") / entry["revision"] / "model"
            expected = {f["path"]: f for f in entry["files"] if f["role"] == "model"}
            for path, item in expected.items():
                actual = model_dir / path
                if not actual.is_file() or sha256(actual) != item["sha256"]:
                    mismatches.append(f"{repo}/{path}")
            extra += [f"{repo}/{p.relative_to(model_dir)}" for p in model_dir.rglob("*")
                      if p.is_file() and str(p.relative_to(model_dir)) not in expected]
        self.check("model_metadata_hashes_match", not mismatches and not extra)
        return {"source_sha256": source_sha, "metadata_mismatches": mismatches,
                "unmanifested_files": extra}

    def system_inventory(self):
        smi = run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,"
                   "pci.bus_id,clocks.max.sm,power.limit", "--format=csv,noheader"])
        gpus = [line.split(", ") for line in smi.get("stdout", "").splitlines() if line]
        self.check("gpu_inventory", len(gpus) == self.config["gpu"]
                   and all(self.config["gpu_type"] in gpu[1] for gpu in gpus))
        for name, argv in (("nvidia_smi_q.txt", ["nvidia-smi", "-q"]),
                           ("nvidia_smi_topo.txt", ["nvidia-smi", "topo", "-m"]),
                           ("nvidia_smi_nvlink.txt", ["nvidia-smi", "nvlink", "-s"]),
                           ("ibv_devinfo.txt", ["ibv_devinfo", "-v"])):
            output = run(argv)
            (self.node_dir / name).write_text(
                output.get("stdout", "") + output.get("stderr", "") + output.get("error", ""))
        infiniband = {}
        for device in sorted(Path("/sys/class/infiniband").glob("*")):
            ports = {}
            for port in sorted((device / "ports").glob("*")):
                ports[port.name] = {key: read_text(port / key)
                                    for key in ("state", "phys_state", "rate", "link_layer")}
            infiniband[device.name] = {
                "ports": ports, "fw_ver": read_text(device / "fw_ver"),
                "hca_type": read_text(device / "hca_type"),
                "numa_node": read_text(device / "device" / "numa_node"),
                "net": sorted(p.name for p in (device / "device" / "net").glob("*"))}
        active = [name for name, device in infiniband.items()
                  if any("ACTIVE" in (port["state"] or "") for port in device["ports"].values())]
        self.check("rdma_devices_active", active)
        environment = {}
        for key, value in sorted(os.environ.items()):
            if RECORDED_ENV.match(key):
                secret = any(word in key.upper() for word in SECRET_WORDS)
                environment[key] = "<redacted>" if secret else value
        return {
            "gpus": gpus, "gpu_query": smi, "driver": read_text("/proc/driver/nvidia/version"),
            "infiniband": infiniband, "active_rdma_devices": active,
            "dev_infiniband": sorted(p.name for p in Path("/dev/infiniband").glob("*")),
            "nvidia_peermem_loaded": Path("/sys/module/nvidia_peermem").exists(),
            "gdrdrv_present": Path("/dev/gdrdrv").exists(),
            "dpkg": run(["dpkg-query", "-W", "-f", "${Package}=${Version}\\n", "libgdrapi",
                         "libibverbs1", "ibverbs-providers", "rdma-core", "ucx", "libucx0"]),
            "ucx_info": run(["ucx_info", "-v"]),
            "addresses": run(["hostname", "-I"]),
            "cpu_count": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0)),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in
                               Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith("model name")), None),
            "mem_total": next(line for line in Path("/proc/meminfo").read_text().splitlines()
                              if line.startswith("MemTotal")),
            "shm": run(["df", "-h", "/dev/shm"]), "kernel": platform.release(),
            "environment": environment, "environment_names": sorted(os.environ)}

    def imports(self):
        out = self.work / "imports.json"
        self.finish([self.child(["--role", "imports", "--out", str(out)], "imports.log")],
                    "imports")
        result = json.loads(out.read_text())
        compat = self.config["cuda_compat_path"]
        vllm = [d for d in result["distributions"] if d.lower().startswith("vllm==")]
        self.check("vllm_version", vllm == [f"vllm=={self.config['expected_vllm_version']}"])
        self.check("cuda_execution", result["cuda_sum_of_squares"] == 1240.0
                   and result["cublas_bf16_matmul"])
        self.check("libcuda_from_compat", result["libcuda_loaded"]
                   and all(path.startswith(compat) for path in result["libcuda_loaded"]))
        self.check("nccl_runtime_loaded", result["nccl_runtime_returncode"] == 0
                   and result["nccl_runtime_raw_version"] > 0)
        write_json(self.node_dir / "imports.json", result)
        return {key: result[key] for key in result if key != "distributions"}

    def offline_resolution(self):
        guard_dir = self.work / "network_guard"
        guard_dir.mkdir()
        (guard_dir / "sitecustomize.py").write_text(NETWORK_GUARD)
        env = {"PYTHONPATH": os.pathsep.join(
                   filter(None, [str(guard_dir), os.environ.get("PYTHONPATH")])),
               "HF_HOME": str(self.work / "hf_home"),
               "T01_NETWORK_GUARD_LOG": str(self.node_dir / "network_guard.jsonl")}
        manifest = json.loads((self.metadata_root / "metadata_manifest.json").read_text())
        results = {}
        for label, repo in self.config["models"].items():
            revision = manifest["models"][repo]["revision"]
            model_dir = self.metadata_root / repo.replace("/", "__") / revision / "model"
            out = self.work / f"resolve_{label}.json"
            try:
                self.finish([self.child(["--role", "resolve", "--label", label, "--model-dir",
                                         str(model_dir), "--out", str(out)],
                                        f"resolve_{label}.log", env)], f"resolve {label}")
                results[label] = json.loads(out.read_text())
            except Exception:  # noqa: BLE001 - record each model separately
                results[label] = {"exception": traceback.format_exc()}
            self.check(f"offline_resolution_{label}", results[label].get("resolved"))
        return results

    def nccl(self, label, world, rank_offset, init_method):
        processes = []
        for local_rank in range(self.config["gpu"]):
            rank = rank_offset + local_rank
            processes.append(self.child(
                ["--role", "nccl", "--rank", str(rank), "--local-rank", str(local_rank),
                 "--world", str(world), "--init-method", init_method,
                 "--mib", str(self.config["allreduce_mib"]),
                 "--iterations", str(self.config["allreduce_iterations"]),
                 "--out", str(self.node_dir / f"{label}_rank{rank}.json")],
                f"{label}_rank{rank}.log",
                {**NCCL_DEBUG_ENV,
                 "NCCL_DEBUG_FILE": str(self.node_dir / f"{label}_nccl.%h.%p.log")}))
        self.finish(processes, label)
        ranks = [json.loads((self.node_dir / f"{label}_rank{rank_offset + i}.json").read_text())
                 for i in range(self.config["gpu"])]
        lines = [line for log in self.node_dir.glob(f"{label}_nccl.*.log")
                 for line in log.read_text(errors="replace").splitlines()]
        channels = [line for line in lines if " via " in line]
        transport = {
            "nccl_version_lines": sorted({line.split("NCCL INFO ", 1)[-1] for line in lines
                                          if "NCCL version" in line}),
            "net_ib_using": sorted({line.split("NCCL INFO ", 1)[-1] for line in lines
                                    if "NET/IB : Using" in line})[:4],
            "channels_total": len(channels),
            "channels_net_ib": sum("NET/IB" in line or "NET/Stepfun-IB/" in line
                                   for line in channels),
            "channels_gdrdma": sum("GDRDMA" in line for line in channels),
            "channels_net_socket": sum("NET/Socket" in line for line in channels),
            "channels_p2p": sum("P2P" in line for line in channels),
            "nvls": any("NVLS" in line for line in lines),
        }
        return {"ranks": ranks, "transport": transport,
                "platform_nccl_env": {k: v for k, v in os.environ.items()
                                      if k.startswith("NCCL_")}}

    def nccl_intra_node(self):
        result = self.nccl("nccl_intra", self.config["gpu"], 0,
                           f"tcp://127.0.0.1:{self.config['intra_node_port']}")
        self.check("nccl_intra_node_allreduce", all(r["correct"] for r in result["ranks"]))
        return result

    def rendezvous(self):
        write_json(self.node_dir / "rendezvous.json",
                   {"host": self.host, "ip": primary_ipv4(), "time": time.time()})
        expected = self.config["replica"]
        found = wait_for(lambda: len(list(self.output.glob("nodes/*/rendezvous.json"))) == expected,
                         self.config["peer_wait_seconds"], interval=5.0)
        peers = sorted((json.loads(p.read_text()) for p in
                        self.output.glob("nodes/*/rendezvous.json")), key=lambda p: p["host"])
        self.check("rendezvous", found)
        if not found:
            raise TimeoutError(f"only {len(peers)} of {expected} nodes arrived")
        self.node_rank = [p["host"] for p in peers].index(self.host)
        self.master_ip = peers[0]["ip"]
        return {"peers": peers, "node_rank": self.node_rank, "master_ip": self.master_ip}

    def nccl_cross_node(self):
        if self.node_rank is None:
            raise RuntimeError("rendezvous did not complete")
        world = self.config["gpu"] * self.config["replica"]
        result = self.nccl("nccl_cross", world, self.node_rank * self.config["gpu"],
                           f"tcp://{self.master_ip}:{self.config['cross_node_port']}")
        transport = result["transport"]
        self.check("nccl_cross_node_allreduce", all(r["correct"] for r in result["ranks"]))
        self.check("nccl_cross_node_uses_ib",
                   transport["channels_net_ib"] > 0 and transport["channels_net_socket"] == 0)
        return result

    def nixl_cross_node(self):
        if self.node_rank is None:
            raise RuntimeError("rendezvous did not complete")
        out = self.node_dir / "nixl_read.json"
        self.finish([self.child(["--role", "nixl", "--out", str(out)], "nixl_read.log",
                                {"UCX_PROTO_INFO": "y", "UCX_LOG_LEVEL": "debug"})],
                    "nixl READ")
        result = json.loads(out.read_text())
        self.check("nixl_cross_node_read", result["passed"])
        return result

    def download_audit(self):
        roots = sorted({str(p) for p in (self.work, self.metadata_root, self.node_dir)})
        owned_candidates = weight_candidates(roots, self.weight_suffixes)
        cache_after = weight_candidates(self.cache_roots, self.weight_suffixes)
        cache_changes = {path: evidence for path, evidence in cache_after.items()
                         if self.cache_candidates_before.get(path) != evidence}
        weight_files = sorted(set(owned_candidates) | set(cache_changes))
        guard_log = self.node_dir / "network_guard.jsonl"
        attempts = guard_log.read_text().splitlines() if guard_log.exists() else []
        hf_home = self.work / "hf_home"
        self.check("no_weight_files", not weight_files)
        self.check("no_network_attempts_offline_resolution", not attempts)
        return {"roots": roots, "suffixes": self.weight_suffixes, "weight_files": weight_files,
                "owned_candidates": owned_candidates, "cache_roots": self.cache_roots,
                "cache_candidates_before": self.cache_candidates_before,
                "cache_candidates_after": cache_after,
                "new_or_modified_cache_candidates": cache_changes,
                "scope": "Suffix candidates in task-owned roots and before/after cache metadata; "
                         "not proof of all network downloads. Unsuffixed blobs, transient files, "
                         "and paths outside these roots are not covered. Shared-cache changes "
                         "are not attributed to this task. The network guard covers only Python "
                         "socket calls in offline-resolution children. Model weights remain forbidden.",
                "network_attempts": attempts,
                "hf_home_files": sorted(str(p.relative_to(hf_home)) for p in hf_home.rglob("*")
                                        if p.is_file()) if hf_home.exists() else []}

    def run(self):
        for name in ("source_and_metadata", "system_inventory", "imports", "offline_resolution",
                     "nccl_intra_node", "rendezvous", "nccl_cross_node", "nixl_cross_node",
                     "download_audit"):
            self.phase(name, getattr(self, name))
        self.receipt["passed"] = (all(self.receipt["checks"].values()) and not any(
            "exception" in phase for phase in self.receipt["phases"].values()))
        self.receipt["finished"] = time.time()
        write_json(self.node_dir / "receipt.json", self.receipt)
        # Keep this replica alive until every node has written its receipt.
        wait_for(lambda: len(list(self.output.glob("nodes/*/receipt.json")))
                 == self.config["replica"], self.config["final_wait_seconds"], interval=5.0)
        print(json.dumps({"host": self.host, "passed": self.receipt["passed"],
                          "receipt": str(self.node_dir / "receipt.json")}), flush=True)
        return 0 if self.receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", choices=("imports", "resolve", "nccl", "nixl"))
    parser.add_argument("--out")
    parser.add_argument("--label")
    parser.add_argument("--model-dir")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--init-method")
    parser.add_argument("--mib", type=int)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.role:
        if args.role == "nixl":
            return role_nixl(args)
        {"imports": role_imports, "resolve": role_resolve, "nccl": role_nccl}[args.role](args)
        return 0
    config = json.loads(args.config.read_text())
    return Node(args.config, config).run()


if __name__ == "__main__":
    sys.exit(main())
