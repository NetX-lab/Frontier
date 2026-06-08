import os
import sys

from frontier.config import (
    AICONFIGURATOR_BACKEND_RELEASE_ERROR,
    DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
    SimulationConfig,
)
from frontier.errors import FrontierMemoryOOMError
from frontier.logger import set_log_level
from frontier.utils.random import set_seeds


_DISAGGREGATED_ARCHITECTURES = {"pd-disaggregation", "pd-af-disaggregation"}
_DISAGGREGATED_CLUSTER_OPTION_PREFIXES = (
    "--cluster_config_prefill_",
    "--cluster_config_decode_",
    "--cluster_config_decode_attn_",
    "--cluster_config_decode_ffn_",
)
_DISAGGREGATED_CLUSTER_OPTIONS = frozenset(
    {
        "--cluster_config_af_pipeline_num_micro_batch",
    }
)
_DISAGGREGATED_TRANSFER_OPTION_MARKERS = (
    "kv_cache_transfer_config",
    "m2n_transfer_config",
)
_AICONFIGURATOR_BACKEND_CONFIG_OPTION_PREFIXES = (
    "--aiconfigurator_cc_backend_config_",
)
_AICONFIGURATOR_BACKEND_TYPE_OPTIONS = frozenset(
    {
        "--cc_backend_config_type",
        "--cluster_config_cc_backend_config_type",
        "--cluster_config_prefill_cc_backend_config_type",
        "--cluster_config_decode_cc_backend_config_type",
        "--cluster_config_decode_attn_cc_backend_config_type",
        "--cluster_config_decode_ffn_cc_backend_config_type",
    }
)


def _get_cli_option_value(argv: list[str], option: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == option:
            if index + 1 >= len(argv):
                return None
            return argv[index + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", maxsplit=1)[1]
    return None


def _normalize_cli_option(option: str) -> str:
    if option.startswith("--no-"):
        return f"--{option[len('--no-'):]}"
    return option


def _has_disaggregated_cluster_option(argv: list[str]) -> bool:
    for arg in argv:
        option = _normalize_cli_option(arg.split("=", maxsplit=1)[0])
        if option in _DISAGGREGATED_CLUSTER_OPTIONS:
            return True
        if option.startswith(_DISAGGREGATED_CLUSTER_OPTION_PREFIXES):
            return True
    return False


def _has_disaggregated_transfer_option(argv: list[str]) -> bool:
    return any(
        arg.startswith("--") and any(
            marker in arg for marker in _DISAGGREGATED_TRANSFER_OPTION_MARKERS
        )
        for arg in argv
    )


def _has_aiconfigurator_backend_option(argv: list[str]) -> bool:
    for index, arg in enumerate(argv):
        if not arg.startswith("--"):
            continue
        option, has_value, inline_value = arg.partition("=")
        if any(
            option.startswith(prefix)
            for prefix in _AICONFIGURATOR_BACKEND_CONFIG_OPTION_PREFIXES
        ):
            return True
        if option in _AICONFIGURATOR_BACKEND_TYPE_OPTIONS:
            if has_value:
                value = inline_value
            elif index + 1 < len(argv):
                value = argv[index + 1]
            else:
                value = ""
            if value.strip().lower() == "aiconfigurator":
                return True
    return False


def _has_truthy_cli_bool(argv: list[str], option: str) -> bool:
    for arg in argv:
        if arg == option:
            return True
        if arg.startswith(f"{option}="):
            value = arg.split("=", maxsplit=1)[1].strip().lower()
            return value in {"1", "true", "yes", "on"}
    return False


def _exit_if_disaggregated_architecture_requested(argv: list[str]) -> None:
    sys_arch = _get_cli_option_value(argv, "--sys_arch")
    has_disaggregated_cluster_args = _has_disaggregated_cluster_option(argv)
    has_disaggregated_transfer_args = _has_disaggregated_transfer_option(argv)
    has_pd_af_cuda_graph_arg = _has_truthy_cli_bool(argv, "--use_cuda_graph")
    if (
        sys_arch in _DISAGGREGATED_ARCHITECTURES
        or has_disaggregated_cluster_args
        or has_disaggregated_transfer_args
        or has_pd_af_cuda_graph_arg
    ):
        print(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR, file=sys.stderr)
        raise SystemExit(1)


def _exit_if_aiconfigurator_backend_requested(argv: list[str]) -> None:
    if _has_aiconfigurator_backend_option(argv):
        print(AICONFIGURATOR_BACKEND_RELEASE_ERROR, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    try:
        _exit_if_disaggregated_architecture_requested(sys.argv[1:])
        _exit_if_aiconfigurator_backend_requested(sys.argv[1:])
        log_level = os.environ.get("FRONTIER_LOG_LEVEL")
        if log_level:
            set_log_level(log_level)
        config: SimulationConfig = SimulationConfig.create_from_cli_args()
        if log_level:
            # SimulationConfig applies CLI/default logging in __post_init__.
            # Reapply the environment override so long-running probes can stay quiet.
            set_log_level(log_level)
        set_seeds(config.seed)

        from frontier.simulator import Simulator

        simulator = Simulator(config)
        simulator.run()
    except FrontierMemoryOOMError as exc:
        print(f"FRONTIER_MEMORY_OOM: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ValueError as exc:
        if str(exc) in {
            AICONFIGURATOR_BACKEND_RELEASE_ERROR,
            DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
        }:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        raise


if __name__ == "__main__":
    main()
