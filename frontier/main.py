import os
import sys

from frontier.config import (
    AICONFIGURATOR_BACKEND_RELEASE_ERROR,
    DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
    PD_AF_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR,
    PD_AF_PREFIX_CACHING_RELEASE_ERROR,
    PD_AF_TRACE_REPLAY_DEFERRED_ERROR,
    SimulationConfig,
)
from frontier.errors import FrontierMemoryOOMError
from frontier.logger import set_log_level
from frontier.utils.random import set_seeds


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


def _exit_if_aiconfigurator_backend_requested(argv: list[str]) -> None:
    if _has_aiconfigurator_backend_option(argv):
        print(AICONFIGURATOR_BACKEND_RELEASE_ERROR, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    try:
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
        error_message = str(exc)
        if error_message in {
            AICONFIGURATOR_BACKEND_RELEASE_ERROR,
            DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
            PD_AF_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR,
            PD_AF_PREFIX_CACHING_RELEASE_ERROR,
        } or error_message.startswith(PD_AF_TRACE_REPLAY_DEFERRED_ERROR):
            print(error_message, file=sys.stderr)
            raise SystemExit(1) from exc
        raise


if __name__ == "__main__":
    main()
