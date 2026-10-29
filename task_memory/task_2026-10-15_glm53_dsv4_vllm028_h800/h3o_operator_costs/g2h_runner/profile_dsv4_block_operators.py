"""Reuse the accepted C26 prefill bootstrap, substituting only the profiler CLI."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    sys.path[:0] = [config["source_snapshot"], config["worker_wheel_site"]]
    from frontier.profiling.attention import dsv4_attention_main
    from frontier.profiling.model_runner.dsv4_block_operators import main as direct
    original = dsv4_attention_main.main
    dsv4_attention_main.main = direct
    try:
        runpy.run_path(str(Path(config["source_snapshot"]) / "profile_dsv4_prefill.py"), run_name="__main__")
    finally:
        dsv4_attention_main.main = original


if __name__ == "__main__":
    main()
