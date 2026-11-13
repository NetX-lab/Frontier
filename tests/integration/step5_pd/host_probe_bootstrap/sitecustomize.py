"""Opt-in host-step probe bootstrap; never prevent native server startup."""
import os
import sys


if os.environ.get('STEP5_HOST_STEP_PROBE'):
    try:
        from host_step_probe import install

        install()
    except Exception as error:
        print(f'STEP5_HOST_STEP_PROBE UNVERIFIED bootstrap: {error!r}', file=sys.stderr, flush=True)
