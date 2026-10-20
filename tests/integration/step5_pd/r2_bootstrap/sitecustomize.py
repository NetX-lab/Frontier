"""Opt-in diagnostic bootstrap; never prevent native server startup."""
import os
import sys


if os.environ.get('STEP5_PD_STATE_OBSERVATION'):
    try:
        from r2_observation import install

        install()
    except Exception as error:
        print(f'STEP5_PD_STATE_OBSERVATION UNVERIFIED bootstrap: {error!r}',
              file=sys.stderr, flush=True)
