"""Helpers for optional Weights & Biases integration."""

from __future__ import annotations

from types import ModuleType
from typing import Optional


WANDB_IMPORT_ERROR = (
    "wandb is required when metrics_config.wandb_project and "
    "metrics_config.wandb_group are set. Install Frontier with the optional "
    "wandb extra, for example: python -m pip install -e \".[wandb]\"."
)


def get_wandb() -> Optional[ModuleType]:
    """Return the wandb module when installed, otherwise None."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def require_wandb() -> ModuleType:
    """Return wandb or fail fast when W&B logging was explicitly enabled."""
    wandb = get_wandb()
    if wandb is None:
        raise ImportError(WANDB_IMPORT_ERROR)
    return wandb
