"""
Loss function registry for PolypSeg Framework.

Usage
-----
Place any loss file in this directory.
Decorate the function with ``@register_loss('name')``.

Example
-------
>>> from loss import get_loss, list_losses
>>> loss_fn = get_loss('structure_loss')
>>> l = loss_fn(pred, mask)
"""

import os
import glob
import importlib

_LOSS_REGISTRY: dict = {}


def register_loss(name: str):
    """Function decorator that registers a loss under ``name``.

    Args:
        name: Case-insensitive lookup key (typically matches the file name).

    Returns:
        The original callable unchanged.
    """
    def decorator(fn):
        _LOSS_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def _auto_import() -> None:
    """Scan this directory and import every non-private .py module."""
    loss_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(loss_dir, "*.py"))):
        module_name = os.path.basename(f)[:-3]
        if module_name != "__init__" and not module_name.startswith("_"):
            try:
                importlib.import_module(f"loss.{module_name}")
            except ImportError as e:
                print(f"[loss] Skipping '{module_name}': missing dependency ({e})")


def get_loss(name: str):
    """Return the loss callable registered under ``name``.

    Args:
        name: Loss name (case-insensitive), e.g. ``'structure_loss'``.

    Returns:
        Callable with signature ``fn(pred, mask) -> torch.Tensor``.

    Raises:
        AssertionError: If the name is not in the registry.
    """
    if not _LOSS_REGISTRY:
        _auto_import()
    key = name.lower()
    assert key in _LOSS_REGISTRY, (
        f"Loss '{name}' not found. Available: {list(_LOSS_REGISTRY.keys())}"
    )
    return _LOSS_REGISTRY[key]


def list_losses() -> list[str]:
    """Return all registered loss names."""
    if not _LOSS_REGISTRY:
        _auto_import()
    return list(_LOSS_REGISTRY.keys())
