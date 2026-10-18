"""
Model registry for PolypSeg Framework.

Usage
-----
Any model file placed in this directory is auto-imported.
Each model class must be decorated with ``@register_model('name')``.

Example
-------
>>> from models import get_model, list_models
>>> model = get_model('pranet')
>>> print(list_models())
"""

import os
import glob
import importlib

_MODEL_REGISTRY: dict = {}


def register_model(name: str):
    """Class decorator that registers a model under ``name``.

    Args:
        name: Case-insensitive lookup key.

    Returns:
        The original class unchanged.
    """
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _auto_import() -> None:
    """Scan this directory and import every non-private .py module."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(model_dir, "*.py"))):
        module_name = os.path.basename(f)[:-3]
        if module_name != "__init__" and not module_name.startswith("_"):
            try:
                importlib.import_module(f"models.{module_name}")
            except ImportError as e:
                print(f"[models] Skipping '{module_name}': missing dependency ({e})")


def get_model(name: str):
    """Instantiate and return the model registered under ``name``.

    Args:
        name: Model name (case-insensitive), e.g. ``'pranet'``.

    Returns:
        An initialised ``nn.Module`` instance.

    Raises:
        AssertionError: If the name is not found in the registry.
    """
    if not _MODEL_REGISTRY:
        _auto_import()
    key = name.lower()
    assert key in _MODEL_REGISTRY, (
        f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}"
    )
    return _MODEL_REGISTRY[key]()


def list_models() -> list[str]:
    """Return all registered model names."""
    if not _MODEL_REGISTRY:
        _auto_import()
    return list(_MODEL_REGISTRY.keys())
