"""autocatpath - reproducible ML-potential reaction pathway explorer."""

from importlib.metadata import PackageNotFoundError, version

from .config import Config

try:  # single source of truth: the installed package metadata (pyproject version)
    __version__ = version("autocatpath")
except PackageNotFoundError:  # source tree with no dist metadata
    __version__ = "0.0.0+unknown"

__all__ = ["Config"]
