"""precis-mcp bridge for autocatpath — the `pathway` kind.

Installed via ``pip install autocatpath[precis]`` and discovered by a running
precis-mcp server through the ``precis.handlers`` / ``precis.migrations``
entry points (see ``autocatpath``'s ``pyproject.toml``). Design-of-record:
``docs/design/autocatpath-integration.md`` in the precis-mcp repo.

``PathwayHandler`` is exposed lazily so that ``autocatpath.precis.runner`` (the
pure, precis-free half) stays importable in environments without
precis-mcp installed — importing the handler pulls in ``precis``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PathwayHandler"]


def __getattr__(name: str) -> Any:
    if name == "PathwayHandler":
        from .handler import PathwayHandler

        return PathwayHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
