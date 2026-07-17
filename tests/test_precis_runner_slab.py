"""The precis `structure` input seam at the ``run_pathway`` entry point.

``catpath.precis.runner`` imports only catpath (no precis-mcp), so these run
in the base env. We monkeypatch ``run`` so the seam is tested deterministically
without spending a real relax/NEB — the point under test is that an injected
slab is hydrated from extxyz and threaded onto the config for ``_build_net``.
"""

from __future__ import annotations

import io

import pytest
from ase.io import write as ase_write

from catpath.config import SlabConfig
from catpath.precis import runner
from catpath.structures import build_slab


def _extxyz(atoms) -> str:
    buf = io.StringIO()
    ase_write(buf, atoms, format="extxyz")
    return buf.getvalue()


class _Stop(Exception):
    pass


def _capture_run(monkeypatch) -> dict:
    captured: dict = {}

    def _fake_run(cfg, log=None):
        captured["prebuilt_slab"] = getattr(cfg, "_prebuilt_slab", None)
        raise _Stop  # short-circuit before any compute

    monkeypatch.setattr(runner, "run", _fake_run)
    return captured


_CONFIG = {"substrate": "NO", "target": "NO2", "network": "oxidation",
           "slab": {"element": "Pd", "size": [2, 2, 3]}}


def test_run_pathway_hydrates_and_threads_injected_slab(monkeypatch):
    captured = _capture_run(monkeypatch)
    slab = build_slab(SlabConfig(element="Pd", size=(2, 2, 3)))
    with pytest.raises(_Stop):
        runner.run_pathway(_CONFIG, slab_extxyz=_extxyz(slab))
    got = captured["prebuilt_slab"]
    assert got is not None
    assert len(got) == len(slab)
    assert got.get_chemical_formula() == slab.get_chemical_formula()


def test_run_pathway_without_slab_leaves_config_clean(monkeypatch):
    captured = _capture_run(monkeypatch)
    with pytest.raises(_Stop):
        runner.run_pathway(_CONFIG)
    assert captured["prebuilt_slab"] is None  # label-built slab, as before


def test_injected_slab_does_not_leak_into_content_key(monkeypatch):
    """The Atoms travel on a runtime attr, never into to_dict/content_key —
    so two runs that differ only by slab bytes still key on the config."""
    from catpath.precis.runner import content_key
    from catpath.config import Config

    cfg = Config.from_dict(_CONFIG)
    key_before = content_key(cfg.to_dict())
    cfg._prebuilt_slab = build_slab(SlabConfig(element="Pd", size=(2, 2, 3)))  # type: ignore[attr-defined]
    assert content_key(cfg.to_dict()) == key_before
