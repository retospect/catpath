"""Structure rendering: matplotlib (default) and the optional POV-Ray backend.

The POV-Ray *binary* may be absent in CI, so we test what we can without it:
scene construction (writes a valid .pov), backend resolution/fallback, and the
matplotlib path.  The actual ray-trace runs only when `povray` is on PATH.
"""

import numpy as np
import pytest

from atosim import render
from atosim.config import SlabConfig
from atosim.network import build_ammonia_network


def _no_on_slab():
    net = build_ammonia_network(SlabConfig(size=(2, 2, 3), vacuum=8.0))
    slab = net.slab()
    atoms = net.states()["NO"].build(slab)
    return atoms, slab.info["n_slab"]


def test_resolve_backend_matplotlib_passthrough():
    backend, warn = render.resolve_backend("matplotlib")
    assert backend == "matplotlib" and warn is None


def test_resolve_backend_povray_fallback_matches_binary():
    backend, warn = render.resolve_backend("povray")
    if render.povray_available():
        assert backend == "povray" and warn is None
    else:
        assert backend == "matplotlib" and "povray" in warn


def test_matplotlib_thumb_array_shape():
    atoms, n_slab = _no_on_slab()
    window = render.view_window({"NO": atoms}, n_slab)
    arr = render.thumb_array(atoms, n_slab, window)  # default backend
    assert arr.ndim == 3 and arr.shape[-1] == 4      # RGBA
    assert arr.dtype == np.uint8


def test_pov_scene_is_written_even_without_binary(tmp_path):
    """Scene construction (the whole ASE-POV wiring) works with no `povray`."""
    atoms, n_slab = _no_on_slab()
    sub, _ = render.active_site(atoms, n_slab)
    n_ads = len(atoms) - n_slab
    window = render.view_window({"NO": atoms}, n_slab)
    out = tmp_path / "scene.pov"
    inputs = render._write_pov_scene(sub, render.TOP_VIEW, window, n_ads,
                                     out, width=200, bonds=True)
    assert out.exists()
    assert out.with_suffix(".ini").exists()          # POVRAYInputs points at .ini
    assert inputs.path.suffix == ".ini"
    text = out.read_text()
    assert "camera" in text and "orthographic" in text


@pytest.mark.skipif(not render.povray_available(),
                    reason="povray binary not installed")
def test_povray_thumb_array_renders():
    atoms, n_slab = _no_on_slab()
    window = render.view_window({"NO": atoms}, n_slab)
    arr = render.thumb_array(atoms, n_slab, window, backend="povray", width=160)
    assert arr.ndim == 3 and arr.shape[-1] == 4 and arr.size > 0
