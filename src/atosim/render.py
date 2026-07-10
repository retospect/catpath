"""Render adsorbate/active-site structure thumbnails.

Design (per the discussion): every thumbnail uses the SAME fixed camera and the
SAME zoom window so states are directly comparable. Two canonical views per
state -- **top** (down the surface normal, reads the adsorption site) and
**side** (along the cell a-axis, reads height/tilt) -- both fixed across nodes.
The slab is cropped to the active site (adsorbate + nearby surface atoms) so the
tiny reagent isn't lost in a sea of metal.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

# Fixed cameras (ASE rotation strings) shared by every thumbnail.
TOP_VIEW = ""              # look straight down +z (surface normal)
SIDE_VIEW = "-90x"         # look along the cell a-axis (surface edge-on)


def povray_available() -> bool:
    """True when the ``povray`` binary is on PATH (the ray-traced backend)."""
    return shutil.which("povray") is not None


def resolve_backend(requested: str) -> tuple[str, str | None]:
    """Return (backend, warning). Downgrade povray->matplotlib if unavailable."""
    if requested == "povray" and not povray_available():
        return "matplotlib", ("render.backend=povray requested but the 'povray' "
                              "binary is not on PATH; falling back to matplotlib "
                              "(install: apt-get install povray)")
    return requested, None


def active_site(atoms, n_slab: int, radius: float = 5.5):
    """Center the adsorbate, wrap, and crop to nearby surface atoms."""
    a = atoms.copy()
    a.set_constraint()  # constraints are irrelevant for rendering
    ads = list(range(n_slab, len(a)))
    if not ads:
        return a, radius
    pos = a.get_positions()
    cxy = pos[ads][:, :2].mean(0)
    cc = (0.5 * (a.cell[0] + a.cell[1]))[:2]
    pos[:, :2] += cc - cxy
    a.set_positions(pos)
    a.wrap()
    pos = a.get_positions()
    keep = [i for i in range(len(a))
            if i >= n_slab or np.linalg.norm(pos[i, :2] - cc) <= radius]
    return a[keep], float(cc[0])


def view_window(structures: dict, n_slab: int, radius: float = 5.5) -> float:
    """One fixed half-window (A) used for every state, from the largest site."""
    w = radius
    for atoms in structures.values():
        sub, _ = active_site(atoms, n_slab, radius)
        ads = sub.get_positions()[[i for i, s in enumerate(sub)
                                   if i >= len(sub) - (len(atoms) - n_slab)]]
        if len(ads):
            w = max(w, float(np.linalg.norm(ads[:, :2] - ads[:, :2].mean(0),
                                            axis=1).max()) + 1.5)
    return w


def _draw_pair(fig, gs_top, gs_side, atoms, n_slab, window, radius, label=None):
    sub, _ = active_site(atoms, n_slab, radius)
    ax_t, ax_s = fig.add_subplot(gs_top), fig.add_subplot(gs_side)
    plot_atoms(sub, ax_t, rotation=TOP_VIEW, radii=0.5)
    plot_atoms(sub, ax_s, rotation=SIDE_VIEW, radii=0.5)
    for ax in (ax_t, ax_s):
        ax.set_axis_off()
        ax.set_aspect("equal")
    cx, cy = sub.get_positions()[:, 0].mean(), sub.get_positions()[:, 1].mean()
    ax_t.set_xlim(cx - window, cx + window)   # fixed extents -> shared scale
    ax_t.set_ylim(cy - window, cy + window)
    if label:
        ax_t.set_title(label, fontsize=8, fontweight="bold")


# --- POV-Ray backend (ray-traced; optional, needs the `povray` binary) -------

def _projected_center(sub, rotation: str, n_ads: int) -> np.ndarray:
    """Adsorbate centroid in the rotated image plane -> fixes the crop per view."""
    from ase.io.utils import PlottingVariables
    pv = PlottingVariables(sub, scale=1.0, rotation=rotation)
    proj = pv.positions[:, :2]
    ads = proj[len(sub) - n_ads:] if n_ads else proj
    return ads.mean(0)


def _write_pov_scene(sub, rotation: str, window: float, n_ads: int,
                     out_pov: Path, width: int, bonds: bool):
    """Write a .pov/.ini scene (no render). Fixed bbox -> shared zoom per view."""
    from ase.io.pov import get_bondpairs, write_pov
    cx, cy = _projected_center(sub, rotation, n_ads)
    bbox = (cx - window, cy - window, cx + window, cy + window)
    settings: dict = {"canvas_width": int(width), "transparent": True,
                      "camera_type": "orthographic"}
    if bonds:
        settings["bondatoms"] = get_bondpairs(sub, radius=1.1)
    return write_pov(str(out_pov), sub, rotation=rotation, radii=0.6,
                     show_unit_cell=0, bbox=bbox, povray_settings=settings)


def _pov_view_array(sub, rotation: str, window: float, n_ads: int,
                    width: int, bonds: bool) -> np.ndarray:
    """Ray-trace one view to an RGBA uint8 array."""
    with tempfile.TemporaryDirectory() as td:
        pov = Path(td) / "scene.pov"
        inputs = _write_pov_scene(sub, rotation, window, n_ads, pov, width, bonds)
        png = inputs.render(clean_up=False)
        arr = mpimg.imread(str(png))
    if arr.ndim == 3 and arr.shape[-1] == 3:  # opaque -> add full alpha
        arr = np.dstack([arr, np.ones(arr.shape[:2])])
    return (arr * 255).astype(np.uint8)


def _vstack_rgba(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Stack two RGBA images vertically, padding to equal width (transparent)."""
    w = max(top.shape[1], bottom.shape[1])

    def pad(a):
        if a.shape[1] == w:
            return a
        p = np.zeros((a.shape[0], w - a.shape[1], 4), a.dtype)
        return np.hstack([a, p])

    return np.vstack([pad(top), pad(bottom)])


def _pov_pair_array(atoms, n_slab: int, window: float, radius: float,
                    width: int, bonds: bool) -> np.ndarray:
    """Ray-traced top-over-side pair (mirrors the matplotlib thumbnail layout)."""
    sub, _ = active_site(atoms, n_slab, radius)
    n_ads = max(0, len(atoms) - n_slab)
    top = _pov_view_array(sub, TOP_VIEW, window, n_ads, width, bonds)
    side = _pov_view_array(sub, SIDE_VIEW, window, n_ads, width, bonds)
    return _vstack_rgba(top, side)


def thumb_array(atoms, n_slab: int, window: float, radius: float = 5.5,
                px: int = 220, backend: str = "matplotlib",
                width: int = 320, bonds: bool = True) -> np.ndarray:
    """Return an RGBA array: top view stacked over side view (roughly square).

    ``backend="povray"`` ray-traces the pair (requires the ``povray`` binary);
    the default matplotlib backend draws flat CPK circles with no dependencies.
    """
    if backend == "povray":
        return _pov_pair_array(atoms, n_slab, window, radius, width, bonds)
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    gs = fig.add_gridspec(2, 1, hspace=0.02)
    _draw_pair(fig, gs[0, 0], gs[1, 0], atoms, n_slab, window, radius)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return arr


def _cell_label(name: str, est, labels: bool) -> str | None:
    if not labels:
        return None
    if est is not None:
        return f"{name}  ({est.mean:+.2f}±{est.std:.2f} eV)"
    return name


def gallery(structures: dict, node_energies: dict, path: str | Path,
            n_slab: int, cols: int = 4, radius: float = 5.5,
            labels: bool = True, backend: str = "matplotlib",
            width: int = 320, bonds: bool = True) -> None:
    """Contact sheet: dual view per state, optionally labelled with energy +- sd.

    ``backend="povray"`` ray-traces each cell (needs the ``povray`` binary);
    otherwise flat matplotlib CPK circles are drawn (the dependency-free default).
    """
    names = list(structures)
    window = view_window(structures, n_slab, radius)
    rows = (len(names) + cols - 1) // cols

    if backend == "povray":
        fig = plt.figure(figsize=(2.2 * cols, 2.6 * rows))
        for k, name in enumerate(names):
            ax = fig.add_subplot(rows, cols, k + 1)
            arr = _pov_pair_array(structures[name], n_slab, window, radius,
                                  width, bonds)
            ax.imshow(arr)
            ax.set_axis_off()
            lab = _cell_label(name, node_energies.get(name), labels)
            if lab:
                ax.set_title(lab, fontsize=8, fontweight="bold")
        fig.suptitle("Active-site structures (POV-Ray; top / side, shared scale)",
                     fontsize=11)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(2.2 * cols, 2.6 * rows))
    outer = fig.add_gridspec(rows, cols, wspace=0.15, hspace=0.28)
    for k, name in enumerate(names):
        r, c = divmod(k, cols)
        inner = outer[r, c].subgridspec(2, 1, hspace=0.02)
        lab = _cell_label(name, node_energies.get(name), labels)
        _draw_pair(fig, inner[0, 0], inner[1, 0], structures[name], n_slab,
                   window, radius, label=lab)
    fig.suptitle("Active-site structures (top / side views, shared scale)",
                 fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
