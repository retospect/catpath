"""Render adsorbate/active-site structure thumbnails.

Design (per the discussion): every thumbnail uses the SAME fixed camera and the
SAME zoom window so states are directly comparable. Two canonical views per
state -- **top** (down the surface normal, reads the adsorption site) and
**side** (along the cell a-axis, reads height/tilt) -- both fixed across nodes.
The slab is cropped to the active site (adsorbate + nearby surface atoms) so the
tiny reagent isn't lost in a sea of metal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

# Fixed cameras (ASE rotation strings) shared by every thumbnail.
TOP_VIEW = ""              # look straight down +z (surface normal)
SIDE_VIEW = "-90x"         # look along the cell a-axis (surface edge-on)


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


def thumb_array(atoms, n_slab: int, window: float, radius: float = 5.5,
                px: int = 220) -> np.ndarray:
    """Return an RGBA array: top view stacked over side view (roughly square)."""
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    gs = fig.add_gridspec(2, 1, hspace=0.02)
    _draw_pair(fig, gs[0, 0], gs[1, 0], atoms, n_slab, window, radius)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return arr


def gallery(structures: dict, node_energies: dict, path: str | Path,
            n_slab: int, cols: int = 4, radius: float = 5.5,
            labels: bool = True) -> None:
    """Contact sheet: dual view per state, optionally labelled with energy +- sd."""
    names = list(structures)
    window = view_window(structures, n_slab, radius)
    rows = (len(names) + cols - 1) // cols
    fig = plt.figure(figsize=(2.2 * cols, 2.6 * rows))
    outer = fig.add_gridspec(rows, cols, wspace=0.15, hspace=0.28)
    for k, name in enumerate(names):
        r, c = divmod(k, cols)
        inner = outer[r, c].subgridspec(2, 1, hspace=0.02)
        est = node_energies.get(name)
        lab = name
        if labels and est is not None:
            lab = f"{name}  ({est.mean:+.2f}±{est.std:.2f} eV)"
        _draw_pair(fig, inner[0, 0], inner[1, 0], structures[name], n_slab,
                   window, radius, label=lab if labels else None)
    fig.suptitle("Active-site structures (top / side views, shared scale)",
                 fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
