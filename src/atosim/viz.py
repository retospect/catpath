"""Rendering: the reaction graph and the substrate x intermediate energy map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / container-safe
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402


def _place_thumbs(ax, positions: dict, thumbs: dict, zoom: float = 0.42) -> None:
    """Overlay structure thumbnails at the given {node: (x, y)} positions."""
    for name, xy in positions.items():
        arr = thumbs.get(name)
        if arr is not None:
            ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), xy, frameon=False,
                                pad=0, zorder=5)
            ax.add_artist(ab)


def draw_graph(g: nx.DiGraph, path: str | Path, title: str = "Reaction graph",
               thumbs: dict | None = None) -> None:
    """Layered energy-ordered layout with barrier-labelled edges.

    When ``thumbs`` (a {node: RGBA array} map) is given, structure thumbnails are
    overlaid on the nodes.
    """
    # x by topological order, y by relative energy -> reads like a profile.
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        order = list(g.nodes)
    xpos = {n: i for i, n in enumerate(order)}
    pos = {n: (xpos[n], g.nodes[n]["rel_energy"]) for n in g.nodes}

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(g), 5))
    node_colors = ["#d9534f" if g.nodes[n]["low_confidence"] else "#4a90d9"
                   for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_size=1600, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color="white")

    reaction = [(u, v) for u, v, d in g.edges(data=True) if d.get("kind") != "supply"]
    supply = [(u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "supply"]
    nx.draw_networkx_edges(g, pos, edgelist=reaction, ax=ax, arrowsize=18,
                           node_size=1600, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(g, pos, edgelist=supply, ax=ax, arrowsize=12,
                           node_size=1600, style="dashed", edge_color="#999999",
                           connectionstyle="arc3,rad=0.05")
    elabels = {
        (u, v): f"Ea={d['barrier']:.2f}\n±{d['barrier_std']:.2f} eV"
        for u, v, d in g.edges(data=True) if d.get("kind") != "supply"
    }
    nx.draw_networkx_edge_labels(g, pos, edge_labels=elabels, ax=ax, font_size=8)
    if thumbs:
        _place_thumbs(ax, pos, thumbs, zoom=0.5)
    ax.set_ylabel("relative energy (eV)")
    ax.set_title(title)
    ax.margins(0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hump(x0, y0, x1, y1, y_peak, n=40):
    """Smooth cosine curve from (x0,y0) up to a peak y_peak then down to (x1,y1)."""
    xm = (x0 + x1) / 2
    t1 = np.linspace(0, 1, n // 2)
    xa = x0 + (xm - x0) * t1
    ya = y0 + (y_peak - y0) * (1 - np.cos(np.pi * t1)) / 2
    t2 = np.linspace(0, 1, n // 2)
    xb = xm + (x1 - xm) * t2
    yb = y_peak + (y1 - y_peak) * (1 - np.cos(np.pi * t2)) / 2
    return np.concatenate([xa, xb]), np.concatenate([ya, yb])


def draw_profile(g, path: str | Path, title: str = "Reaction energy profile",
                 caption: str | None = None, show_caption: bool = True,
                 png_meta: dict | None = None, thumbs: dict | None = None) -> None:
    """Reaction-coordinate energy diagram.

    Each species is a bold horizontal level line labelled with its name; each
    reaction is a transition-state *bump* between two levels whose height above
    the reactant equals the barrier; barrierless "supply" steps are dashed
    connectors.  Every root->leaf pathway is drawn as its own coloured profile,
    so competing pathways are overlaid on one plot.

    Uncertainty: level energies and barriers carry a +/- 1 s.d. band across
    samples (seeds x models).  ``caption`` (when ``show_caption``) is printed as a
    visible provenance footer; ``png_meta`` is embedded as PNG tEXt chunks so the
    provenance travels with the pixels even on the caption-less image.
    """
    roots = [n for n, d in g.in_degree() if d == 0] or [next(iter(g.nodes))]
    leaves = [n for n, d in g.out_degree() if d == 0]
    paths = []
    for r in roots:
        for lf in leaves:
            paths.extend(nx.all_simple_paths(g, r, lf))
    if not paths:
        paths = [list(g.nodes)]

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(2 + 2.2 * max(len(p) for p in paths), 6))
    half = 0.34  # half-width of a level bar
    labeled: set[tuple[int, str]] = set()

    for pi, nodes in enumerate(paths):
        color = cmap(pi % 10)
        ys = [g.nodes[n]["rel_energy"] for n in nodes]
        es = [g.nodes[n].get("energy_std", 0.0) for n in nodes]
        for i, (n, y, sd) in enumerate(zip(nodes, ys, es)):
            ax.hlines(y, i - half, i + half, color=color, lw=4, zorder=3)
            if sd > 1e-6:  # +/- 1 s.d. band across seeds
                ax.fill_between([i - half, i + half], [y - sd] * 2, [y + sd] * 2,
                                color=color, alpha=0.18, lw=0, zorder=1)
            if (i, n) not in labeled:  # label each level once per x position
                lab = f"{n}" if sd <= 1e-6 else f"{n}\n({y:+.2f}±{sd:.2f})"
                ax.text(i, y + sd + 0.03, lab, ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=color, zorder=4)
                labeled.add((i, n))
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            d = g[u][v]
            y0, y1 = ys[i], ys[i + 1]
            if d.get("kind") == "supply":
                ax.plot([i + half, i + 1 - half], [y0, y1], ls=":",
                        color=color, alpha=0.6, lw=1.5, zorder=2)
            else:
                peak = y0 + max(d["barrier"], 0.0)
                xs, yc = _hump(i + half, y0, i + 1 - half, y1, peak)
                ax.plot(xs, yc, color=color, lw=2, zorder=2)
                bsd = d.get("barrier_std", 0.0)
                if bsd > 1e-6:  # barrier uncertainty as a cap at the TS
                    ax.errorbar((2 * i + 1) / 2, peak, yerr=bsd, color=color,
                                capsize=3, elinewidth=1, zorder=2)
                if d["barrier"] > 1e-3:
                    lab = (f"Ea={d['barrier']:.2f}" if bsd <= 1e-6
                           else f"Ea={d['barrier']:.2f}±{bsd:.2f}")
                    ax.annotate(lab, xy=((2 * i + 1) / 2, peak + bsd),
                                xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=7, color=color)
        ax.plot([], [], color=color, lw=3, label=" -> ".join(nodes))

    if thumbs:  # one thumbnail per state, above its (first) level line
        placed: dict = {}
        for pi, nodes in enumerate(paths):
            for i, n in enumerate(nodes):
                if n not in placed:
                    placed[n] = (i, g.nodes[n]["rel_energy"])
        _place_thumbs(ax, placed, thumbs, zoom=0.33)

    ax.set_xlabel("reaction coordinate")
    ax.set_ylabel("relative energy (eV)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.margins(x=0.05, y=0.15)

    visible = bool(caption) and show_caption
    if visible:
        fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7,
                 color="#555555")
        fig.subplots_adjust(bottom=0.12)

    fig.tight_layout(rect=(0, 0.03, 1, 1) if visible else None)
    # PNG tEXt metadata travels with the pixels even on the caption-less image
    fig.savefig(path, dpi=150, metadata=png_meta or None)
    plt.close(fig)


def energy_map(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    path: str | Path,
    title: str = "Substrate x intermediate energy map",
) -> None:
    """Heatmap: rows=substrates, cols=intermediate states, cell=relative energy.

    A star marks the highest-energy (rate-limiting) state in each row.
    """
    m = np.asarray(matrix, float)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * m.shape[1], 1.2 + 0.8 * m.shape[0]))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    ax.set_xticks(range(m.shape[1]), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]), row_labels)
    fig.colorbar(im, ax=ax, label="relative energy (eV)")

    for i in range(m.shape[0]):
        row = m[i]
        if np.all(np.isnan(row)):
            continue
        star = int(np.nanargmax(row))
        for j in range(m.shape[1]):
            if np.isnan(m[i, j]):
                continue
            mark = " *" if j == star else ""
            ax.text(j, i, f"{m[i, j]:.2f}{mark}", ha="center", va="center",
                    color="white", fontsize=8,
                    fontweight="bold" if j == star else "normal")
    ax.set_title(title + "\n(* = highest-energy / rate-limiting state)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
