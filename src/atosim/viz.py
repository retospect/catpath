"""Rendering: the reaction graph and the substrate x intermediate energy map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / container-safe
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402


def draw_graph(g: nx.DiGraph, path: str | Path, title: str = "Reaction graph") -> None:
    """Layered energy-ordered layout with barrier-labelled edges."""
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
    ax.set_ylabel("relative energy (eV)")
    ax.set_title(title)
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
