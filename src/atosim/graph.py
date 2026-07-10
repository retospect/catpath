"""Build the reaction graph (states = nodes, reactions = edges) and serialize."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .uncertainty import Estimate


def build_graph(
    node_energies: dict[str, Estimate],
    edges: list[dict],
    energy_ref: float | None = None,
) -> nx.DiGraph:
    """Assemble a DiGraph.

    ``node_energies``: state name -> energy Estimate (absolute, eV).
    ``edges``: list of ``{reactant, product, barrier: Estimate, delta_e: Estimate}``.
    ``energy_ref``: if given, node ``rel_energy`` is (energy - ref).
    """
    g = nx.DiGraph()
    ref = energy_ref if energy_ref is not None else min(
        (e.mean for e in node_energies.values()), default=0.0
    )
    for name, est in node_energies.items():
        g.add_node(
            name,
            energy=est.mean, energy_std=est.std,
            rel_energy=est.mean - ref,
            low_confidence=est.low_confidence,
        )
    zero = Estimate(0.0, 0.0, 0, [])
    for e in edges:
        kind = e.get("kind", "reaction")
        b: Estimate = e.get("barrier") or zero
        d: Estimate = e.get("delta_e") or zero
        g.add_edge(
            e["reactant"], e["product"],
            name=e.get("name", f"{e['reactant']}->{e['product']}"),
            kind=kind,
            barrier=b.mean, barrier_std=b.std,
            delta_e=d.mean, delta_e_std=d.std,
            low_confidence=(kind == "reaction") and (b.low_confidence or d.low_confidence),
        )
    return g


def to_json(g: nx.DiGraph, path: str | Path) -> None:
    data = nx.node_link_data(g, edges="links")
    Path(path).write_text(json.dumps(data, indent=2))


def to_csv(g: nx.DiGraph, nodes_path: str | Path, edges_path: str | Path) -> None:
    import csv

    with open(nodes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "energy_eV", "energy_std", "rel_energy_eV", "low_confidence"])
        for n, d in g.nodes(data=True):
            w.writerow([n, f"{d['energy']:.4f}", f"{d['energy_std']:.4f}",
                        f"{d['rel_energy']:.4f}", d["low_confidence"]])
    with open(edges_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reactant", "product", "kind", "barrier_eV", "barrier_std",
                    "delta_e_eV", "delta_e_std", "low_confidence"])
        for u, v, d in g.edges(data=True):
            w.writerow([u, v, d.get("kind", "reaction"),
                        f"{d['barrier']:.4f}", f"{d['barrier_std']:.4f}",
                        f"{d['delta_e']:.4f}", f"{d['delta_e_std']:.4f}",
                        d["low_confidence"]])
