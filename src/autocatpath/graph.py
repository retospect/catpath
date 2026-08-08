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
        d: Estimate = e.get("delta_e") or zero
        attrs: dict = {
            "name": e.get("name", f"{e['reactant']}->{e['product']}"),
            "kind": kind,
            "delta_e": d.mean, "delta_e_std": d.std,
        }
        if "barrier" in e:
            # normal case -- also covers today's pre-existing "supply" edge
            # convention (kind="supply", no "barrier" key -> the `or zero`
            # fallback below never triggers here since this branch requires
            # the key to be present; kept for a hypothetical explicit-None).
            b: Estimate = e.get("barrier") or zero
            attrs["barrier"] = b.mean
            attrs["barrier_std"] = b.std
            attrs["low_confidence"] = (kind == "reaction") and (b.low_confidence or d.low_confidence)
        elif kind == "supply":
            # barrierless stoichiometry bridge (+O*/+H*) -- unchanged
            # pre-existing convention: no "barrier" key on the input edge,
            # zero on the graph edge.
            attrs["barrier"] = zero.mean
            attrs["barrier_std"] = zero.std
            attrs["low_confidence"] = False
        else:
            # SCREENING mode (SearchConfig.screening): no barrier was ever
            # computed for this reaction step (relax-only run, NEB skipped
            # entirely) -- leave the graph edge with NO `barrier` attribute
            # at all, never a fabricated 0.0. Downstream readers that treat
            # absence honestly: analysis.rate_limiting's `.get("barrier")`
            # -> None (no rate_Ea scalar); electrochem's `edge_barrier`
            # dict's `.get("barrier", 0.0)` -> thermodynamic-only span.
            attrs["low_confidence"] = d.low_confidence
        g.add_edge(e["reactant"], e["product"], **attrs)
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
            # SCREENING mode leaves a reaction edge with no barrier attribute
            # at all -- write blank cells rather than crashing on a missing key.
            barrier = d.get("barrier")
            barrier_std = d.get("barrier_std")
            w.writerow([u, v, d.get("kind", "reaction"),
                        "" if barrier is None else f"{barrier:.4f}",
                        "" if barrier_std is None else f"{barrier_std:.4f}",
                        f"{d['delta_e']:.4f}", f"{d['delta_e_std']:.4f}",
                        d["low_confidence"]])
