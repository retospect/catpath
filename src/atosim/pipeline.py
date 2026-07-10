"""Orchestrate one exploration run end-to-end.

Flow (per the plan):
    build network -> for each seed, for each step:
        rattle & pre-relax & relax reactant + product (validate geometry)
        -> NEB barrier   (this is one "partial")
    -> aggregate partials across seeds (mean +/- spread)
    -> reaction graph + substrate x intermediate energy map -> write outputs.

The per-seed unit (:func:`run_one_seed`) is deliberately standalone and
JSON-serialisable so an orchestrator (Snakemake) can fan out seeds across jobs
and call :func:`aggregate_partials` to combine them.  Uses the pluggable
calculator, so the same code runs on EMT (dev) or MACE/fairchem (production).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calculators import check_supported, make_calculator
from .config import Config
from .graph import build_graph, to_csv, to_json
from .network import Network, StateSpec, build_network
from .neb import neb_barrier
from .relax import pre_relax, relax
from .structures import rattle_adsorbate, symbols_of
from .uncertainty import Estimate, aggregate
from .validate import geometry_ok
from .viz import draw_graph, draw_profile, energy_map


def g_has_edge(edges: list[dict], a: str, b: str) -> bool:
    return any(e["reactant"] == a and e["product"] == b for e in edges)


@dataclass
class Results:
    node_energies: dict[str, Estimate] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    pathway: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)  # supply edges
    warnings: list[str] = field(default_factory=list)


def _relax_state(state: StateSpec, slab, n_slab, cfg: Config, seed: int):
    """Rattle -> pre-relax -> relax one state; return (result, geometry_report)."""
    base = state.build(slab)
    start = rattle_adsorbate(base, n_slab, seed=seed, amplitude=0.15)
    cleaned = pre_relax(start, make_calculator(cfg.mlip))
    res = relax(cleaned, make_calculator(cfg.mlip),
                fmax=cfg.search.fmax, max_steps=cfg.search.max_steps)
    geo = geometry_ok(res.atoms, n_slab)
    return res, geo


def run_one_seed(cfg: Config, seed: int, log=print) -> dict:
    """Run the whole network for a single seed -> a JSON-serialisable partial."""
    net = build_network(cfg.slab, cfg.network)
    slab = net.slab()
    n_slab = slab.info["n_slab"]

    all_syms: set[str] = set()
    for st in net.steps:
        all_syms |= symbols_of(st.reactant.build(slab))
    check_supported(all_syms, cfg.mlip)

    states: dict[str, float] = {}
    steps: dict[str, dict] = {}
    warnings: list[str] = []

    for step in net.steps:
        log(f"[{step.name}] seed={seed}: relaxing endpoints")
        r_res, r_geo = _relax_state(step.reactant, slab, n_slab, cfg, seed)
        p_res, p_geo = _relax_state(step.product, slab, n_slab, cfg, seed)
        for st, res, geo in ((step.reactant, r_res, r_geo), (step.product, p_res, p_geo)):
            # keep the lowest energy seen for a state within this seed
            states[st.name] = min(states.get(st.name, np.inf), res.energy)
            if not geo.ok:
                warnings.append(f"{st.name} seed={seed} geometry: {'; '.join(geo.reasons)}")
            if not res.converged:
                warnings.append(f"{st.name} seed={seed} not converged")

        log(f"[{step.name}] seed={seed}: NEB")
        entry = {"reactant": step.reactant.name, "product": step.product.name,
                 "barrier": None, "delta_e": None}
        try:
            bar = neb_barrier(
                r_res.atoms, p_res.atoms,
                make_calc=lambda: make_calculator(cfg.mlip),
                n_images=cfg.search.neb_images,
                fmax=cfg.search.neb_fmax, max_steps=cfg.search.neb_max_steps,
            )
            entry["barrier"] = bar.barrier
            entry["delta_e"] = bar.delta_e
            if not bar.converged:
                warnings.append(f"{step.name} seed={seed} NEB not converged")
        except Exception as e:  # abandon this seed's edge, keep going
            warnings.append(f"{step.name} seed={seed} NEB failed: {e}")
        steps[step.name] = entry

    return {"seed": seed, "states": states, "steps": steps, "warnings": warnings}


def aggregate_partials(cfg: Config, partials: list[dict]) -> Results:
    """Combine per-seed partials into mean +/- spread Estimates."""
    net = build_network(cfg.slab, cfg.network)
    results = Results(pathway=net.order())
    results.links = list(net.links)

    state_vals: dict[str, list[float]] = {}
    step_bar: dict[str, list[float]] = {}
    step_de: dict[str, list[float]] = {}
    step_meta: dict[str, tuple[str, str]] = {}

    for p in partials:
        results.warnings.extend(p.get("warnings", []))
        for name, e in p["states"].items():
            state_vals.setdefault(name, []).append(e)
        for sname, s in p["steps"].items():
            step_meta[sname] = (s["reactant"], s["product"])
            if s["barrier"] is not None:
                step_bar.setdefault(sname, []).append(s["barrier"])
            if s["delta_e"] is not None:
                step_de.setdefault(sname, []).append(s["delta_e"])

    for name, vals in state_vals.items():
        results.node_energies[name] = aggregate(vals, cfg.search.energy_thresh)
    for sname, (r, pr) in step_meta.items():
        results.edges.append({
            "name": sname, "reactant": r, "product": pr,
            "barrier": aggregate(step_bar.get(sname, []), cfg.search.energy_thresh),
            "delta_e": aggregate(step_de.get(sname, []), cfg.search.energy_thresh),
        })
    return results


def run(cfg: Config, log=print) -> Results:
    """Convenience: run all seeds in-process and aggregate."""
    partials = [run_one_seed(cfg, seed, log=log) for seed in cfg.search.seeds]
    return aggregate_partials(cfg, partials)


def write_outputs(cfg: Config, results: Results, log=print) -> Path:
    outdir = Path(cfg.outdir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    cfg.snapshot(outdir / "config.snapshot.yaml")

    ref = results.node_energies[results.pathway[0]].mean

    # dashed "supply" edges (+O* / +H*) bridge states of different stoichiometry
    # so the branching graph is connected. They carry no reaction barrier.
    edges = list(results.edges)
    for a, b in results.links:
        if not g_has_edge(results.edges, a, b):
            edges.append({"name": f"{a}->{b}", "reactant": a, "product": b,
                          "kind": "supply"})

    g = build_graph(results.node_energies, edges, energy_ref=ref)
    to_json(g, outdir / "graph.json")
    to_csv(g, outdir / "nodes.csv", outdir / "edges.csv")
    title = f"{cfg.substrate} -> {cfg.target} on {cfg.slab.element}"
    draw_profile(g, outdir / "graph.png", title=title)          # energy profile
    draw_graph(g, outdir / "graph_network.png", title=title)    # node/DAG view

    # substrate x intermediate energy map (one row per substrate; here 1 substrate)
    cols = results.pathway
    row = [results.node_energies[c].mean - ref for c in cols]
    energy_map(np.array([row]), [cfg.substrate], cols, outdir / "energy_map.png")
    with open(outdir / "energy_map.csv", "w") as f:
        f.write("substrate," + ",".join(cols) + "\n")
        f.write(cfg.substrate + "," + ",".join(f"{v:.4f}" for v in row) + "\n")

    summary = {
        "name": cfg.name,
        "substrate": cfg.substrate, "target": cfg.target,
        "backend": cfg.mlip.backend, "seeds": cfg.search.seeds,
        "pathway": results.pathway,
        "nodes": {k: v.as_dict() for k, v in results.node_energies.items()},
        "edges": [
            {"name": e["name"], "reactant": e["reactant"], "product": e["product"],
             "barrier": e["barrier"].as_dict(), "delta_e": e["delta_e"].as_dict()}
            for e in results.edges
        ],
        "warnings": results.warnings,
    }
    (outdir / "results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote outputs to {outdir}")
    return outdir
