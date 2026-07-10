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

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calculators import check_supported, make_calculator
from .config import Config
from .graph import build_graph, to_csv, to_json
from .network import Network, StateSpec, build_network
from . import provenance, render
from .neb import neb_barrier
from .relax import pre_relax, relax
from .structures import (
    default_lattice,
    equilibrium_lattice,
    rattle_adsorbate,
    symbols_of,
)
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
    models: list[str] = field(default_factory=list)  # distinct model tags used
    structures: dict = field(default_factory=dict)  # state -> relaxed Atoms (not serialised)
    lattice: dict = field(default_factory=dict)  # model tag -> relaxed lattice constant (A)
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


def run_one_seed(cfg: Config, seed: int, log=print, collect: dict | None = None) -> dict:
    """Run the whole network for a single seed -> a JSON-serialisable partial.

    If ``collect`` (a dict) is passed, the lowest-energy relaxed ``Atoms`` seen
    for each state is stored in it (for structure thumbnails). Kept out of the
    JSON partial since ``Atoms`` are not serialisable.
    """
    net = build_network(cfg.slab, cfg.network, cfg.reagents)
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
            if collect is not None:
                prev = collect.get(st.name)
                if prev is None or res.energy < prev[0]:
                    collect[st.name] = (res.energy, res.atoms.copy())
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
    """Combine partials (over seeds and/or models) into mean +/- spread.

    Each partial is referenced to the substrate (root) state *before* pooling, so
    energies from different models -- which have different absolute offsets -- are
    combined on a common relative scale.  The resulting spread therefore captures
    both seed and model uncertainty.
    """
    net = build_network(cfg.slab, cfg.network, cfg.reagents)
    order = net.order()
    ref_state = order[0]
    results = Results(pathway=order, links=list(net.links))

    state_vals: dict[str, list[float]] = {}
    step_bar: dict[str, list[float]] = {}
    step_de: dict[str, list[float]] = {}
    step_meta: dict[str, tuple[str, str]] = {}
    models: set[str] = set()

    for p in partials:
        models.add(p.get("model", cfg.mlip.backend))
        results.warnings.extend(p.get("warnings", []))
        ref = p["states"].get(ref_state)  # per-partial reference removes offset
        for name, e in p["states"].items():
            state_vals.setdefault(name, []).append(e - ref if ref is not None else e)
        for sname, s in p["steps"].items():
            step_meta[sname] = (s["reactant"], s["product"])
            if s["barrier"] is not None:  # barriers are already relative
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
    results.models = sorted(models)
    return results


def run(cfg: Config, log=print) -> Results:
    """Run every (model x seed) combination in-process and aggregate.

    Model uncertainty (from ``mlip.models``) and seed uncertainty are pooled into
    one mean +/- spread per level and barrier.
    """
    partials: list[dict] = []
    structures: dict[str, tuple] = {}
    lattice: dict[str, float] = {}
    specs = cfg.mlip.specs()
    for si, (backend, model) in enumerate(specs):
        tag = f"{backend}:{model}" if model else backend
        c = copy.deepcopy(cfg)
        c.mlip.backend, c.mlip.model, c.mlip.models = backend, model, []
        # relax the bulk lattice constant to THIS potential (removes epitaxial strain)
        if c.slab.relax_lattice and c.slab.a is None:
            a0 = equilibrium_lattice(c.slab.element, lambda: make_calculator(c.mlip))
            a_ref = default_lattice(c.slab.element)
            log(f"[{tag}] relaxed lattice a={a0:.4f} A "
                f"(default {a_ref:.4f} A, strain {(a_ref/a0 - 1) * 100:+.2f}%)")
            c.slab.a = a0
            lattice[tag] = a0
        for seed in cfg.search.seeds:
            log(f"=== model={tag} seed={seed} ===")
            # capture geometries only from the first model (single reference set)
            p = run_one_seed(c, seed, log=log,
                             collect=structures if si == 0 else None)
            p["model"] = tag
            partials.append(p)
    results = aggregate_partials(cfg, partials)
    results.structures = {k: v[1] for k, v in structures.items()}
    results.lattice = lattice
    return results


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
    cap = provenance.caption(cfg, results)
    pmeta = provenance.png_metadata(cfg, results)
    # two versions: clean (no visible text) and annotated (visible footer).
    # Both embed the provenance as PNG tEXt so it travels with the pixels.
    draw_profile(g, outdir / "graph.png", title=title, caption=cap,
                 show_caption=False, png_meta=pmeta)
    draw_profile(g, outdir / "graph_annotated.png", title=title, caption=cap,
                 show_caption=True, png_meta=pmeta)
    draw_graph(g, outdir / "graph_network.png", title=title)       # node/DAG view
    (outdir / "methods.md").write_text(provenance.methods_text(cfg, results))

    # structure thumbnails: a gallery + with-thumbnail variants of both graphs
    if results.structures:
        sample = next(iter(results.structures.values()))
        n_slab = int(sample.info.get("n_slab", len(sample)))
        window = render.view_window(results.structures, n_slab)
        thumbs = {name: render.thumb_array(atoms, n_slab, window)
                  for name, atoms in results.structures.items()}
        render.gallery(results.structures, results.node_energies,
                       outdir / "gallery.png", n_slab)
        draw_profile(g, outdir / "graph_thumbs.png", title=title, caption=cap,
                     show_caption=True, png_meta=pmeta, thumbs=thumbs)
        draw_graph(g, outdir / "graph_network_thumbs.png", title=title, thumbs=thumbs)

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
        "backend": cfg.mlip.backend, "models": results.models,
        "seeds": cfg.search.seeds,
        "n_samples": max(1, len(results.models)) * len(cfg.search.seeds),
        "relaxed_lattice_A": results.lattice,
        "energy_reference": f"relative to substrate state '{results.pathway[0]}'",
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
