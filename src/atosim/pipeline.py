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

from .calculators import check_supported, make_calculator, resolve_backend
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


def _build_net(cfg: Config, log=lambda *a, **k: None) -> Network:
    """Build the reaction network for this run.

    For ``network: auto`` the exploration is bounded by ``cfg.auto`` and, if
    ``cfg.auto.prune_energy`` is set, high-energy branches are dropped by a
    deterministic rough-energy pass so every seed prunes to the same network.
    """
    net = build_network(cfg.slab, cfg.network, cfg.reagents, cfg.substrate,
                        cfg.target, max_extra=cfg.auto.max_extra,
                        max_states=cfg.auto.max_states)
    if cfg.network == "auto" and cfg.auto.prune_energy is not None:
        from .explore import prune_by_rough_energy
        net = prune_by_rough_energy(net, lambda: make_calculator(cfg.mlip),
                                    cfg.target, cfg.auto.prune_energy, log=log)
    return net


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


def relax_states(cfg: Config, seed: int, log=print) -> dict[str, float]:
    """Relax every network state for one seed (no NEB) -> {state: energy}.

    The cheap building block for a cross-model state-energy comparison: it skips
    the expensive barrier search and only reports the relaxed adsorption energy
    of each state.
    """
    net = _build_net(cfg, log)
    slab = net.slab()
    n_slab = slab.info["n_slab"]
    all_syms: set[str] = set()
    for st in net.states().values():
        all_syms |= symbols_of(st.build(slab))
    check_supported(all_syms, cfg.mlip)
    out: dict[str, float] = {}
    for name, st in net.states().items():
        res, geo = _relax_state(st, slab, n_slab, cfg, seed)
        out[name] = res.energy
        flag = "" if geo.ok else f"  (geometry: {'; '.join(geo.reasons)})"
        log(f"  [{name}] seed={seed}: E={res.energy:.3f} eV{flag}")
    return out


# element -> gas-phase reference molecule for the chemical potential mu = E/natoms
_GAS_REF = {"H": "H2", "N": "N2", "O": "O2"}


def _reference_energies(cfg: Config, slab, elements: set[str], log=print):
    """Clean-slab energy + per-element gas-phase chemical potential (this potential).

    ``mu[X] = E(gas X2) / 2`` and the clean-slab energy anchor a *formation*
    energy that cancels each potential's per-atom reference convention -- the
    only way a cross-model comparison of composition-changing states is physical.
    """
    from ase.build import molecule

    e_slab = relax(slab.copy(), make_calculator(cfg.mlip),
                   fmax=cfg.search.fmax, max_steps=cfg.search.max_steps).energy
    log(f"  ref: clean slab E={e_slab:.3f} eV")
    mu: dict[str, float] = {}
    for el in sorted(elements):
        name = _GAS_REF.get(el)
        if name is None:
            raise ValueError(f"no gas reference for element {el!r} "
                             "(formation referencing supports H, N, O)")
        mol = molecule(name)
        mol.cell = [12.0, 12.0, 12.0]
        mol.center()
        mol.pbc = True
        e = relax(mol, make_calculator(cfg.mlip), fmax=0.03, max_steps=200).energy
        mu[el] = e / len(mol)
        log(f"  ref: mu[{el}]={mu[el]:.3f} eV (1/2 {name})")
    return e_slab, mu


def run_states(cfg: Config, log=print, reference: str = "formation") -> dict:
    """Relax states for every seed; return per-state energy samples.

    One backend per call (the active env's), so several of these -- run in each
    backend's venv -- combine into a cross-model comparison via
    :func:`atosim.viz.compare_boxplot`.

    ``reference="formation"`` (default) reports each state as a formation energy
    vs gas-phase references + clean slab, computed *in this potential*, so
    composition-changing states are comparable across potentials. ``"substrate"``
    just subtracts the root state (only meaningful within one potential).
    """
    net = _build_net(cfg, log)
    root = net.order()[0]
    resolved = resolve_backend(cfg.mlip.backend)
    tag = f"{resolved}:{cfg.mlip.model}" if cfg.mlip.model else resolved
    states = net.states()

    e_slab = mu = None
    if reference == "formation":
        elements: set[str] = set()
        for st in states.values():
            elements |= set(st.adsorbate_counts())
        e_slab, mu = _reference_energies(cfg, net.slab(), elements, log)

    per_state: dict[str, list[float]] = {}
    for seed in cfg.search.seeds:
        log(f"=== {tag} seed={seed} ===")
        e = relax_states(cfg, seed, log=log)
        if reference == "formation":
            for name, val in e.items():
                counts = states[name].adsorbate_counts()
                ref = e_slab + sum(counts[el] * mu[el] for el in counts)
                per_state.setdefault(name, []).append(val - ref)
        else:
            r = e.get(root)
            for name, val in e.items():
                per_state.setdefault(name, []).append(val - r if r is not None else val)
    return {"model": tag, "reference": reference, "network": cfg.network,
            "substrate": cfg.substrate, "target": cfg.target, "order": net.order(),
            "seeds": list(cfg.search.seeds), "states": per_state}


def run_one_seed(cfg: Config, seed: int, log=print, collect: dict | None = None) -> dict:
    """Run the whole network for a single seed -> a JSON-serialisable partial.

    If ``collect`` (a dict) is passed, the lowest-energy relaxed ``Atoms`` seen
    for each state is stored in it (for structure thumbnails). Kept out of the
    JSON partial since ``Atoms`` are not serialisable.
    """
    net = _build_net(cfg, log)
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
                retries=cfg.search.neb_retries,
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
    net = _build_net(cfg)
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
        resolved = resolve_backend(backend)  # `auto` -> best installed ML backend
        if resolved != backend:
            log(f"backend: auto -> {resolved} (best installed ML potential)")
        backend = resolved
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
        backend, warn = render.resolve_backend(cfg.render.backend)
        if warn:
            log(f"warning: {warn}")
            results.warnings.append(warn)
        rk = {"backend": backend, "width": cfg.render.width,
              "bonds": cfg.render.bonds}
        sample = next(iter(results.structures.values()))
        n_slab = int(sample.info.get("n_slab", len(sample)))
        window = render.view_window(results.structures, n_slab)
        thumbs = {name: render.thumb_array(atoms, n_slab, window, **rk)
                  for name, atoms in results.structures.items()}
        render.gallery(results.structures, results.node_energies,
                       outdir / "gallery.png", n_slab, **rk)
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
        "backend": resolve_backend(cfg.mlip.backend), "models": results.models,
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
