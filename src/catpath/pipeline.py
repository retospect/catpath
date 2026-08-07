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
from ase.constraints import Hookean

from . import electrochem
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
    # Injected slab (the precis `structure` seam) travels on cfg as a runtime
    # side-channel — not a dataclass field, so it never leaks into
    # to_dict/snapshot/content_key. Every build path funnels through here, so
    # stamping it once reaches all `net.slab()` call sites.
    net.prebuilt_slab = getattr(cfg, "_prebuilt_slab", None)
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


# Endpoint binding pre-flight (see `_relax_state_bound`): a one-sided harmonic
# tether used to try to reseat a desorbed adsorbate before giving up on it.
BIND_TETHER_K = 5.0  # eV/A^2 -- Hookean spring constant for the reseat tether
BIND_TETHER_RT = 2.6  # A -- tether rest length; only pulls beyond this distance
BIND_RESEAT_ATTEMPTS = 1  # reseat attempts (fresh rattle each time) before giving up


def _detached(geo) -> bool:
    """True if `geometry_ok` failed because an adsorbate atom desorbed (not just a clash)."""
    return any("detached" in r for r in geo.reasons)


def _tether_constraints(atoms, n_slab: int) -> list:
    """One Hookean per adsorbate atom, anchored to its nearest slab atom."""
    pos = atoms.get_positions()
    slab_pos = pos[:n_slab]
    cons = []
    for i in range(n_slab, len(atoms)):
        j = int(np.argmin(np.linalg.norm(slab_pos - pos[i], axis=1)))
        cons.append(Hookean(a1=i, a2=j, k=BIND_TETHER_K, rt=BIND_TETHER_RT))
    return cons


def _relax_state_bound(state: StateSpec, slab, n_slab, cfg: Config, seed: int):
    """Like `_relax_state`, but gates on whether the adsorbate actually binds.

    If the plain relax leaves the adsorbate detached from the slab, try a
    restrained-then-released reseat -- tether each adsorbate atom to its nearest
    slab atom, relax, drop the tether, relax again unconstrained -- up to
    ``BIND_RESEAT_ATTEMPTS`` times before concluding it genuinely does not bind.
    Returns ``(result, geometry_report, bound)``.
    """
    res, geo = _relax_state(state, slab, n_slab, cfg, seed)
    if geo.ok:
        return res, geo, True
    if not _detached(geo):
        # some other geometry failure (e.g. an adsorbate-adsorbate clash) --
        # not what a binding tether fixes.
        return res, geo, False

    for attempt in range(BIND_RESEAT_ATTEMPTS):
        reseat_seed = seed * 1000 + attempt + 1  # different seed each attempt
        start = rattle_adsorbate(state.build(slab), n_slab, seed=reseat_seed, amplitude=0.15)
        start.set_constraint(list(start.constraints) + _tether_constraints(start, n_slab))

        cleaned = pre_relax(start, make_calculator(cfg.mlip))
        res_teth = relax(cleaned, make_calculator(cfg.mlip),
                         fmax=cfg.search.fmax, max_steps=cfg.search.max_steps)

        released = res_teth.atoms
        released.set_constraint([c for c in released.constraints
                                 if not isinstance(c, Hookean)])
        res2 = relax(released, make_calculator(cfg.mlip),
                     fmax=cfg.search.fmax, max_steps=cfg.search.max_steps)
        geo2 = geometry_ok(res2.atoms, n_slab)
        res, geo = res2, geo2  # keep the best (last) attempt as the reported outcome
        if geo2.ok:
            return res2, geo2, True

    return res, geo, False


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


def _gas_energy(cfg: Config, name: str) -> float:
    """Relaxed total energy of a gas-phase molecule in a large box (this potential)."""
    from ase.build import molecule

    mol = molecule(name)
    mol.cell = [12.0, 12.0, 12.0]
    mol.center()
    mol.pbc = True
    return relax(mol, make_calculator(cfg.mlip), fmax=0.03, max_steps=200).energy


def _reference_energies(cfg: Config, slab, elements: set[str], log=print):
    """Clean-slab energy + per-element gas-phase chemical potential (this potential).

    Anchors a *formation* energy that cancels each potential's per-atom reference
    convention -- the only way a cross-model comparison of composition-changing
    states is physical.  References: ``mu_H = 1/2 E(H2)``, ``mu_N = 1/2 E(N2)``,
    and ``mu_O = E(H2O) - 2 mu_H`` (water, NOT 1/2 O2 -- O2 is a notoriously bad
    reference under PBE-trained potentials, and water keeps O comparable).
    """
    e_slab = relax(slab.copy(), make_calculator(cfg.mlip),
                   fmax=cfg.search.fmax, max_steps=cfg.search.max_steps).energy
    log(f"  ref: clean slab E={e_slab:.3f} eV")
    mu: dict[str, float] = {}
    # O is referenced through water, which needs mu_H, so compute H whenever O is asked.
    if "H" in elements or "O" in elements:
        mu["H"] = _gas_energy(cfg, "H2") / 2
        log(f"  ref: mu[H]={mu['H']:.3f} eV (1/2 H2)")
    if "N" in elements:
        mu["N"] = _gas_energy(cfg, "N2") / 2
        log(f"  ref: mu[N]={mu['N']:.3f} eV (1/2 N2)")
    if "O" in elements:
        mu["O"] = _gas_energy(cfg, "H2O") - 2 * mu["H"]
        log(f"  ref: mu[O]={mu['O']:.3f} eV (H2O - 2 mu_H)")
    missing = elements - set(mu)
    if missing:
        raise ValueError(f"no gas reference for element(s) {sorted(missing)} "
                         "(formation referencing supports H, N, O)")
    return e_slab, mu


def run_states(cfg: Config, log=print, reference: str = "formation") -> dict:
    """Relax states for every seed; return per-state energy samples.

    One backend per call (the active env's), so several of these -- run in each
    backend's venv -- combine into a cross-model comparison via
    :func:`catpath.viz.compare_boxplot`.

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


def run_barriers(cfg: Config, log=print) -> dict:
    """Run NEB for every elementary step and every seed -> per-step Ea samples.

    Activation energies are composition-conserving (NEB runs between same-atom
    endpoints), so they are directly comparable across potentials -- no formation
    referencing needed.  Reuses :func:`run_one_seed` (which relaxes endpoints and
    runs the climbing-image NEB), keeping just the barriers.
    """
    net = _build_net(cfg, log)
    resolved = resolve_backend(cfg.mlip.backend)
    tag = f"{resolved}:{cfg.mlip.model}" if cfg.mlip.model else resolved
    steps: dict[str, dict] = {}
    warnings: list[str] = []
    for seed in cfg.search.seeds:
        part = run_one_seed(cfg, seed, log=log)
        warnings.extend(part.get("warnings", []))
        for sname, s in part["steps"].items():
            d = steps.setdefault(sname, {"barrier": [], "delta_e": [],
                                         "reactant": s["reactant"], "product": s["product"]})
            if s["barrier"] is not None:
                d["barrier"].append(s["barrier"])
            if s["delta_e"] is not None:
                d["delta_e"].append(s["delta_e"])
    return {"model": tag, "network": cfg.network, "substrate": cfg.substrate,
            "target": cfg.target, "order": net.order(),
            "seeds": list(cfg.search.seeds), "steps": steps, "warnings": warnings}


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
        if cfg.search.bind_preflight:
            r_res, r_geo, r_bound = _relax_state_bound(step.reactant, slab, n_slab, cfg, seed)
            p_res, p_geo, p_bound = _relax_state_bound(step.product, slab, n_slab, cfg, seed)
        else:  # exact pre-preflight behavior: NEB always runs regardless of geometry
            r_res, r_geo = _relax_state(step.reactant, slab, n_slab, cfg, seed)
            p_res, p_geo = _relax_state(step.product, slab, n_slab, cfg, seed)
            r_bound = p_bound = True
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
        if not (r_bound and p_bound):
            # endpoint binding pre-flight failed -- an honest "no barrier" beats a
            # climbing-image NEB anchored on a desorbed endpoint (garbage geometry
            # in, garbage barrier out).
            for st, geo, bound in ((step.reactant, r_geo, r_bound),
                                   (step.product, p_geo, p_bound)):
                if not bound:
                    warnings.append(
                        f"{st.name} seed={seed} INFEASIBLE: adsorbate does not bind — "
                        f"detached, desorbs to {geo.adsorbate_height:.1f} A after "
                        f"restrained relax; try a different site or dopant")
            log(f"[{step.name}] seed={seed}: NEB skipped (endpoint does not bind)")
            steps[step.name] = entry
            continue
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
        # relax the bulk lattice constant to THIS potential (removes epitaxial
        # strain) — skipped when a slab is injected: its geometry (and lattice)
        # is the caller's, so there is nothing to rebuild.
        injected = getattr(cfg, "_prebuilt_slab", None) is not None
        if c.slab.relax_lattice and c.slab.a is None and not injected:
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


def _apply_electrochemistry(g, cfg: Config, results: Results, log=print) -> dict | None:
    """CHE post-processing (docs/proposals/pathway-potential-lever.md, slice 1):
    stamp ``n_H`` (relative to the root state -- see :func:`electrochem.n_H_rel`)
    onto every node of ``g`` in place, and return the ``U_L``/``U_opt``/
    ``span_at_UL``/``span_at_Uopt`` scalar bundle for the substrate->target
    pathway. Pure post-processing over the energies already in ``g`` -- no new
    relax/NEB calls. ``None`` if there is no electrochemistry config, or no
    root->target path to compute a pathway span/limiting-potential over.
    """
    ec = cfg.electrochemistry
    if ec is None or not results.pathway:
        return None
    root = results.pathway[0]
    nodes = set(g.nodes())
    target = cfg.target if cfg.target in nodes else results.pathway[-1]

    for n, data in g.nodes(data=True):
        data["n_H"] = electrochem.n_H_rel(n, root)

    edges = list(g.edges())
    path = electrochem.reaction_path(edges, root, target)
    if len(path) < 2:
        log(f"electrochemistry: no root->target path ({root} -> {target}); "
            "skipping U_L/span")
        return None

    node_energy0 = {n: d["rel_energy"] for n, d in g.nodes(data=True)}
    edge_barrier = {(u, v): d.get("barrier", 0.0) for u, v, d in g.edges(data=True)}
    path_edges = list(zip(path, path[1:]))

    u_l = electrochem.limiting_potential(node_energy0, path_edges, root)
    u_opt, span_opt = electrochem.optimal_span_u(
        path, node_energy0, edge_barrier, root, window=ec.U_window)
    span_ul = (electrochem.energetic_span_u(path, node_energy0, edge_barrier, root, u_l)
              if u_l is not None else None)
    return {"U_L": u_l, "U_opt": u_opt, "span_at_UL": span_ul, "span_at_Uopt": span_opt}


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
    che = _apply_electrochemistry(g, cfg, results, log=log)  # stamps n_H onto g's nodes
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
    if che is not None:
        summary.update(che)  # U_L, U_opt, span_at_UL, span_at_Uopt (additive)
    (outdir / "results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote outputs to {outdir}")
    return outdir
