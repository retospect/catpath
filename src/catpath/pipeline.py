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

from . import provenance, render
from .calculators import check_supported, make_calculator, resolve_backend
from .config import Config
from .graph import build_graph, to_csv, to_json
from .neb import neb_barrier
from .network import Network, StateSpec, build_network
from .relax import pre_relax, relax
from .structures import (
    default_lattice,
    equilibrium_lattice,
    rattle_adsorbate,
    symbols_of,
)
from .uncertainty import Estimate, aggregate
from .validate import binding_site_ok, geometry_ok
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
    ads_barriers: dict[str, float] = field(default_factory=dict)  # state -> adsorption barrier (eV)


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
# tether used to try to reseat a desorbed adsorbate before giving up on it. The
# live values are ``cfg.search.bind_tether_{k,rt}`` / ``bind_reseat_attempts``;
# these module constants are the fallback defaults (see config.SearchConfig).
BIND_TETHER_K = 7.5  # eV/A^2 -- Hookean spring constant for the reseat tether
BIND_TETHER_RT = 2.0  # A -- rest length; must sit inside the real M-adsorbate bond
BIND_RESEAT_ATTEMPTS = 3  # reseat attempts (fresh rattle each time) before giving up


def _detached(geo) -> bool:
    """True if `geometry_ok` failed because an adsorbate fragment desorbed (not just a clash)."""
    return any("detached" in r for r in geo.reasons)


def _binding_site(state: StateSpec, res, geo, n_slab: int) -> tuple[bool, list[str]]:
    """Verify a BOUND endpoint binds through its ``*``-designated atom.

    Skipped (returns ok) when the endpoint is already detached (that failure is
    reported separately) — the check only distinguishes a bound-but-flipped
    geometry from a correctly-seated one. Returns ``(ok, reasons)`` where reasons
    carry the ``wrong-site`` trust-gate substring."""
    if not geo.ok:
        return True, []
    return binding_site_ok(res.atoms, n_slab, state.placement_heights())


def _tether_constraints(atoms, n_slab: int, k: float, rt: float) -> list:
    """One Hookean per adsorbate atom, anchored to its nearest slab atom."""
    pos = atoms.get_positions()
    slab_pos = pos[:n_slab]
    cons = []
    for i in range(n_slab, len(atoms)):
        j = int(np.argmin(np.linalg.norm(slab_pos - pos[i], axis=1)))
        cons.append(Hookean(a1=i, a2=j, k=k, rt=rt))
    return cons


# Default dissolving-tether ramp (fractions of ``k``) when the config predates
# ``bind_tether_ramp``. Ends at 0.0 — the unconstrained relax the endpoint must
# survive to count as bound.
BIND_TETHER_RAMP = [1.0, 0.5, 0.25, 0.1, 0.0]
BIND_ADS_BARRIER_MAX = 0.75  # eV -- above this, a reseat is activated adsorption


def _dissolve_reseat(start, n_slab, cfg, k, rt, ramp):
    """Relax through a DISSOLVING tether: full-k pull-in, then k scaled down the
    ``ramp`` to 0, so the fragment settles into the well adiabatically instead of
    springing back off when the spring is dropped in one step.

    Returns ``(final_result, geometry_report)`` — the final entry of ``ramp`` is
    0.0, so the reported geometry is from a fully UNCONSTRAINED relax (no residual
    spring holding a non-binding fragment in place).
    """
    res = None
    atoms = start
    base = [c for c in start.constraints if not isinstance(c, Hookean)]
    for frac in ramp:
        atoms = atoms.copy()
        cons = list(base)
        if frac > 0.0:
            cons += _tether_constraints(atoms, n_slab, k * frac, rt)
        atoms.set_constraint(cons)
        res = relax(atoms, make_calculator(cfg.mlip),
                    fmax=cfg.search.fmax, max_steps=cfg.search.max_steps)
        atoms = res.atoms
    geo = geometry_ok(res.atoms, n_slab)
    return res, geo


def _adsorption_barrier(desorbed, bound, cfg, n_images: int = 5) -> float:
    """The adsorption barrier along the desorbed -> bound coordinate: the PES
    energy the fragment must climb ABOVE its desorbed asymptote to reach the well
    (0 if the approach is monotonically downhill = barrierless chemisorption).

    Measured with a short, loose climbing-image NEB (IDPP-interpolated) between
    the (untethered) desorbed geometry and the reseated bound geometry. A RELAXED
    band is essential here: a straight-line interpolation clips a laterally-offset
    fragment through a surface atom and reports a large spurious hump; the NEB lets
    each image find its minimum-energy lateral position, so the barrier reflects
    the true minimum-energy approach. No Hookean penalty enters the reported PES
    (the tether only shaped the reseat; the barrier is measured spring-free).
    Only runs on the desorb-then-reseat branch (already the expensive path), so it
    does not add cost to endpoints that bind on the first relax.
    """
    if len(desorbed) != len(bound):
        return 0.0
    bar = neb_barrier(desorbed, bound,
                      make_calc=lambda: make_calculator(cfg.mlip),
                      n_images=n_images, fmax=0.2, max_steps=40, retries=0)
    return max(0.0, bar.barrier)


def _relax_state_bound(state: StateSpec, slab, n_slab, cfg: Config, seed: int):
    """Like `_relax_state`, but gates on whether the adsorbate actually binds.

    If the plain relax leaves a fragment detached from the slab, try a
    DISSOLVING-tether reseat -- tether each adsorbate atom to its nearest slab
    atom (rest length ``bind_tether_rt`` sits INSIDE the real bond, so the tether
    pulls a desorbing fragment back into the chemisorption well rather than
    leashing it at arm's length), then relax through a decreasing schedule of
    spring constants (``bind_tether_ramp``) down to an unconstrained relax, so the
    fragment settles adiabatically instead of re-desorbing on an abrupt release --
    up to ``bind_reseat_attempts`` times (fresh pose each) before concluding it
    genuinely does not bind. The per-stage pull-in energies give an ADSORPTION
    BARRIER; a reseat that only binds by crossing more than ``bind_ads_barrier_max``
    is activated adsorption and is reported non-binding (untrusted).

    Returns ``(result, geometry_report, bound, diag)``, where ``diag`` traces the
    plain relax + each reseat attempt (anchor/worst heights, adsorption barrier)
    for downstream analysis.
    """
    k = getattr(cfg.search, "bind_tether_k", BIND_TETHER_K)
    rt = getattr(cfg.search, "bind_tether_rt", BIND_TETHER_RT)
    attempts = getattr(cfg.search, "bind_reseat_attempts", BIND_RESEAT_ATTEMPTS)
    ramp = getattr(cfg.search, "bind_tether_ramp", None) or BIND_TETHER_RAMP
    ads_max = getattr(cfg.search, "bind_ads_barrier_max", BIND_ADS_BARRIER_MAX)

    res, geo = _relax_state(state, slab, n_slab, cfg, seed)
    diag = {"state": state.name, "rt": rt, "k": k,
            "plain_anchor": round(geo.anchor_height, 2),
            "plain_worst": round(geo.adsorbate_height, 2),
            "attempts": [], "reseated": False, "ads_barrier": None}
    if geo.ok:
        return res, geo, True, diag
    if not _detached(geo):
        # some other geometry failure (e.g. an adsorbate-adsorbate clash) --
        # not what a binding tether fixes.
        return res, geo, False, diag

    desorbed_atoms = res.atoms  # the untethered detached geometry = adsorption ref
    best_res, best_geo = res, geo
    for attempt in range(attempts):
        reseat_seed = seed * 1000 + attempt + 1  # different seed each attempt
        start = rattle_adsorbate(state.build(slab), n_slab, seed=reseat_seed, amplitude=0.15)
        cleaned = pre_relax(start, make_calculator(cfg.mlip))

        res2, geo2 = _dissolve_reseat(cleaned, n_slab, cfg, k, rt, ramp)
        ads_barrier = _adsorption_barrier(desorbed_atoms, res2.atoms, cfg) if geo2.ok else 0.0
        # activated adsorption: bound only by climbing a real barrier -> the site
        # does not spontaneously bind this fragment. Treat as still-detached.
        activated = geo2.ok and ads_barrier > ads_max
        diag["attempts"].append({"anchor": round(geo2.anchor_height, 2),
                                 "worst": round(geo2.adsorbate_height, 2),
                                 "ok": bool(geo2.ok and not activated),
                                 "ads_barrier": round(ads_barrier, 3)})
        if geo2.ok and not activated:
            diag["reseated"] = True
            diag["ads_barrier"] = round(ads_barrier, 3)
            return res2, geo2, True, diag
        if activated:
            # record the barrier that made it untrusted (max over attempts seen)
            prev = diag["ads_barrier"] or 0.0
            diag["ads_barrier"] = round(max(prev, ads_barrier), 3)
        # keep the closest-approach attempt as the reported (still-unbound) outcome
        if geo2.anchor_height < best_geo.anchor_height:
            best_res, best_geo = res2, geo2

    return best_res, best_geo, False, diag


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
    ads_barriers: dict[str, float] = {}  # per-state adsorption barrier (reseat pull-in)

    for step in net.steps:
        log(f"[{step.name}] seed={seed}: relaxing endpoints")
        if cfg.search.bind_preflight:
            r_res, r_geo, r_bound, r_diag = _relax_state_bound(step.reactant, slab, n_slab, cfg, seed)
            p_res, p_geo, p_bound, p_diag = _relax_state_bound(step.product, slab, n_slab, cfg, seed)
        else:  # exact pre-preflight behavior: NEB always runs regardless of geometry
            r_res, r_geo = _relax_state(step.reactant, slab, n_slab, cfg, seed)
            p_res, p_geo = _relax_state(step.product, slab, n_slab, cfg, seed)
            r_bound = p_bound = True
            r_diag = p_diag = None
        # bond-site (*) identity: only meaningful on a bound endpoint (a detached
        # one is already untrusted) and only when we have the StateSpec placement.
        r_site_ok, r_site_reasons = _binding_site(step.reactant, r_res, r_geo, n_slab)
        p_site_ok, p_site_reasons = _binding_site(step.product, p_res, p_geo, n_slab)
        for st, res, geo, diag, site_reasons in (
                (step.reactant, r_res, r_geo, r_diag, r_site_reasons),
                (step.product, p_res, p_geo, p_diag, p_site_reasons)):
            # keep the lowest energy seen for a state within this seed
            states[st.name] = min(states.get(st.name, np.inf), res.energy)
            if collect is not None:
                prev = collect.get(st.name)
                if prev is None or res.energy < prev[0]:
                    collect[st.name] = (res.energy, res.atoms.copy())
            if diag is not None and diag.get("ads_barrier") is not None:
                # record the adsorption barrier the dissolving tether traversed
                # (max over states if a name recurs across steps)
                ab = float(diag["ads_barrier"])
                ads_barriers[st.name] = max(ads_barriers.get(st.name, 0.0), ab)
            if not geo.ok:
                warnings.append(f"{st.name} seed={seed} geometry: {'; '.join(geo.reasons)}")
            elif site_reasons:
                # bound, but through the wrong atom — the * designates a different
                # binder. Keep the "wrong-site" trust-gate substring.
                warnings.append(
                    f"{st.name} seed={seed} {'; '.join(site_reasons)}")
            elif diag is not None and diag.get("reseated"):
                # INFO diagnostic: the reseat rescued a plain-relax desorption.
                # MUST avoid the trust-gate substrings ("detached" / "NEB not
                # converged" / "wrong-site") so a rescued (now-bound) endpoint
                # stays trusted.
                last = diag["attempts"][-1]
                ab = diag.get("ads_barrier")
                ab_txt = f", adsorption barrier {ab} eV" if ab is not None else ""
                warnings.append(
                    f"{st.name} seed={seed} RESEATED ok: plain closest "
                    f"{diag['plain_anchor']}A -> bound closest {last['anchor']}A "
                    f"(tether rt={diag['rt']} k={diag['k']}, "
                    f"{len(diag['attempts'])} attempt(s){ab_txt})")
            if not res.converged:
                warnings.append(f"{st.name} seed={seed} not converged")

        log(f"[{step.name}] seed={seed}: NEB")
        entry = {"reactant": step.reactant.name, "product": step.product.name,
                 "barrier": None, "delta_e": None}
        if not (r_bound and p_bound and r_site_ok and p_site_ok):
            # endpoint binding pre-flight failed -- an honest "no barrier" beats a
            # climbing-image NEB anchored on a desorbed OR mis-bound endpoint
            # (garbage geometry in, garbage barrier out). Wrong-site endpoints were
            # already warned above; here we just skip the NEB for them too.
            for st, geo, bound, diag in ((step.reactant, r_geo, r_bound, r_diag),
                                         (step.product, p_geo, p_bound, p_diag)):
                if not bound:
                    trace = ""
                    activated = diag is not None and diag.get("ads_barrier") is not None
                    if diag is not None:
                        tries = "; ".join(
                            f"a{i + 1} closest {a['anchor']}A worst {a['worst']}A"
                            f" ads_barrier {a['ads_barrier']}eV"
                            for i, a in enumerate(diag["attempts"])) or "no reseat run"
                        trace = (f" [preflight rt={diag['rt']} k={diag['k']}: plain closest "
                                 f"{diag['plain_anchor']}A worst {diag['plain_worst']}A; "
                                 f"reseat {tries}]")
                    if activated:
                        # bound only by crossing a real adsorption barrier -> genuine
                        # activated adsorption, not spontaneous chemisorption. Keep the
                        # trust-gate substring "detached" so the barrier stays untrusted.
                        warnings.append(
                            f"{st.name} seed={seed} INFEASIBLE: activated adsorption — "
                            f"reseats only by climbing {diag['ads_barrier']} eV "
                            f"(> bind_ads_barrier_max); treated as detached / "
                            f"non-binding{trace}")
                    else:
                        warnings.append(
                            f"{st.name} seed={seed} INFEASIBLE: adsorbate does not bind — "
                            f"detached, desorbs to {geo.adsorbate_height:.1f} A (closest "
                            f"{geo.anchor_height:.1f} A) after restrained relax; try a "
                            f"different site or dopant{trace}")
            log(f"[{step.name}] seed={seed}: NEB skipped (endpoint unbound or mis-bound)")
            steps[step.name] = entry
            continue
        try:
            bar = neb_barrier(
                r_res.atoms, p_res.atoms,
                make_calc=lambda: make_calculator(cfg.mlip),
                n_images=cfg.search.neb_images,
                fmax=cfg.search.neb_fmax, max_steps=cfg.search.neb_max_steps,
                retries=cfg.search.neb_retries,
                n_slab=n_slab,
                tether={"k": getattr(cfg.search, "bind_tether_k", BIND_TETHER_K),
                        "rt": getattr(cfg.search, "bind_tether_rt", BIND_TETHER_RT),
                        "ramp": getattr(cfg.search, "bind_tether_ramp", None)
                        or BIND_TETHER_RAMP},
            )
            entry["barrier"] = bar.barrier
            entry["delta_e"] = bar.delta_e
            if not bar.converged:
                warnings.append(f"{step.name} seed={seed} NEB not converged")
            if bar.desorbed_images:
                # a mid-band image floated off the slab -> the reported TS is not
                # on a surface path. Keep the "detached" trust-gate substring.
                warnings.append(
                    f"{step.name} seed={seed} NEB mid-band desorption: "
                    f"{bar.desorbed_images} interior image(s) detached from slab "
                    f"— barrier off a non-surface path")
        except Exception as e:  # abandon this seed's edge, keep going
            warnings.append(f"{step.name} seed={seed} NEB failed: {e}")
        steps[step.name] = entry

    return {"seed": seed, "states": states, "steps": steps, "warnings": warnings,
            "ads_barriers": ads_barriers}


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
        # adsorption barrier is a property of the reseat, not a stochastic energy:
        # keep the worst (max) seen across seeds/models as the conservative report.
        for name, ab in p.get("ads_barriers", {}).items():
            results.ads_barriers[name] = max(results.ads_barriers.get(name, 0.0), float(ab))
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
        # per-state pics dumped individually (top+side, same fixed camera/window as
        # the gallery) so each adsorbed state is inspectable on its own.
        import matplotlib.image as _mpimg
        states_dir = outdir / "states"
        states_dir.mkdir(exist_ok=True)
        for name, arr in thumbs.items():
            safe = name.replace("/", "_").replace(" ", "_")
            _mpimg.imsave(states_dir / f"{safe}.png", np.asarray(arr))
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
